"""NetworkX <-> PyG conversion, graph-level conditioning, dataset assembly,
splits, and ``.tar.zst`` archive loading.

The GAT model in :mod:`mis.models` ingests both per-node features
(``data.x``: degree + normalized degree) and graph-level features
(``data.graph_x``: density, log|V|, 9-way family one-hot). This module
attaches both at preprocessing time so the model just reads them off.
"""
from __future__ import annotations

import math
import random
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm

from . import generators as gen
from . import solvers


# ---- graph-level conditioning ---------------------------------------------

# Canonical family ordering for one-hot. Add new families by appending.
FAMILY_ONEHOT = list(gen.FAMILIES)
GRAPH_FEATURE_DIM = 2 + len(FAMILY_ONEHOT)   # density, log|V|, family one-hot


def family_one_hot(family: str) -> torch.Tensor:
    """One-hot encoding of ``family`` over ``FAMILY_ONEHOT``. Unknown families
    are encoded as the all-zero vector."""
    v = torch.zeros(len(FAMILY_ONEHOT), dtype=torch.float)
    if family in FAMILY_ONEHOT:
        v[FAMILY_ONEHOT.index(family)] = 1.0
    return v


def graph_features(G: nx.Graph, family: str) -> torch.Tensor:
    """Compute the [graph_in] graph-level feature vector for one graph.

    Layout: [density, log|V|, *family_one_hot(family)] of length GRAPH_FEATURE_DIM.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = (2 * m) / max(1, n * (n - 1))
    feats = torch.tensor([density, math.log(max(1, n))], dtype=torch.float)
    return torch.cat([feats, family_one_hot(family)], dim=0)


# ---- single-graph conversion ----------------------------------------------

def nx_to_pyg(G: nx.Graph, family: str, *, mis_nodes: list | None = None,
              metadata: dict | None = None) -> Data:
    """Convert NetworkX graph to PyG Data with node + graph-level features.

    Sets:
        x          : [n, 2] (degree, degree/(n-1))
        edge_index : [2, 2|E|] bidirectional COO
        graph_x    : [1, GRAPH_FEATURE_DIM] (broadcast at forward time)
        y          : [n] binary, if mis_nodes given
        family, ... : extra metadata attributes
    """
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    n = G.number_of_nodes()
    deg = dict(G.degree())

    x = torch.zeros((n, 2), dtype=torch.float)
    for v in range(n):
        x[v, 0] = float(deg.get(v, 0))
        x[v, 1] = float(deg.get(v, 0)) / max(1, n - 1)

    edges = list(G.edges())
    if edges:
        us, vs = zip(*edges)
        edge_index = torch.tensor([list(us) + list(vs), list(vs) + list(us)],
                                  dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, num_nodes=n)
    data.graph_x = graph_features(G, family).unsqueeze(0)   # [1, graph_in]
    data.family = family

    if mis_nodes is not None:
        y = torch.zeros(n, dtype=torch.float)
        for v in mis_nodes:
            if 0 <= v < n:
                y[v] = 1.0
        data.y = y
        data.mis_size = int(y.sum().item())

    if metadata:
        for k, v in metadata.items():
            setattr(data, k, v)
    return data


def pyg_to_nx(data: Data) -> nx.Graph:
    """Convert PyG Data back to a simple undirected NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().tolist() if data.edge_index.numel() else []
    for u, v in edges:
        if u < v:
            G.add_edge(u, v)
    return G


def attach_graph_features(data: Data, family: str | None = None) -> Data:
    """Add ``graph_x`` to a Data that doesn't have it yet (e.g. legacy shards).

    O(1) — reads num_nodes and edge_index shape directly; no NetworkX round-trip.
    Mutates the Data in place and also returns it.
    """
    if hasattr(data, "graph_x") and data.graph_x is not None:
        return data
    fam = family if family is not None else getattr(data, "family", "unknown")
    n = int(data.num_nodes)
    m = int(data.edge_index.shape[1] // 2) if data.edge_index.numel() else 0
    density = (2 * m) / max(1, n * (n - 1))
    feats = torch.tensor([density, math.log(max(1, n))], dtype=torch.float)
    data.graph_x = torch.cat([feats, family_one_hot(fam)], dim=0).unsqueeze(0)
    if not hasattr(data, "family") or data.family is None:
        data.family = fam
    return data


# ---- size sampling --------------------------------------------------------

def sample_size_supervised(rng: random.Random) -> int:
    """50% small (20-60), 30% medium (60-120), 20% large (120-200)."""
    r = rng.random()
    if r < 0.50:
        return rng.randint(20, 60)
    if r < 0.80:
        return rng.randint(60, 120)
    return rng.randint(120, 200)


def sample_size_log_uniform(rng: random.Random, lo: int, hi: int) -> int:
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


# ---- supervised dataset ----------------------------------------------------

def build_supervised(count: int = 900, seed: int = 42, exact_cap: int = 35,
                     max_n: int = 200, n_restarts: int = 4,
                     show_progress: bool = True) -> tuple[list[Data], pd.DataFrame]:
    """Build a labeled MIS dataset across all 9 families."""
    families = gen.FAMILIES
    per_family = count // len(families)
    rows, datas = [], []
    iterator = range(per_family * len(families))
    if show_progress:
        iterator = tqdm(iterator, desc="supervised graphs")

    for idx in iterator:
        fam_idx = idx // per_family
        i = idx % per_family
        family = families[fam_idx]
        s = seed + 1_000_000 * fam_idx + i
        rng = random.Random(s)

        try:
            G = gen.sample_instance(family, rng,
                                    size_target=sample_size_supervised(rng))
            G = nx.convert_node_labels_to_integers(G, first_label=0)
            if G.number_of_nodes() > max_n:
                keep = rng.sample(range(G.number_of_nodes()), max_n)
                G = nx.convert_node_labels_to_integers(G.subgraph(keep).copy(),
                                                      first_label=0)
            if G.number_of_nodes() < 5:
                continue

            mis_rng = random.Random(s ^ 0xDEADBEEF)
            if G.number_of_nodes() <= exact_cap:
                mis = solvers.exact_mis_bb(G, node_cap=exact_cap)
                if mis is None:
                    mis = solvers.simulated_annealing(G, rng=mis_rng, n_restarts=n_restarts)
            else:
                mis = solvers.simulated_annealing(G, rng=mis_rng, n_restarts=n_restarts)

            n = G.number_of_nodes()
            data = nx_to_pyg(G, family, mis_nodes=mis,
                             metadata={"graph_id": idx, "seed": s})
            datas.append(data)
            rows.append({
                "graph_id": idx, "family": family,
                "n_nodes": n, "n_edges": G.number_of_edges(),
                "density": (2 * G.number_of_edges()) / max(1, n * (n - 1)),
                "mis_size": int(data.y.sum().item()),
                "mis_fraction": float(data.y.mean().item()),
                "seed": s,
            })
        except Exception as e:
            print(f"  [warn] skipping {family} #{idx}: {e}")

    if not rows:
        return [], pd.DataFrame(columns=["graph_id", "family", "n_nodes", "n_edges",
                                         "density", "mis_size", "mis_fraction", "seed"])
    return datas, pd.DataFrame(rows).sort_values("graph_id").reset_index(drop=True)


# ---- unsupervised dataset --------------------------------------------------

def build_unsupervised(count: int = 900, seed: int = 42,
                       min_n: int = 50, max_n: int = 800,
                       show_progress: bool = True) -> tuple[list[Data], pd.DataFrame]:
    """Build an unlabeled dataset with log-uniform sizes."""
    families = gen.FAMILIES
    per_family = count // len(families)
    rows, datas = [], []
    iterator = range(per_family * len(families))
    if show_progress:
        iterator = tqdm(iterator, desc="unsupervised graphs")

    for idx in iterator:
        fam_idx = idx // per_family
        i = idx % per_family
        family = families[fam_idx]
        s = seed + 1_000_000 * fam_idx + i
        rng = random.Random(s)

        try:
            target = sample_size_log_uniform(rng, min_n, max_n)
            G = gen.sample_instance(family, rng, size_target=target)
            G = nx.convert_node_labels_to_integers(G, first_label=0)
            if G.number_of_nodes() < 5:
                continue
            if G.number_of_nodes() > max_n:
                keep = rng.sample(range(G.number_of_nodes()), max_n)
                G = nx.convert_node_labels_to_integers(G.subgraph(keep).copy(),
                                                      first_label=0)
            n = G.number_of_nodes()
            data = nx_to_pyg(G, family,
                             metadata={"graph_id": idx, "seed": s, "size_target": target})
            datas.append(data)
            rows.append({
                "graph_id": idx, "family": family,
                "n_nodes": n, "n_edges": G.number_of_edges(),
                "density": (2 * G.number_of_edges()) / max(1, n * (n - 1)),
                "size_target": target, "seed": s,
            })
        except Exception as e:
            print(f"  [warn] skipping {family} #{idx}: {e}")

    if not rows:
        return [], pd.DataFrame(columns=["graph_id", "family", "n_nodes", "n_edges",
                                         "density", "size_target", "seed"])
    return datas, pd.DataFrame(rows).sort_values("graph_id").reset_index(drop=True)


# ---- splits ----------------------------------------------------------------

@dataclass
class Split:
    train: list[Data]
    val: list[Data]
    test: list[Data]


def stratified_split(datas: list[Data], train_frac: float = 0.7,
                     val_frac: float = 0.15, seed: int = 0) -> Split:
    """Family-stratified split. Guarantees a non-empty train split per family;
    val/test are filled if there's enough data."""
    rng = random.Random(seed)
    by_family: dict[str, list[Data]] = {}
    for d in datas:
        by_family.setdefault(d.family, []).append(d)

    train, val, test = [], [], []
    for fam, items in by_family.items():
        rng.shuffle(items)
        n = len(items)
        if n >= 3:
            n_train = max(1, int(n * train_frac))
            n_val = max(1, int(n * val_frac))
            n_train = min(n_train, n - 2)
            n_val = min(n_val, n - n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return Split(train, val, test)


def family_holdout_split(datas: list[Data], heldout: list[str],
                         val_frac: float = 0.15, seed: int = 0) -> Split:
    """Train/val on graphs whose family is *not* in ``heldout``; test = heldout.

    Used for the per-family generalization experiment: train on 8 families,
    test on the 9th, repeat.
    """
    rng = random.Random(seed)
    in_dist = [d for d in datas if d.family not in heldout]
    out_dist = [d for d in datas if d.family in heldout]
    rng.shuffle(in_dist); rng.shuffle(out_dist)
    n_val = max(1, int(len(in_dist) * val_frac))
    return Split(train=in_dist[n_val:], val=in_dist[:n_val], test=out_dist)


# ---- I/O -------------------------------------------------------------------

def save_dataset(datas: list[Data], meta_df: pd.DataFrame,
                 out_dir: str | Path, shard_size: int = 500) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(datas), shard_size):
        torch.save(datas[i:i + shard_size],
                   out_dir / f"shard_{i // shard_size:04d}.pt")
    meta_df.to_csv(out_dir / "metadata.csv", index=False)


def load_dataset(in_dir: str | Path,
                 ensure_graph_x: bool = True,
                 per_family: int | None = None,
                 max_n: int | None = None,
                 seed: int = 0) -> tuple[list[Data], pd.DataFrame]:
    """Load shards + metadata.csv from a directory.

    Args:
        ensure_graph_x : retrofit ``graph_x`` onto Data objects that lack it
                         (legacy MS3 shards predate this).
        per_family     : if set, keep at most this many graphs per family
                         (stratified subsampling, applied while loading so
                         we don't materialize the whole 14 GB unsup archive
                         when we only need 180 graphs).
        max_n          : if set, drop graphs with ``num_nodes > max_n``.
        seed           : RNG seed used by per_family subsampling.
    """
    in_dir = Path(in_dir)
    rng = random.Random(seed)
    by_family_kept: dict[str, list[Data]] = {}
    all_datas: list[Data] = []

    for shard in sorted(in_dir.glob("shard_*.pt")):
        chunk = torch.load(shard, weights_only=False)
        for d in chunk:
            if max_n is not None and int(d.num_nodes) > max_n:
                continue
            if ensure_graph_x:
                attach_graph_features(d, getattr(d, "family", None))
            if per_family is not None:
                fam = getattr(d, "family", "unknown")
                bucket = by_family_kept.setdefault(fam, [])
                # Reservoir-sample so any later shard can replace earlier picks.
                if len(bucket) < per_family:
                    bucket.append(d)
                else:
                    j = rng.randint(0, sum(1 for _ in bucket))
                    if j < per_family:
                        bucket[j] = d
            else:
                all_datas.append(d)
        # If per-family subsampling and every family is full, we *could* stop,
        # but reservoir sampling needs to see all candidates — keep going.

    if per_family is not None:
        all_datas = [d for items in by_family_kept.values() for d in items]
        rng.shuffle(all_datas)

    meta = pd.read_csv(in_dir / "metadata.csv") if (in_dir / "metadata.csv").exists() \
        else pd.DataFrame()
    return all_datas, meta


# ---- .tar.zst loader -------------------------------------------------------

def _decompress_zstd(in_path: Path, out_path: Path) -> None:
    """Decompress a .zst file to ``out_path``. Tries (in order): the
    ``zstandard`` Python module, the ``zstd`` CLI, ``tar --use-compress-program``.
    Raises RuntimeError with a clear message if none work.
    """
    try:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(in_path, "rb") as src, open(out_path, "wb") as dst:
            dctx.copy_stream(src, dst)
        return
    except ImportError:
        pass

    if shutil.which("zstd"):
        subprocess.check_call(["zstd", "-d", "-f", "-o", str(out_path), str(in_path)])
        return

    raise RuntimeError(
        f"Cannot decompress {in_path}: install the 'zstandard' Python package "
        f"(`pip install zstandard`) or the `zstd` command-line tool. "
        f"Alternatively, pre-decompress {in_path.name} to {out_path.name} "
        f"with any tool that handles .tar.zst (e.g. 7-Zip on Windows)."
    )


def extract_archive(archive_path: str | Path, out_dir: str | Path,
                    skip_if_exists: bool = True) -> Path:
    """Extract a ``.tar.zst`` archive into ``out_dir``. Returns the path to the
    inner directory (typically ``out_dir/<archive-stem>``)."""
    archive_path = Path(archive_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve stem: 'supervised.tar.zst' -> 'supervised'
    stem = archive_path.name
    for suf in (".tar.zst", ".tar.gz", ".tar"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    target = out_dir / stem

    if skip_if_exists and target.exists() and any(target.glob("shard_*.pt")):
        return target

    # 1) decompress to a temp .tar
    tar_path = out_dir / f"{stem}.tar"
    if archive_path.suffix == ".zst" or archive_path.name.endswith(".tar.zst"):
        _decompress_zstd(archive_path, tar_path)
    else:
        tar_path = archive_path

    # 2) extract the tar into out_dir, skipping macOS resource forks (._*)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers()
                   if not Path(m.name).name.startswith("._")]
        nested = any(m.name.startswith(stem + "/") for m in members)
        tf.extractall(path=(target.parent if nested else target),
                      members=members)
    if archive_path != tar_path and tar_path.exists():
        tar_path.unlink()
    return target


def load_archive(archive_path: str | Path, work_dir: str | Path,
                 ensure_graph_x: bool = True
                 ) -> tuple[list[Data], pd.DataFrame]:
    """Convenience: extract a .tar.zst archive and load the dataset inside.

    Caches the extracted shards in ``work_dir`` so re-runs are instant.
    """
    target = extract_archive(archive_path, work_dir)
    return load_dataset(target, ensure_graph_x=ensure_graph_x)
