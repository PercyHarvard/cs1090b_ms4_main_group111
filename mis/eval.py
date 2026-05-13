"""Unified evaluation harness: compare classical and learned MIS solvers on
a common dataset, with per-graph and per-family aggregates.

A "method" is a callable ``f(data, G) -> list[int]`` returning a node-id list
that should form an independent set in ``G``. ``compare_methods`` runs many
methods on the same graphs and returns a tidy DataFrame.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm.auto import tqdm

from . import data as dataio
from . import decode as decmod
from . import solvers
from .train import predict_scores


Method = Callable[[Data, nx.Graph], list[int]]


# ---- per-graph metric ------------------------------------------------------

def evaluate_method(data: Data, G: nx.Graph, method: Method) -> dict | None:
    """Run ``method``; return {'size', 'valid', 'runtime', 'chosen'} or
    ``None`` if the method declined (returned None) — typically because
    the graph is out of scope for this solver (e.g. exact B&B on large G).
    """
    t0 = time.time()
    chosen = method(data, G)
    dt = time.time() - t0
    if chosen is None:
        return None
    valid = solvers.is_independent_set(G, chosen)
    return {"size": len(chosen), "valid": valid, "runtime": dt, "chosen": chosen}


# ---- method factories ------------------------------------------------------

def make_classical_methods(rng_seed: int = 0,
                           sa_restarts: int = 4,
                           sa_max_steps: int | None = None,
                           include_exact: bool = True,
                           exact_cap: int = 35) -> dict[str, Method]:
    """Greedy / randomized greedy / SA / (optional) exact B&B as Methods."""
    import math
    import random

    def greedy_md(data, G):
        return solvers.greedy_min_degree(G)

    def greedy_rand(data, G):
        return solvers.greedy_random(G, random.Random(rng_seed))

    def _sa_naive(G, rng, n_steps):
        nodes = list(G.nodes())
        adj = {u: set(G.neighbors(u)) for u in nodes}
        current: set = set()
        best: set = set()
        T = 2.0
        for _ in range(n_steps):
            prop = set(current)
            if rng.random() < 0.5:
                blocked = set(prop)
                for u in prop:
                    blocked |= adj[u]
                feas = [u for u in nodes if u not in blocked]
                if feas:
                    prop.add(rng.choice(feas))
            else:
                if prop:
                    prop.remove(rng.choice(tuple(prop)))
            delta = len(current) - len(prop)
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current = prop
                if len(current) > len(best):
                    best = set(current)
        return sorted(best)

    def sa(data, G):
        steps = sa_max_steps if sa_max_steps is not None else 200
        return _sa_naive(G, random.Random(rng_seed), min(steps, 200))

    methods: dict[str, Method] = {
        "greedy_min_degree": greedy_md,
        "greedy_random":     greedy_rand,
        "sa_restarts":       sa,
    }
    if include_exact:
        def exact(data, G):
            return solvers.exact_mis_bb(G, node_cap=exact_cap)
        methods["exact"] = exact
    return methods


def make_gnn_method(model: nn.Module, decoder: str = "greedy_conflict_aware",
                    name: str | None = None,
                    device: str | None = None) -> tuple[str, Method]:
    """Wrap a trained model + decoder as a Method."""
    name = name or f"{getattr(model, 'kind', 'gnn')}+{decoder}"
    device = device or next(model.parameters()).device

    def method(data, G):
        scores = predict_scores(model, data, device=device)
        return decmod.decode(data, scores, method=decoder)
    return name, method


# ---- batch comparison ------------------------------------------------------

def compare_methods(datas: Iterable[Data], methods: dict[str, Method],
                    show_progress: bool = True) -> pd.DataFrame:
    """Run every method on every graph; long-form DataFrame.

    Columns: graph_id, family, n_nodes, n_edges, method, size, valid, runtime,
    label_size (if y available)."""
    rows = []
    iterator = list(datas)
    if show_progress:
        iterator = tqdm(iterator, desc="evaluate")

    for data in iterator:
        G = dataio.pyg_to_nx(data)
        family = getattr(data, "family", "unknown")
        graph_id = getattr(data, "graph_id", -1)
        n = data.num_nodes
        m = data.edge_index.shape[1] // 2
        true_size = (int(data.y.sum().item())
                     if hasattr(data, "y") and data.y is not None else None)
        for name, method in methods.items():
            r = evaluate_method(data, G, method)
            if r is None:                              # method declined (e.g. exact on big G)
                continue
            rows.append({
                "graph_id": graph_id, "family": family,
                "n_nodes": n, "n_edges": m,
                "method": name,
                "size": r["size"], "valid": r["valid"],
                "runtime": r["runtime"],
                "label_size": true_size,
            })
    return pd.DataFrame(rows)


# ---- aggregation -----------------------------------------------------------

def summarize(df: pd.DataFrame, ref_method: str | None = None) -> pd.DataFrame:
    """Mean size / runtime / valid-fraction by (family, method).
    Adds ``ratio_to_ref`` if ``ref_method`` is in df (per-graph normalization)."""
    if ref_method is not None and ref_method in df["method"].unique():
        ref = (df[df["method"] == ref_method][["graph_id", "size"]]
               .rename(columns={"size": "ref_size"}))
        df = df.merge(ref, on="graph_id", how="left")
        df["ratio_to_ref"] = df["size"] / df["ref_size"].clip(lower=1)
    agg = {"mean_size": ("size", "mean"),
           "median_size": ("size", "median"),
           "mean_runtime": ("runtime", "mean"),
           "valid_fraction": ("valid", "mean"),
           "n_graphs": ("size", "count")}
    if "ratio_to_ref" in df.columns:
        agg["ratio_to_ref"] = ("ratio_to_ref", "mean")
    return df.groupby(["family", "method"]).agg(**agg).round(4).reset_index()


def overall(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("method")
              .agg(mean_size=("size", "mean"),
                   mean_runtime=("runtime", "mean"),
                   valid_fraction=("valid", "mean"),
                   n_graphs=("size", "count"))
              .round(4)
              .sort_values("mean_size", ascending=False))


def approximation_ratio(df: pd.DataFrame, label_col: str = "label_size") -> pd.DataFrame:
    """Per-graph IS-size / label-size, then mean by (family, method).

    Useful when ``label_size`` is the SA / exact reference attached to the
    supervised dataset (data.y).
    """
    if label_col not in df.columns:
        raise KeyError(f"need column {label_col!r}; build dataset with labels")
    sub = df.dropna(subset=[label_col]).copy()
    sub["ratio"] = sub["size"] / sub[label_col].clip(lower=1)
    return (sub.groupby(["family", "method"])
               .agg(mean_ratio=("ratio", "mean"),
                    median_ratio=("ratio", "median"),
                    mean_size=("size", "mean"),
                    mean_label=(label_col, "mean"),
                    n=("ratio", "count"))
               .round(4)
               .reset_index())
