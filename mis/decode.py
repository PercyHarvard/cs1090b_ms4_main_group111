"""Decoders that turn soft per-node MIS scores into a *valid* independent set.

The model outputs a probability ``p_v`` per node, but a soft vector is not an
IS. Three decoders:

    greedy_conflict_aware   Sort nodes by score descending; add each one to
                            the set if no neighbor is already chosen. Always
                            feasible. (This is the headline decoder in the
                            slide diagram.)

    threshold_repair        Threshold at ``tau``; iteratively drop the most-
                            conflicted node until valid. Sensitive to ``tau``;
                            we sweep it lightly in the notebook.

    sequential_local        Greedy + a 1-out-2-in local-repair pass that
                            swaps a chosen node for two non-adjacent neighbors
                            when doing so increases the set size.

All decoders return a sorted list of node ids that form an independent set
in the input graph.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def _to_numpy(scores: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(scores, torch.Tensor):
        return scores.detach().cpu().float().numpy()
    return np.asarray(scores, dtype=float)


def _adjacency(data: Data) -> dict[int, set[int]]:
    """{node: set(neighbors)} from a PyG Data."""
    n = data.num_nodes
    adj: dict[int, set[int]] = {v: set() for v in range(n)}
    if data.edge_index.numel():
        for u, v in data.edge_index.t().tolist():
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
    return adj


# ---- greedy conflict-aware (headline decoder) -----------------------------

def greedy_conflict_aware(data: Data, scores) -> list[int]:
    """Sort by score descending; add each node if no neighbor is chosen.
    Always returns a valid IS."""
    adj = _adjacency(data)
    s = _to_numpy(scores)
    order = np.argsort(-s)
    chosen: set[int] = set()
    blocked: set[int] = set()
    for v in order:
        v = int(v)
        if v in blocked:
            continue
        chosen.add(v)
        blocked.add(v)
        blocked.update(adj[v])
    return sorted(chosen)


# Backwards-compat alias used in earlier MS3 work.
greedy_projection = greedy_conflict_aware


# ---- threshold + repair ---------------------------------------------------

def threshold_repair(data: Data, scores, tau: float = 0.5) -> list[int]:
    """Threshold at ``tau``; remove highest-conflict node until valid."""
    adj = _adjacency(data)
    s = _to_numpy(scores)
    chosen = {int(v) for v in np.where(s >= tau)[0]}
    if not chosen:
        return greedy_conflict_aware(data, scores)
    while True:
        conflicts = [(u, v) for u in chosen for v in adj[u] & chosen if u < v]
        if not conflicts:
            break
        cnt: dict[int, int] = {}
        for u, v in conflicts:
            cnt[u] = cnt.get(u, 0) + 1
            cnt[v] = cnt.get(v, 0) + 1
        worst = max(cnt.items(),
                    key=lambda kv: (kv[1], -float(s[kv[0]])))[0]
        chosen.remove(worst)
    return sorted(chosen)


# ---- sequential + local-repair --------------------------------------------

def sequential_local(data: Data, scores, max_passes: int = 1) -> list[int]:
    """Greedy + 1-out-2-in local-repair sweep."""
    adj = _adjacency(data)
    chosen = set(greedy_conflict_aware(data, scores))
    s = _to_numpy(scores)
    for _ in range(max_passes):
        improved = False
        for u in list(chosen):
            blocked = set()
            for c in chosen - {u}:
                blocked |= adj[c]
            free = sorted((adj[u] - chosen) - blocked,
                          key=lambda x: -float(s[x]))
            for i, a in enumerate(free):
                for b in free[i + 1:]:
                    if b in adj[a]:
                        continue
                    chosen.discard(u); chosen.add(a); chosen.add(b)
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return sorted(chosen)


# ---- registry --------------------------------------------------------------

DECODERS = {
    "greedy_conflict_aware": greedy_conflict_aware,
    "threshold_repair":      threshold_repair,
    "sequential_local":      sequential_local,
    # legacy alias
    "greedy_projection":     greedy_conflict_aware,
}


def decode(data: Data, scores, method: str = "greedy_conflict_aware",
           **kwargs) -> list[int]:
    if method not in DECODERS:
        raise KeyError(f"unknown decoder {method!r}; known: {list(DECODERS)}")
    return DECODERS[method](data, scores, **kwargs)
