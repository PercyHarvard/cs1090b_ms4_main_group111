"""Classical MIS heuristics: greedy variants, simulated annealing, exact B&B.

Adapted from ms3/mis_gnn-main/synthetic_data/build_supervised_dataset.py and
ms3/mis_gnn-main/MS2_jz_final_clean.ipynb.

Every solver returns a list of node ids forming an independent set in ``G``.
``solve_mis`` is the top-level entry point: tries exact for small graphs,
falls back to SA-with-restarts for larger ones.
"""
from __future__ import annotations

import math
import random
import time
from typing import Optional

import networkx as nx
import numpy as np


# ---- validity check --------------------------------------------------------

def is_independent_set(G: nx.Graph, nodes) -> bool:
    """True iff ``nodes`` induces an empty subgraph in ``G``."""
    nset = set(nodes)
    return all(v not in nset for u in nset for v in G.neighbors(u))


# ---- greedy variants -------------------------------------------------------

def greedy_min_degree(G: nx.Graph) -> list:
    """Pick the lowest-degree node iteratively (deterministic)."""
    sub = G.copy()
    mis = []
    while sub.number_of_nodes():
        v = min(sub.nodes(), key=lambda x: sub.degree(x))
        mis.append(v)
        sub.remove_nodes_from([v] + list(sub.neighbors(v)))
    return mis


def greedy_random(G: nx.Graph, rng: random.Random) -> list:
    """Random-order greedy IS construction."""
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    chosen = set()
    blocked = set()
    for u in nodes:
        if u not in blocked:
            chosen.add(u)
            blocked.add(u)
            blocked.update(G.neighbors(u))
    return sorted(chosen)


def best_greedy_random(G: nx.Graph, n_restarts: int = 16,
                       rng: random.Random | None = None) -> list:
    """Best of n_restarts random-order greedy runs."""
    rng = rng or random.Random(0)
    best = greedy_min_degree(G)
    for _ in range(n_restarts):
        cand = greedy_random(G, rng)
        if len(cand) > len(best):
            best = cand
    return best


# ---- simulated annealing ---------------------------------------------------

def _sa_one(G: nx.Graph, rng: random.Random, init: list, max_steps: int,
            T0: float = 1.0, T_end: float = 1e-3) -> list:
    """One SA run with feasibility-preserving moves (add / remove / swap)."""
    nodes = list(G.nodes())
    adj = {u: set(G.neighbors(u)) for u in nodes}
    current = {u for u in init if u in adj}
    best = set(current)

    def feasible_to_add(S):
        blocked = set(S)
        for u in S:
            blocked |= adj[u]
        return [u for u in nodes if u not in blocked]

    temps = (np.geomspace(T0, T_end, max_steps) if max_steps > 1 else [T_end])
    for T in temps:
        prop = set(current)
        r = rng.random()
        if r < 0.45:
            feas = feasible_to_add(prop)
            if feas:
                prop.add(rng.choice(feas))
        elif r < 0.75:
            if prop:
                prop.remove(rng.choice(tuple(prop)))
        else:
            if prop:
                prop.remove(rng.choice(tuple(prop)))
            feas = feasible_to_add(prop)
            if feas:
                prop.add(rng.choice(feas))
        delta = len(current) - len(prop)
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            current = prop
            if len(current) > len(best):
                best = set(current)
    return sorted(best)


def simulated_annealing(G: nx.Graph, rng: random.Random | None = None,
                        n_restarts: int = 5, max_steps: Optional[int] = None) -> list:
    """SA with multiple warm starts (min-degree greedy + random-order greedies)."""
    rng = rng or random.Random(0)
    n = G.number_of_nodes()
    if max_steps is None:
        max_steps = min(1000, max(200, 4 * n))
    starts = [greedy_min_degree(G)]
    starts += [greedy_random(G, random.Random(rng.randint(0, 2**31)))
               for _ in range(max(0, n_restarts - 1))]
    best = []
    for init in starts:
        cand = _sa_one(G, random.Random(rng.randint(0, 2**31)), init, max_steps)
        if len(cand) > len(best):
            best = cand
    return best


# ---- exact branch & bound --------------------------------------------------

def exact_mis_bb(G: nx.Graph, node_cap: int = 40) -> list | None:
    """Branch-and-bound exact MIS. Returns None if |V| > node_cap."""
    if G.number_of_nodes() > node_cap:
        return None
    nodes = list(G.nodes())
    adj = {u: set(G.neighbors(u)) for u in nodes}
    best: list = []

    def recurse(cands: set, current: list) -> None:
        nonlocal best
        if not cands:
            if len(current) > len(best):
                best = list(current)
            return
        if len(current) + len(cands) <= len(best):
            return
        u = max(cands, key=lambda x: len(adj[x] & cands))
        current.append(u)
        recurse(cands - {u} - adj[u], current)
        current.pop()
        recurse(cands - {u}, current)

    recurse(set(nodes), [])
    return sorted(best)


# ---- alternative exact via complement clique -------------------------------

def exact_mis_via_clique(G: nx.Graph, time_limit: float = 5.0) -> list | None:
    """Use NetworkX find_cliques on the complement. Returns None if it stalls
    past ``time_limit`` seconds (rough cutoff via wall-clock check)."""
    start = time.time()
    comp = nx.complement(G)
    best: list = []
    for clique in nx.find_cliques(comp):
        if len(clique) > len(best):
            best = list(clique)
        if time.time() - start > time_limit:
            return None
    return sorted(best)


# ---- top-level dispatch ----------------------------------------------------

def solve_mis(G: nx.Graph, rng: random.Random | None = None,
              exact_cap: int = 35) -> list:
    """Best-effort MIS: exact B&B for small graphs, SA-restarts otherwise."""
    rng = rng or random.Random(0)
    if G.number_of_nodes() <= exact_cap:
        ex = exact_mis_bb(G, node_cap=exact_cap)
        if ex is not None:
            return ex
    return simulated_annealing(G, rng=rng, n_restarts=6)
