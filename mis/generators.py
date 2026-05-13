"""DIMACS-faithful synthetic graph generators (9 clique-benchmark families).

Adapted from ms3/mis_gnn-main/synthetic_data/dimacs_generators.py.
Each ``make_*`` function returns a NetworkX graph; ``sample_instance`` is the
unified entry point for randomized sampling at a target size.

Families: brockington, c_fat, hamming, johnson, keller, p_hat, sanchis,
sanchis_random, steiner (MANN). See ms3/mis_gnn-main/dimacs_generators.md
and ms3/mis_gnn-main/dimacs_docs/CONSTRUCTIONS.md for derivations.
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Iterable

import networkx as nx


# ---- helpers ---------------------------------------------------------------

def _add_edge_noise(G: nx.Graph, rng: random.Random, fraction: float) -> nx.Graph:
    """Flip a random fraction of edges (number of flips = fraction * |E|)."""
    if fraction <= 0:
        return G
    nodes = list(G.nodes())
    n_flips = int(fraction * G.number_of_edges())
    for _ in range(n_flips):
        u, v = rng.sample(nodes, 2)
        if G.has_edge(u, v):
            G.remove_edge(u, v)
        else:
            G.add_edge(u, v)
    return G


def _subsample_vertices(G: nx.Graph, rng: random.Random, keep_fraction: float) -> nx.Graph:
    """Induced subgraph on a random fraction of vertices."""
    if keep_fraction >= 1.0:
        return G
    nodes = list(G.nodes())
    k = max(1, int(keep_fraction * len(nodes)))
    return G.subgraph(rng.sample(nodes, k)).copy()


def _relabel_ints(G: nx.Graph) -> nx.Graph:
    return nx.convert_node_labels_to_integers(G, first_label=0)


# ---- 1. brockington --------------------------------------------------------

def make_brockington(rng: random.Random, n: int, clique_size: int,
                     edge_density: float, defender: int = 0) -> nx.Graph:
    """Hidden-clique graph (Brockington & Culberson). Plants a clique of size
    ``clique_size`` inside a near-random graph at the requested density."""
    if clique_size >= n:
        raise ValueError("clique_size must be < n")
    q = max(0.0, min(1.0, (1.0 - edge_density) * (1.0 + 0.02 * defender)))
    planted = set(rng.sample(range(n), clique_size))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v in itertools.combinations(range(n), 2):
        if u in planted and v in planted:
            continue
        if rng.random() < q:
            G.add_edge(u, v)
    C = nx.complement(G)
    C.graph["planted_clique"] = sorted(planted)
    return C


# ---- 2. c_fat --------------------------------------------------------------

def make_c_fat(rng: random.Random, n: int, c_param: float = 1.0,
               noise_edges: float = 0.0) -> nx.Graph:
    """Circular partition ring (Pardalos c-fat fault-diagnosis style)."""
    if n < 2:
        raise ValueError("n must be >= 2")
    k = max(1, int(n / (c_param * math.log(max(n, 2)))))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in range(n):
        pu = u % k
        for v in range(u + 1, n):
            diff = abs(pu - (v % k))
            if diff <= 1 or diff == k - 1:
                G.add_edge(u, v)
    return _add_edge_noise(G, rng, noise_edges)


# ---- 3. hamming ------------------------------------------------------------

def make_hamming(rng: random.Random, n: int, d: int,
                 noise_edges: float = 0.0, keep_fraction: float = 1.0) -> nx.Graph:
    """Binary Hamming graph H(n, d): edges between vectors with dist >= d."""
    if n < 1:
        raise ValueError("n must be >= 1")
    vertices = [tuple(b) for b in itertools.product((0, 1), repeat=n)]
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for i, u in enumerate(vertices):
        for v in vertices[i + 1:]:
            if sum(a != b for a, b in zip(u, v)) >= d:
                G.add_edge(u, v)
    G = _subsample_vertices(G, rng, keep_fraction)
    _add_edge_noise(G, rng, noise_edges)
    return _relabel_ints(G)


# ---- 4. johnson ------------------------------------------------------------

def make_johnson(rng: random.Random, n: int, w: int, d: int,
                 noise_edges: float = 0.0, keep_fraction: float = 1.0) -> nx.Graph:
    """Johnson graph J(n, w, d): edges between w-subsets with dist >= d."""
    if not 0 <= w <= n:
        raise ValueError("need 0 <= w <= n")
    vertices = [frozenset(s) for s in itertools.combinations(range(n), w)]
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for i, u in enumerate(vertices):
        for v in vertices[i + 1:]:
            if 2 * (w - len(u & v)) >= d:
                G.add_edge(u, v)
    G = _subsample_vertices(G, rng, keep_fraction)
    _add_edge_noise(G, rng, noise_edges)
    return _relabel_ints(G)


# ---- 5. keller -------------------------------------------------------------

def make_keller(rng: random.Random, dimension: int,
                noise_edges: float = 0.0, keep_fraction: float = 1.0) -> nx.Graph:
    """Keller graph in dimension ``d`` over Z_4."""
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    vertices = list(itertools.product(range(4), repeat=dimension))
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for i, u in enumerate(vertices):
        for v in vertices[i + 1:]:
            two_apart = other = False
            for a, b in zip(u, v):
                sub = abs(a - b)
                if sub == 2:
                    two_apart = True
                elif sub != 0:
                    other = True
            if two_apart and other:
                G.add_edge(u, v)
    G = _subsample_vertices(G, rng, keep_fraction)
    _add_edge_noise(G, rng, noise_edges)
    return _relabel_ints(G)


# ---- 6. p_hat --------------------------------------------------------------

def make_p_hat(rng: random.Random, n: int, a: float, b: float) -> nx.Graph:
    """Non-uniform random graph (Gendreau/Soriano/Salvail). Per-vertex p_i ~
    U[a, b]; edge (i, j) included with prob (p_i + p_j)/2."""
    if not 0.0 <= a <= b <= 1.0:
        raise ValueError("need 0 <= a <= b <= 1")
    p = [a] * n if (b - a) < 0.01 else [rng.uniform(a, b) for _ in range(n)]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < (p[i] + p[j]) / 2:
                G.add_edge(i, j)
    return G


# ---- 7. sanchis ------------------------------------------------------------

def make_sanchis(rng: random.Random, n: int, m: int, c: int,
                 rr: int | None = None) -> nx.Graph:
    """Sanchis-style graph with a planted c-clique, exactly m edges."""
    if not 1 <= c <= n:
        raise ValueError("need 1 <= c <= n")
    min_e, max_e = c * (c - 1) // 2, n * (n - 1) // 2
    if not min_e <= m <= max_e:
        raise ValueError(f"m must be in [{min_e}, {max_e}]")
    planted = rng.sample(range(n), c)
    planted_set = set(planted)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v in itertools.combinations(planted, 2):
        G.add_edge(u, v)
    pairs = [(u, v) for u, v in itertools.combinations(range(n), 2)
             if not (u in planted_set and v in planted_set)]
    rng.shuffle(pairs)
    for u, v in pairs[: m - G.number_of_edges()]:
        G.add_edge(u, v)
    G.graph["planted_clique"] = sorted(planted)
    G.graph["rr"] = rr
    return G


# ---- 8. sanchis_random -----------------------------------------------------

def make_sanchis_random(rng: random.Random, n: int, density: float) -> nx.Graph:
    """Erdős–Rényi at the given density (sanr family)."""
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must lie in [0, 1]")
    return nx.fast_gnp_random_graph(n, density, seed=rng.randint(0, 2**31 - 1))


# ---- 9. steiner / MANN -----------------------------------------------------

def _sts_blocks_bose(t: int) -> list[tuple[int, int, int]]:
    m = 2 * t + 1
    inv2 = pow(2, -1, m)
    A, B, C = (lambda i: i), (lambda i: m + i), (lambda i: 2 * m + i)
    blocks = [tuple(sorted((A(i), B(i), C(i)))) for i in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            avg = ((i + j) * inv2) % m
            blocks += [tuple(sorted((A(i), A(j), B(avg)))),
                       tuple(sorted((B(i), B(j), C(avg)))),
                       tuple(sorted((C(i), C(j), A(avg))))]
    return blocks


def _sts_blocks_hardcoded(order: int) -> list[tuple[int, int, int]]:
    if order == 7:
        return [(0, 1, 2), (0, 3, 4), (0, 5, 6),
                (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)]
    if order == 13:
        base = [(0, 1, 4), (0, 2, 7)]
        return [tuple(sorted((a + s) % 13 for a in t)) for t in base for s in range(13)]
    raise ValueError(f"no hardcoded STS for order {order}")


def sts_blocks(order: int, rng: random.Random | None = None) -> list[tuple[int, int, int]]:
    """Steiner triple system blocks for orders 7, 13, or v ≡ 3 (mod 6)."""
    if order <= 0:
        raise ValueError("order must be positive")
    if order in (7, 13):
        blocks = _sts_blocks_hardcoded(order)
    elif order % 6 == 3:
        blocks = _sts_blocks_bose((order - 3) // 6)
    else:
        raise ValueError(f"STS not supported for order {order}")
    if rng is not None:
        perm = list(range(order))
        rng.shuffle(perm)
        blocks = [tuple(sorted(perm[i] for i in t)) for t in blocks]
        rng.shuffle(blocks)
    return blocks


def make_mann(rng: random.Random, sts_order: int = 9,
              noise_edges: float = 0.0, keep_fraction: float = 1.0) -> nx.Graph:
    """MANN_a{k}-style graph from Mannino's set-cover→clique transform on STS."""
    blocks = sts_blocks(sts_order, rng=rng)
    G = nx.Graph()
    G.add_nodes_from(range(sts_order))
    next_aux = sts_order
    for block in blocks:
        if len(block) == 2:
            G.add_edge(block[0], block[1])
            continue
        aux = list(range(next_aux, next_aux + len(block)))
        next_aux += len(block)
        G.add_nodes_from(aux)
        for e in block:
            for a in aux:
                G.add_edge(e, a)
        for a, b in itertools.combinations(aux, 2):
            G.add_edge(a, b)
    C = nx.complement(G)
    C = _subsample_vertices(C, rng, keep_fraction)
    _add_edge_noise(C, rng, noise_edges)
    return _relabel_ints(C)


# ---- unified dispatch ------------------------------------------------------

GENERATORS = {
    "brockington":     make_brockington,
    "c_fat":           make_c_fat,
    "hamming":         make_hamming,
    "johnson":         make_johnson,
    "keller":          make_keller,
    "p_hat":           make_p_hat,
    "sanchis":         make_sanchis,
    "sanchis_random":  make_sanchis_random,
    "steiner":         make_mann,
}
FAMILIES = list(GENERATORS)


def _nearest(values: Iterable[int], target: int) -> int:
    return min(values, key=lambda v: abs(v - target))


def sample_instance(family: str, rng: random.Random,
                    size_target: int | None = None) -> nx.Graph:
    """Generate one random instance from ``family`` near ``size_target`` nodes."""
    if family not in GENERATORS:
        raise KeyError(f"unknown family {family!r}; known: {FAMILIES}")
    n = size_target or rng.choice([50, 100, 200, 400])

    if family == "brockington":
        c = max(3, int(rng.uniform(0.1, 0.25) * n))
        return make_brockington(rng, n=n, clique_size=c,
                                edge_density=rng.uniform(0.45, 0.8),
                                defender=rng.choice([0, 1, 2, 3, 4]))
    if family == "c_fat":
        return make_c_fat(rng, n=n, c_param=rng.uniform(1.0, 10.0),
                          noise_edges=rng.uniform(0.0, 0.05))
    if family == "hamming":
        keep = rng.uniform(0.7, 1.0)
        dim = min(10, max(3, int(round(math.log2(max(n / keep, 8))))))
        d = rng.choice([2, 3, max(2, dim // 2)])
        return make_hamming(rng, n=dim, d=d,
                            noise_edges=rng.uniform(0.0, 0.05), keep_fraction=keep)
    if family == "johnson":
        keep = rng.uniform(0.7, 1.0)
        catalog = [(6, 2, 15), (6, 3, 20), (8, 2, 28), (8, 3, 56),
                   (9, 3, 84), (10, 3, 120), (10, 4, 210), (12, 3, 220),
                   (12, 4, 495), (14, 4, 1001)]
        n_base, w, _ = min(catalog, key=lambda t: abs(t[2] * keep - n))
        return make_johnson(rng, n=n_base, w=w, d=rng.choice([2, 4]),
                            noise_edges=rng.uniform(0.0, 0.05), keep_fraction=keep)
    if family == "keller":
        keep = rng.uniform(0.7, 1.0)
        dim = _nearest([2, 3, 4, 5], int(round(math.log(max(n / keep, 4), 4))))
        return make_keller(rng, dimension=dim,
                           noise_edges=rng.uniform(0.0, 0.05), keep_fraction=keep)
    if family == "p_hat":
        a = rng.uniform(0.2, 0.5)
        b = rng.uniform(a, min(1.0, a + 0.4))
        return make_p_hat(rng, n=n, a=a, b=b)
    if family == "sanchis":
        c = max(4, int(rng.uniform(0.08, 0.20) * n))
        m = max(int(rng.uniform(0.4, 0.8) * n * (n - 1) / 2), c * (c - 1) // 2)
        return make_sanchis(rng, n=n, m=m, c=c,
                            rr=rng.randint(0, max(0, n // (3 * c))))
    if family == "sanchis_random":
        return make_sanchis_random(rng, n=n, density=rng.uniform(0.3, 0.8))
    if family == "steiner":
        keep = rng.uniform(0.85, 1.0)
        order = _nearest([9, 15, 21, 27, 33, 39, 45],
                         int(round((math.sqrt(1 + 8 * n / keep) - 1) / 2)))
        return make_mann(rng, sts_order=order,
                         noise_edges=rng.uniform(0.0, 0.05), keep_fraction=keep)
    raise AssertionError("unreachable")
