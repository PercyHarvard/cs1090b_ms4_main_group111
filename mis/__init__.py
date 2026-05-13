"""MIS GNN package — Group 111, CS 1090B MS4.

Modules:
    generators : DIMACS-faithful synthetic graph generators (9 families).
    solvers    : classical MIS heuristics (greedy, SA, exact B&B).
    data       : NetworkX <-> PyG conversion, graph-level conditioning,
                 dataset assembly + splits, .tar.zst loader.
    models     : Conditioned residual GAT (the final model) + GCN / SAGE
                 baselines for comparison.
    decode     : decoders that turn soft probabilities into a valid IS.
    train      : unsupervised (and optional supervised) training loop.
    eval       : unified per-graph + by-family evaluation harness.
    dqn        : S2V-DQN Q-network (MisQNet) + state-tag utilities.
    dqn_train  : DQN training loop, replay buffer, greedy/beam decoder.

Code adapted from MS3 (ms3/mis_gnn-main/) with extensions for MS4.
"""

__version__ = "0.3.0"
