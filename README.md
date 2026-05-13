# Neural Combinatorial Optimization for Maximum Independent Set

CS 1090B (Spring 2026) final project — **Group 111**
Mohammad Khan · Elvin Lo · Perce Thaveesittikullarp · Jerry Zhang

We train an unsupervised conditioned-residual Graph Attention Network (GAT)
for the **Maximum Independent Set** (MIS) problem, then extend to iterative
S2V-DQN decoding with beam search, following Khalil et al. (NeurIPS 2017).
Graph families are inspired by the DIMACS 2nd Implementation Challenge.

## Quick start

```bash
# 1. Install dependencies (edit the torch line for your CUDA version if needed).
pip install -r requirements.txt

# 2. Open the main notebook and run Restart Kernel + Run All.
jupyter lab cs1090b_ms4_main_group111.ipynb
```

**No data download or pre-processing needed** — the notebook generates all
synthetic data from a fixed seed at runtime (~30 s). Pre-trained checkpoints
(`gat_cond.pt`, `s2v_dqn_tuned_imit.pt`) are downloaded automatically from
Google Drive on first run if not already present in `outputs/`.

`QUICK_MODE = True` (default) runs end-to-end in **~10–25 min** on CPU or
~5–10 min on GPU. Set `QUICK_MODE = False` for full training (~1–3 h on GPU).

## Repository layout

```
.
├── cs1090b_ms4_main_group111.ipynb    # main MS4 notebook (the deliverable)
├── requirements.txt
├── mis/                                # core Python package
│   ├── generators.py                   # 9 DIMACS-faithful synthetic graph families
│   ├── solvers.py                      # greedy, randomized greedy, SA, exact B&B
│   ├── data.py                         # NetworkX ↔ PyG, splits, archive loader
│   ├── models.py                       # ConditionedResidualGAT + GCN/SAGE baselines
│   ├── decode.py                       # greedy / threshold / local-repair decoders
│   ├── train.py                        # unsupervised IS loss + training loop
│   ├── eval.py                         # per-graph + by-family evaluation harness
│   ├── dqn.py                          # MisQNet Q-network + state-tag helpers
│   └── dqn_train.py                    # DQN training loop + replay buffer
└── scripts/
    ├── build_data.py                   # pre-generate data archives (optional)
    ├── smoke_test.py                   # quick import + shape sanity check
    ├── run_gat_vs_baselines.py         # GAT vs classical baselines (outputs CSV)
    ├── run_eval_unsupervised.py        # unsupervised-set evaluation
    ├── train_dqn.py                    # CLI DQN trainer (uses MisQNet architecture)
    └── eval_dqn.py                     # CLI DQN evaluator (uses MisQNet architecture)
```

> **Note on scripts/train_dqn.py and eval_dqn.py:** these CLI tools use the
> `MisQNet` architecture from `mis/dqn.py`. The pre-trained checkpoint
> (`s2v_dqn_tuned_imit.pt`) was trained with the `QNet` architecture defined
> inline in the main notebook and is not interchangeable. Use the notebook
> for inference with the checkpoint.

## Method summary

**One-shot GAT (§6–§8).** A 4-layer residual GAT outputs per-node probabilities
trained with the unsupervised differentiable IS loss (Karalias & Loukas, NeurIPS 2020):

$$\mathcal{L}(\mathbf{p}) = -\sum_v p_v + \lambda \sum_{(u,v) \in E} p_u p_v$$

Node inputs (degree, normalized degree) are concatenated with broadcast
graph-level features (density, log|V|, 9-way family one-hot). Soft scores
are decoded via greedy conflict-aware projection. Despite architecture and
hyperparameter sweeps, all one-shot configurations plateau below simulated
annealing — a structural ceiling caused by the lack of state conditioning.

**S2V-DQN (§9–§10).** A state-conditioned Q-network (`QNet`, 138k params,
`hidden=128, layers=6, heads=8`) re-embeds the graph after each node pick,
conditioning on a per-node `in_S` indicator. Trained via imitation
pretraining + n-step Q-learning. Greedy DQN matches classical greedy
baselines; beam search (k ≥ 5) further improves results because each beam
maintains a distinct state tag — unlike static GAT beam search, which
degenerates to the same ranking in every beam.

## References

Khalil et al. *Learning Combinatorial Optimization Algorithms over Graphs*. NeurIPS 2017.

Karalias & Loukas. *Erdős goes neural: an unsupervised learning framework for
combinatorial optimization on graphs*. NeurIPS 2020.

Fey & Lenssen. *Fast Graph Representation Learning with PyTorch Geometric*. 2019.
