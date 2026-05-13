"""End-to-end smoke test on a tiny in-memory dataset.
Verifies the training + diagnostic scripts before kicking off real runs.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mis import data as dataio, models, train as tr, eval as ev, decode as decmod
from mis.train import predict_scores

print("=== smoke test ===")
torch.manual_seed(42)

print("[smoke] building 27 supervised + 27 unsupervised graphs in-memory ...")
sup_datas, _ = dataio.build_supervised(count=27, seed=42, show_progress=False)
print(f"[smoke] sup: {len(sup_datas)} graphs, |V| up to {max(d.num_nodes for d in sup_datas)}")
unsup_datas, _ = dataio.build_unsupervised(count=27, seed=43,
                                            min_n=20, max_n=80,
                                            show_progress=False)
print(f"[smoke] unsup: {len(unsup_datas)} graphs, |V| up to {max(d.num_nodes for d in unsup_datas)}")

split_sup = dataio.stratified_split(sup_datas, train_frac=0.7, val_frac=0.15, seed=42)
split_unsup = dataio.stratified_split(unsup_datas, train_frac=0.7, val_frac=0.15, seed=42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Train both in 3 epochs each, just to exercise the pipeline.
for mode, split in [("supervised", split_sup), ("unsupervised", split_unsup)]:
    print(f"\n[smoke] training mode={mode} ...")
    model = models.make_model("gat_cond", graph_in=dataio.GRAPH_FEATURE_DIM,
                               hidden=32, num_layers=3, heads=2, dropout=0.2)
    cfg = tr.TrainConfig(epochs=3, batch_size=8, lr=5e-4, lam=1.0, logit_l2=1e-2,
                          mode=mode, patience=10, device=device, verbose=True)
    hist = tr.train_model(model, split.train, split.val, cfg)
    print(f"[smoke] {mode} best_epoch={hist.best_epoch} best_val={hist.best_val_loss:.4f}")

    # Test ranking on the supervised test set (always has labels)
    test = split_sup.test
    rank_rows = []
    for d in test[:5]:
        scores = predict_scores(model, d, device=device).numpy()
        from scipy import stats
        import numpy as np
        y = d.y.cpu().numpy()
        r = float(stats.spearmanr(scores, y)[0]) if y.sum() > 0 and y.sum() < len(y) else float("nan")
        rank_rows.append(r)
    print(f"[smoke] {mode} sample Spearman: {rank_rows}")

# Test classical eval with the last model
print("\n[smoke] classical eval on 5 sup-test graphs ...")
methods = ev.make_classical_methods(rng_seed=42, sa_restarts=2, include_exact=False)
gnn_name, gnn_method = ev.make_gnn_method(model, decoder="greedy_conflict_aware",
                                           name="smoke_gnn", device=device)
methods[gnn_name] = gnn_method
df = ev.compare_methods(split_sup.test[:5], methods, show_progress=False)
print(df.to_string())

print("\n=== smoke test PASSED ===")
