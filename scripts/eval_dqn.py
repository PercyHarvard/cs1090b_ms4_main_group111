"""Evaluate a trained DQN Q-network on the supervised test split.

Greedy Q-decoding (epsilon = 0): repeatedly pick argmax Q over still-available
nodes. Reports per-family ratios against data.y, and saves per-graph CSV.

NOTE: This CLI uses MisQNet (mis/dqn.py). The pre-trained checkpoint
s2v_dqn_tuned_imit.pt was trained with QNet (defined inline in the main
notebook) and is not compatible with this script. Use the main notebook for
inference with the checkpoint.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch

from mis import data as dataio, solvers
from mis.dqn import MisQNet, count_parameters
from mis.dqn_train import dqn_decode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", default="data/extracted")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    print(f"[dqn-eval] tag={run_dir.name} hidden={cfg['hidden']} "
          f"layers={cfg['layers']} heads={cfg['heads']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MisQNet(base_node_in=cfg.get("base_node_in", 2),
                     graph_in=cfg["graph_in"],
                     hidden=cfg["hidden"], num_layers=cfg["layers"],
                     heads=cfg["heads"], dropout=cfg["dropout"])
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"[dqn-eval] loaded model ({count_parameters(model):,} params)")

    sup_dir = Path(args.data_dir) / "supervised"
    datas, _ = dataio.load_dataset(sup_dir, ensure_graph_x=True)
    split = dataio.stratified_split(datas, train_frac=0.7, val_frac=0.15, seed=args.seed)
    test_set = split.test if args.limit is None else split.test[:args.limit]
    print(f"[dqn-eval] {len(test_set)} test graphs")

    rows = []
    t0 = time.time()
    for i, d in enumerate(test_set):
        chosen = dqn_decode(model, d, device=device)
        G = dataio.pyg_to_nx(d)
        valid = solvers.is_independent_set(G, chosen)
        sa_size = int(d.y.sum().item())
        rows.append({
            "graph_id": int(getattr(d, "graph_id", -1)),
            "family": d.family, "n_nodes": int(d.num_nodes),
            "sa_size": sa_size,
            "dqn_size": len(chosen), "dqn_valid": valid,
        })
        if (i + 1) % 100 == 0 or i + 1 == len(test_set):
            elapsed = time.time() - t0
            print(f"[dqn-eval] {i+1}/{len(test_set)} graphs, "
                  f"{elapsed:.1f}s ({elapsed/(i+1):.3f}s/graph)")
    elapsed = time.time() - t0
    print(f"[dqn-eval] eval in {elapsed:.1f}s")

    df = pd.DataFrame(rows)
    print(f"[dqn-eval] valid fraction: {df['dqn_valid'].mean():.4f}")
    df["ratio"] = df["dqn_size"] / df["sa_size"].clip(lower=1)
    df.to_csv(run_dir / "dqn_eval.csv", index=False)

    summary = (df.groupby("family").agg(
        mean_ratio=("ratio", "mean"),
        n=("graph_id", "count")).round(4).reset_index())
    print(f"\n[dqn-eval] per-family ratio to label:")
    print(summary.to_string(index=False))
    print(f"\n[dqn-eval] overall: {df['ratio'].mean():.4f}")


if __name__ == "__main__":
    main()
