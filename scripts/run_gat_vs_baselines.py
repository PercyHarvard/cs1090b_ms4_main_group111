from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from mis import data as dataio
from mis import eval as ev
from mis import models
from mis import train as trainmod


def main() -> None:
    DATA = Path("data")
    OUT = Path("outputs")
    OUT.mkdir(exist_ok=True)

    SEED = 42
    torch.manual_seed(SEED)

    print("[1/4] loading supervised dataset")
    sup_datas, sup_meta = dataio.load_archive(DATA / "supervised.tar.zst",
                                              DATA / "extracted")
    print(f"  {len(sup_datas)} graphs across {sup_meta['family'].nunique()} families")

    split = dataio.stratified_split(sup_datas, train_frac=0.7, val_frac=0.15, seed=SEED)
    print(f"  split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    print("[2/4] building model")
    sample = sup_datas[0]
    node_in = int(sample.x.shape[1])
    graph_in = int(sample.graph_x.shape[-1])
    model = models.make_model("gat_cond",
                              node_in=node_in, graph_in=graph_in,
                              hidden=128, num_layers=4, heads=4, dropout=0.2)
    print(f"  params={models.count_parameters(model):,}")

    cfg = trainmod.TrainConfig(
        epochs=25, batch_size=32, lr=5e-4, weight_decay=1e-5,
        lam=1.0, logit_l2=1e-2, mode="unsupervised",
        patience=6, device="cpu",
        checkpoint_path=str(OUT / "gat_local.pt"),
    )

    print("[3/4] training")
    t0 = time.time()
    history = trainmod.train_model(model, split.train, split.val, cfg)
    print(f"  trained in {time.time() - t0:.1f}s; best epoch {history.best_epoch} "
          f"val_loss {history.best_val_loss:.3f}")

    print("[4/4] evaluating GAT vs (weakened) classical baselines on test split")
    classical = ev.make_classical_methods(rng_seed=SEED, sa_restarts=4,
                                          sa_max_steps=600, include_exact=True,
                                          exact_cap=35)
    name, gnn_method = ev.make_gnn_method(model, decoder="greedy_conflict_aware",
                                          name="gat_cond+greedy", device="cpu")
    methods = {**classical, name: gnn_method}

    df = ev.compare_methods(split.test, methods, show_progress=True)
    df.to_csv(OUT / "local_gat_vs_weak_baselines.csv", index=False)

    print()
    print("=== overall ===")
    print(ev.overall(df).to_string())

    print()
    print("=== summary by (family, method), ratio to sa_restarts ===")
    summary = ev.summarize(df, ref_method="sa_restarts")
    print(summary.to_string(index=False))

    print()
    print("=== approximation ratio to data.y (label_size) ===")
    ratio = ev.approximation_ratio(df)
    print(ratio.to_string(index=False))

    summary.to_csv(OUT / "local_gat_summary.csv", index=False)
    ratio.to_csv(OUT / "local_gat_approx_ratio.csv", index=False)
    (OUT / "local_gat_history.json").write_text(json.dumps({
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
    }, indent=2))


if __name__ == "__main__":
    main()
