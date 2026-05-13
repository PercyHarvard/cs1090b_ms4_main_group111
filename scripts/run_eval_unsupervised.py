from __future__ import annotations

import time
from pathlib import Path

import torch

from mis import data as dataio
from mis import eval as ev
from mis import models


def main() -> None:
    DATA = Path("data")
    OUT = Path("outputs")
    OUT.mkdir(exist_ok=True)
    SEED = 42

    print("[1/4] loading unsupervised dataset")
    unsup_datas, unsup_meta = dataio.load_archive(DATA / "unsupervised.tar.zst",
                                                  DATA / "extracted")
    print(f"  {len(unsup_datas)} graphs; n in [{unsup_meta['n_nodes'].min()}, "
          f"{unsup_meta['n_nodes'].max()}]")

    print("[2/4] loading local GAT")
    ckpt_path = OUT / "gat_local.pt"
    sample = unsup_datas[0]
    node_in = int(sample.x.shape[1])
    graph_in = int(sample.graph_x.shape[-1])
    model = models.make_model("gat_cond",
                              node_in=node_in, graph_in=graph_in,
                              hidden=128, num_layers=4, heads=4, dropout=0.2)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to("cpu").eval()
    print(f"  loaded {ckpt_path} ({models.count_parameters(model):,} params)")

    print("[3/4] building methods (no exact, no labels in unsup data)")
    classical = ev.make_classical_methods(rng_seed=SEED, sa_restarts=4,
                                          sa_max_steps=600, include_exact=False)
    name, gnn_method = ev.make_gnn_method(model, decoder="greedy_conflict_aware",
                                          name="gat_cond+greedy", device="cpu")
    methods = {**classical, name: gnn_method}
    print(f"  methods: {list(methods.keys())}")

    print("[4/4] running compare_methods on the full unsupervised set")
    t0 = time.time()
    df = ev.compare_methods(unsup_datas, methods, show_progress=True)
    print(f"  total time: {time.time() - t0:.1f}s")

    long_path = OUT / "unsup_eval_long.csv"
    df.drop(columns=["label_size"], errors="ignore").to_csv(long_path, index=False)
    print(f"  wrote {long_path} ({len(df)} rows)")

    wide = (df.pivot_table(index=["graph_id", "family", "n_nodes", "n_edges"],
                            columns="method", values="size")
              .reset_index())
    wide.columns.name = None
    wide_path = OUT / "unsup_eval_wide.csv"
    wide.to_csv(wide_path, index=False)
    print(f"  wrote {wide_path} ({len(wide)} rows)")

    print()
    print("=== overall (mean IS size per method) ===")
    print(ev.overall(df).to_string())


if __name__ == "__main__":
    main()
