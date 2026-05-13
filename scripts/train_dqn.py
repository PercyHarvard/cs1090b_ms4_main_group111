"""Train the S2V-DQN-style Q-network for MIS (Khalil et al. 2017).

Saves checkpoint, history, config to outputs/<tag>/.

NOTE: This CLI uses MisQNet (mis/dqn.py), the original architecture from the
DQN branch. The main notebook (cs1090b_ms4_main_group111.ipynb) uses QNet, a
newer architecture whose pre-trained checkpoint (s2v_dqn_tuned_imit.pt) is
not compatible with this script. Use this script to retrain from scratch with
MisQNet; use the notebook's inline QNet definition to work with the checkpoint.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from mis import data as dataio
from mis.dqn import MisQNet, count_parameters
from mis.dqn_train import DqnConfig, train_dqn


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", default="data/extracted")
    ap.add_argument("--dataset", choices=["supervised", "unsupervised"], default="supervised")
    ap.add_argument("--max-n", type=int, default=120,
                    help="cap training graphs at this many nodes (rollouts on big "
                         "graphs are slow because they're O(|IS|) GAT calls)")
    ap.add_argument("--seed", type=int, default=42)
    # arch
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    # training
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-step", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-episodes", type=int, default=1500)
    ap.add_argument("--target-update-episodes", type=int, default=100)
    ap.add_argument("--updates-per-episode", type=int, default=4)
    ap.add_argument("--warmup-transitions", type=int, default=1000)
    ap.add_argument("--replay-capacity", type=int, default=50000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-val-graphs", type=int, default=50,
                    help="how many val-set graphs to greedy-decode at each eval checkpoint")
    args = ap.parse_args()

    out_dir = Path("outputs") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print(f"[dqn-train] tag={args.tag} dataset={args.dataset} max_n={args.max_n}")
    print(f"[dqn-train] arch hidden={args.hidden} layers={args.layers} "
          f"heads={args.heads} dropout={args.dropout}")
    print(f"[dqn-train] episodes={args.episodes} batch={args.batch_size} lr={args.lr} "
          f"n_step={args.n_step} gamma={args.gamma}")
    print(f"[dqn-train] device={'cuda' if torch.cuda.is_available() else 'cpu'} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    src_dir = Path(args.data_dir) / args.dataset
    print(f"[dqn-train] loading {src_dir} (max_n={args.max_n}) ...")
    t0 = time.time()
    datas, _ = dataio.load_dataset(src_dir, ensure_graph_x=True, max_n=args.max_n)
    print(f"[dqn-train] loaded {len(datas)} graphs in {time.time()-t0:.1f}s")
    sizes = np.array([int(d.num_nodes) for d in datas])
    print(f"[dqn-train] |V| stats: min={sizes.min()} median={int(np.median(sizes))} "
          f"mean={sizes.mean():.0f} max={sizes.max()}")

    split = dataio.stratified_split(datas, train_frac=0.7, val_frac=0.15, seed=args.seed)
    print(f"[dqn-train] split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    val_subset = split.val[:args.n_val_graphs] if args.n_val_graphs > 0 else None

    model = MisQNet(base_node_in=2, graph_in=dataio.GRAPH_FEATURE_DIM,
                    hidden=args.hidden, num_layers=args.layers,
                    heads=args.heads, dropout=args.dropout)
    print(f"[dqn-train] Q-net params: {count_parameters(model):,}")

    cfg = DqnConfig(
        episodes=args.episodes, batch_size=args.batch_size, lr=args.lr,
        n_step=args.n_step, gamma=args.gamma,
        eps_start=args.eps_start, eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        target_update_episodes=args.target_update_episodes,
        updates_per_episode=args.updates_per_episode,
        warmup_transitions=args.warmup_transitions,
        replay_capacity=args.replay_capacity,
        log_every=args.log_every, eval_every=args.eval_every,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    ckpt_path = out_dir / "model.pt"
    print(f"[dqn-train] training start ...")
    t0 = time.time()
    history = train_dqn(model, split.train, cfg=cfg, val_graphs=val_subset,
                         checkpoint_path=str(ckpt_path))
    elapsed = time.time() - t0
    print(f"[dqn-train] done in {elapsed:.1f}s")
    print(f"[dqn-train] best val mean IS: {max(history.eval_size, default=-1):.3f}")

    config = vars(args).copy()
    config["graph_in"] = dataio.GRAPH_FEATURE_DIM
    config["base_node_in"] = 2
    config["n_train"] = len(split.train)
    config["n_val"] = len(split.val)
    config["n_test"] = len(split.test)
    config["n_params"] = count_parameters(model)
    config["train_seconds"] = elapsed
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "history.json").write_text(json.dumps({
        "episode_size": history.episode_size,
        "episode_reward": history.episode_reward,
        "losses": history.losses,
        "eval_size": history.eval_size,
        "eval_at": history.eval_at,
    }, indent=2))
    print(f"[dqn-train] saved to {out_dir}")


if __name__ == "__main__":
    main()
