"""Generate the supervised + unsupervised MIS datasets used in the MS4 paper.

Outputs uncompressed PyG shards under ``data/extracted/{supervised,unsupervised}/``,
which is exactly the layout the main notebook expects after auto-extraction
of the bundled .tar.zst archives. Use this when you don't have the archives
and want to reproduce the data from scratch.

Usage:
    python scripts/build_data.py                # default counts + seed
    python scripts/build_data.py --count 5004
    python scripts/build_data.py --supervised-only
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

# Allow running from the repo root: `python scripts/build_data.py`.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mis import data as dataio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=5004,
                        help="approximate total graphs per dataset (rounded down "
                             "to a multiple of 9 families)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/extracted"),
                        help="output directory; subdirs supervised/, unsupervised/ "
                             "will be created")
    parser.add_argument("--supervised-only", action="store_true")
    parser.add_argument("--unsupervised-only", action="store_true")
    parser.add_argument("--max-n-supervised", type=int, default=200)
    parser.add_argument("--min-n-unsupervised", type=int, default=50)
    parser.add_argument("--max-n-unsupervised", type=int, default=1500)
    parser.add_argument("--exact-cap", type=int, default=35,
                        help="graphs with n <= this get exact MIS labels (B&B)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.unsupervised_only:
        print(f"Building supervised dataset ({args.count} graphs)...")
        t0 = time.time()
        datas, meta = dataio.build_supervised(
            count=args.count, seed=args.seed,
            max_n=args.max_n_supervised, exact_cap=args.exact_cap)
        sup_dir = args.out / "supervised"
        dataio.save_dataset(datas, meta, sup_dir, shard_size=500)
        print(f"  wrote {len(datas)} supervised graphs to {sup_dir} "
              f"in {time.time() - t0:.1f}s")

    if not args.supervised_only:
        print(f"Building unsupervised dataset ({args.count} graphs)...")
        t0 = time.time()
        datas, meta = dataio.build_unsupervised(
            count=args.count, seed=args.seed + 1,
            min_n=args.min_n_unsupervised, max_n=args.max_n_unsupervised)
        uns_dir = args.out / "unsupervised"
        dataio.save_dataset(datas, meta, uns_dir, shard_size=250)
        print(f"  wrote {len(datas)} unsupervised graphs to {uns_dir} "
              f"in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
