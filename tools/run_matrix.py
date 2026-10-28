#!/usr/bin/env python3
"""Runner for the unaligned sequence ladder experiment matrix.

Executes 3 seeds per cell for both dense baseline and length-invariant variants:
  - Oct 7: l=100 and l=200
  - Oct 8: l=375/500 (native unaligned)

Usage:
  python tools/run_matrix.py --cells l100,l200
  python tools/run_matrix.py --cells native
  python tools/run_matrix.py --cells all
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXEC = sys.executable

CELL_CONFIGS = {
    "l100": {
        "label": "l=100",
        "data_path": "data/ladder/mosi_data_noalign_l100.pkl",
        "runs_prefix": "mosi_ladder_l100",
    },
    "l200": {
        "label": "l=200",
        "data_path": "data/ladder/mosi_data_noalign_l200.pkl",
        "runs_prefix": "mosi_ladder_l200",
    },
    "native": {
        "label": "l=375/500",
        "data_path": "data/mosi_data_noalign.pkl",
        "runs_prefix": "mosi_noalign",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment matrix across cells.")
    parser.add_argument(
        "--cells",
        type=str,
        default="l100,l200",
        help="Comma-separated cell names: l100, l200, native, or 'all'",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3",
        help="Comma-separated seeds (default: 1,2,3)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline,invariant",
        help="Comma-separated variants: baseline, invariant",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=40,
        help="Number of epochs per run (default: 40)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip runs where results.json already exists",
    )
    return parser.parse_args()


def run_single(variant: str, data_path: str, seed: int, run_name: str, num_epochs: int) -> bool:
    run_dir = REPO_ROOT / run_name
    json_path = run_dir / "results.json"
    ckpt_path = run_dir / "checkpoint.pt"

    if json_path.exists() and ckpt_path.exists():
        print(f"[SKIP] Run already complete: {run_name} (found {json_path})")
        return True

    cmd = [
        PYTHON_EXEC,
        str(REPO_ROOT / "run.py"),
        "--variant",
        variant,
        "--dataset",
        "mosi",
        "--data_path",
        str(REPO_ROOT / data_path),
        "--pretrained_model",
        "pretrained/mosei.pt",
        "--drop_rate",
        "0.7",
        "--batch_size",
        "16",
        "--accum_steps",
        "4",
        "--num_epochs",
        str(num_epochs),
        "--seed",
        str(seed),
        "--eval_fidelity",
        "--name",
        str(run_dir),
    ]

    print(f"\n{'='*75}")
    print(f"STARTING RUN: {run_name}")
    print(f"  Variant:    {variant}")
    print(f"  Data Path:  {data_path}")
    print(f"  Seed:       {seed}")
    print(f"  Epochs:     {num_epochs}")
    print(f"{'='*75}\n")

    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print(f"[ERROR] Run failed with returncode {ret.returncode}: {run_name}")
        return False

    print(f"[SUCCESS] Completed {run_name} in {elapsed:.1f}s ({elapsed/60:.2f}m)")
    return True


def main():
    args = parse_args()
    if args.cells.strip().lower() == "all":
        cells_to_run = ["l100", "l200", "native"]
    else:
        cells_to_run = [c.strip() for c in args.cells.split(",") if c.strip()]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    print(f"Executing cells: {cells_to_run} | Variants: {variants} | Seeds: {seeds}")

    total_runs = len(cells_to_run) * len(variants) * len(seeds)
    run_idx = 0

    for cell in cells_to_run:
        if cell not in CELL_CONFIGS:
            print(f"[WARNING] Unknown cell: {cell}, skipping.")
            continue

        cfg = CELL_CONFIGS[cell]
        for variant in variants:
            for seed in seeds:
                run_idx += 1
                run_name = f"runs/{cfg['runs_prefix']}_{variant}_seed{seed}"
                print(f"\n>>> [{run_idx}/{total_runs}] Processing {run_name}...")
                ok = run_single(
                    variant=variant,
                    data_path=cfg["data_path"],
                    seed=seed,
                    run_name=run_name,
                    num_epochs=args.num_epochs,
                )
                if not ok:
                    print(f"Aborting matrix execution due to failure in {run_name}")
                    sys.exit(1)

    print("\n" + "=" * 75)
    print("ALL REQUESTED RUNS FINISHED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
