#!/usr/bin/env python3
"""Summarizes all runs across sequence length ladder cells and outputs one-line summaries for Ash.

Usage:
  python tools/summarize_cells.py
"""

from collections import defaultdict
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

CELL_MAP = {
    "mosi_baseline": ("aligned_l50", "baseline"),
    "mosi_invariant": ("aligned_l50", "invariant"),
    "mosi_ladder_l50_baseline": ("ladder_l50", "baseline"),
    "mosi_ladder_l50_invariant": ("ladder_l50", "invariant"),
    "mosi_ladder_l100_baseline": ("ladder_l100", "baseline"),
    "mosi_ladder_l100_invariant": ("ladder_l100", "invariant"),
    "mosi_ladder_l200_baseline": ("ladder_l200", "baseline"),
    "mosi_ladder_l200_invariant": ("ladder_l200", "invariant"),
    "mosi_noalign_baseline": ("native_375_500", "baseline"),
    "mosi_noalign_invariant": ("native_375_500", "invariant"),
    "mosi_l2prime_baseline": ("l2_prime_5_20", "baseline"),
    "mosi_l2prime_invariant": ("l2_prime_5_20", "invariant"),
    "mosi_l2_baseline": ("l2_full_768_512", "baseline"),
    "mosi_l2_invariant": ("l2_full_768_512", "invariant"),
}


def load_all_runs(runs_dir_name: str = "runs"):
    runs_dir = REPO_ROOT / runs_dir_name
    json_files = sorted(runs_dir.glob("*/results.json"))

    cell_data = defaultdict(lambda: defaultdict(list))

    for jf in json_files:
        run_name = jf.parent.name
        if "smoke" in run_name or "fake" in run_name:
            continue
        # Match against known cell prefixes
        matched = False
        for prefix, (cell, variant) in CELL_MAP.items():
            if run_name.startswith(prefix):
                try:
                    with open(jf) as f:
                        data = json.load(f)
                    cell_data[cell][variant].append({
                        "run_name": run_name,
                        "data": data,
                    })
                    matched = True
                    break
                except Exception as e:
                    print(f"Error reading {jf}: {e}")
        if not matched:
            pass

    return cell_data


def format_cell_summary(cell_data):
    lines = []
    lines.append("=" * 95)
    lines.append(f"{'CELL / OPERATOR':<25} | {'RUN COUNT':<10} | {'MEAN ACC-2 (0-5)':<18} | {'COMPLETE ACC-2 (6)':<20} | {'TEST MAE':<12}")
    lines.append("=" * 95)

    one_liners = []

    for cell, variants in cell_data.items():
        cell_summary_parts = []
        for variant in ["baseline", "invariant"]:
            runs = variants.get(variant, [])
            if not runs:
                continue

            acc05 = [r["data"]["test_results"]["mean_cases_0_to_5"]["acc2"] * 100 for r in runs]
            acc_c6 = [r["data"]["test_results"]["complete_data_case_6"]["acc2"] * 100 for r in runs]
            mae = [r["data"]["test_results"]["aggregated"]["mae"] for r in runs]

            m_05 = np.mean(acc05)
            s_05 = np.std(acc05, ddof=1) if len(acc05) > 1 else 0.0

            m_c6 = np.mean(acc_c6)
            s_c6 = np.std(acc_c6, ddof=1) if len(acc_c6) > 1 else 0.0

            m_mae = np.mean(mae)
            s_mae = np.std(mae, ddof=1) if len(mae) > 1 else 0.0

            n = len(runs)
            label = f"{cell} [{variant}]"
            lines.append(f"{label:<25} | {n:<10d} | {m_05:5.2f}% ± {s_05:4.2f}%   | {m_c6:5.2f}% ± {s_c6:4.2f}%     | {m_mae:.4f} ± {s_mae:.4f}")

            cell_summary_parts.append(
                f"{variant}: Mean0-5={m_05:.2f}% (Complete={m_c6:.2f}%, MAE={m_mae:.4f}, n={n})"
            )

        if cell_summary_parts:
            one_liners.append(f"• Cell [{cell}]: " + " vs ".join(cell_summary_parts))

    lines.append("=" * 95)
    return "\n".join(lines), one_liners


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Summarize runs by cell.")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory containing run results (default: runs)")
    args = parser.parse_args()

    cell_data = load_all_runs(args.runs_dir)
    table_str, one_liners = format_cell_summary(cell_data)

    print(table_str)
    print("\n" + "=" * 95)
    print("ONE-LINE SUMMARY PER CELL (FOR ASH):")
    print("=" * 95)
    for ol in one_liners:
        print(ol)
    print("=" * 95)


if __name__ == "__main__":
    main()
