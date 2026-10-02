"""Benchmark training epoch duration and compute cost across the sequence ladder.

Prices the ladder by measuring wall-clock time for one training epoch for both
the dense baseline and the length-invariant cross-attention generator across
all four ladder lengths:
  - L = 50   (50, 50, 50)
  - L = 100  (50, 100, 100)
  - L = 200  (50, 200, 200)
  - L = Native (50, 375, 500)

Usage:
    python tools/benchmark_epoch_timing.py
    python tools/benchmark_epoch_timing.py --num_timed_epochs 3 --output runs/ladder_timing_benchmark.json
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lenmod.compat
from lenmod.config import make_hp
from lenmod.operator import swap_time_mixers
from src.model import PromptModel
from src.mosidata import MOSIData
from src.utils import transfer_model

# Reset default tensor type to CPU FloatTensor (mosidata sets it to cuda)
torch.set_default_tensor_type("torch.FloatTensor")


LADDER_CONFIGS = [
    {
        "name": "L50",
        "label": "50 / 50 / 50",
        "file": "data/ladder/mosi_data_noalign_l50.pkl",
        "expected_lengths": (50, 50, 50),
    },
    {
        "name": "L100",
        "label": "50 / 100 / 100",
        "file": "data/ladder/mosi_data_noalign_l100.pkl",
        "expected_lengths": (50, 100, 100),
    },
    {
        "name": "L200",
        "label": "50 / 200 / 200",
        "file": "data/ladder/mosi_data_noalign_l200.pkl",
        "expected_lengths": (50, 200, 200),
    },
    {
        "name": "Native",
        "label": "50 / 375 / 500",
        "file": "data/ladder/mosi_data_noalign_native.pkl",
        "expected_lengths": (50, 375, 500),
    },
]


def get_git_commit(repo_path: Path) -> str:
    """Retrieve git HEAD commit hash."""
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        return "unknown"


def count_parameters(model: nn.Module) -> Tuple[int, int, int]:
    """Return (total_params, trainable_params, time_mixing_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count parameters in time-mixing layers (model.l_avp, model.a_lvp, etc.)
    tm_params = 0
    tm_names = [
        "l_avp", "a_lvp", "v_alp",
        "l_ap", "l_vp", "a_lp",
        "a_vp", "v_ap", "v_lp",
    ]
    for name in tm_names:
        if hasattr(model, name):
            module = getattr(model, name)
            tm_params += sum(p.numel() for p in module.parameters())

    return total, trainable, tm_params


def time_single_config(
    variant: str,
    data_file: Path,
    pretrained_path: Path,
    batch_size: int = 16,
    accum_steps: int = 4,
    num_timed_epochs: int = 3,
    device: str = "cuda:0",
    seed: int = 42,
) -> Dict[str, Any]:
    """Measure warmup and steady-state epoch times for a specific configuration."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. Dataset & Loader
    dataset = MOSIData(str(data_file), split_type="train", drop_rate=0.7)
    torch.set_default_tensor_type("torch.FloatTensor")

    valid_dataset = MOSIData(str(data_file), split_type="valid", drop_rate=0.7)
    torch.set_default_tensor_type("torch.FloatTensor")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    num_samples = len(dataset)
    num_batches = len(loader)

    L, A, V = dataset.get_seq_len()
    orig_dims = dataset.get_dim()

    # 2. Strict Build Order
    hp = make_hp(
        L=L,
        A=A,
        V=V,
        orig_dims=orig_dims,
        batch_size=batch_size,
        use_cuda=torch.cuda.is_available(),
    )
    model = PromptModel(hp)

    if variant == "invariant":
        model = swap_time_mixers(model, operator="cross_attn", num_heads=2)

    if pretrained_path.exists():
        transfer_model(model, str(pretrained_path))

    model = model.to(dev)

    total_p, trainable_p, tm_p = count_parameters(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    criterion = nn.L1Loss()

    def run_train_epoch() -> Tuple[float, float, float, float]:
        """Run one training epoch. Returns (total_time, fwd_time, bwd_time, opt_time)."""
        model.train()
        optimizer.zero_grad()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

        t_start = time.perf_counter()
        fwd_acc = 0.0
        bwd_acc = 0.0
        opt_acc = 0.0

        for step, (batch_X, batch_Y, missing_mod) in enumerate(loader):
            text = batch_X[0].to(dev)
            audio = batch_X[1].to(dev)
            vision = batch_X[2].to(dev)
            eval_attr = batch_Y.to(dev).squeeze(-1)

            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_fwd_0 = time.perf_counter()
            preds = model(text, audio, vision, missing_mod)
            loss = criterion(preds, eval_attr) / accum_steps
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            fwd_acc += time.perf_counter() - t_fwd_0

            t_bwd_0 = time.perf_counter()
            loss.backward()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            bwd_acc += time.perf_counter() - t_bwd_0

            if (step + 1) % accum_steps == 0 or (step + 1) == num_batches:
                t_opt_0 = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                optimizer.step()
                optimizer.zero_grad()
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                opt_acc += time.perf_counter() - t_opt_0

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_total = time.perf_counter() - t_start

        return t_total, fwd_acc, bwd_acc, opt_acc

    def run_val_epoch() -> float:
        """Run validation pass. Returns duration in seconds."""
        model.eval()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_val_0 = time.perf_counter()
        with torch.no_grad():
            for batch_X, batch_Y, missing_mod in valid_loader:
                text = batch_X[0].to(dev)
                audio = batch_X[1].to(dev)
                vision = batch_X[2].to(dev)
                eval_attr = batch_Y.to(dev).squeeze(-1)
                _ = model(text, audio, vision, missing_mod)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        return time.perf_counter() - t_val_0

    # Warmup epoch
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    warmup_time, _, _, _ = run_train_epoch()
    _ = run_val_epoch()

    # Timed steady-state epochs
    epoch_times = []
    fwd_times = []
    bwd_times = []
    opt_times = []

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    for _ in range(num_timed_epochs):
        tot, fwd, bwd, opt = run_train_epoch()
        epoch_times.append(tot)
        fwd_times.append(fwd)
        bwd_times.append(bwd)
        opt_times.append(opt)

    val_time = run_val_epoch()

    peak_mem_gib = (
        torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        if dev.type == "cuda"
        else 0.0
    )

    mean_epoch_time = float(np.mean(epoch_times))
    std_epoch_time = float(np.std(epoch_times))
    mean_fwd = float(np.mean(fwd_times))
    mean_bwd = float(np.mean(bwd_times))
    mean_opt = float(np.mean(opt_times))
    throughput = num_samples / mean_epoch_time
    full_40ep_minutes = (mean_epoch_time + val_time) * 40.0 / 60.0

    return {
        "variant": variant,
        "lengths": (L, A, V),
        "num_samples": num_samples,
        "num_batches": num_batches,
        "total_params": total_p,
        "trainable_params": trainable_p,
        "time_mixing_params": tm_p,
        "warmup_time_sec": warmup_time,
        "timed_epochs_sec": epoch_times,
        "mean_epoch_time_sec": mean_epoch_time,
        "std_epoch_time_sec": std_epoch_time,
        "mean_fwd_time_sec": mean_fwd,
        "mean_bwd_time_sec": mean_bwd,
        "mean_opt_time_sec": mean_opt,
        "val_time_sec": val_time,
        "throughput_samples_per_sec": throughput,
        "full_40_epoch_est_min": full_40ep_minutes,
        "peak_vram_gib": peak_mem_gib,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark epoch duration across the sequence ladder."
    )
    parser.add_argument(
        "--num_timed_epochs",
        type=int,
        default=3,
        help="Number of steady-state epochs to time per configuration (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="pretrained/mosei.pt",
        help="Pretrained backbone checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/ladder_timing_benchmark.json",
        help="Output path for JSON results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Execution device (default: cuda:0)",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    pretrained_path = repo_root / args.pretrained_model

    print("=" * 80)
    print("SEQUENCE LADDER EPOCH TIMING BENCHMARK")
    print(f"Device: {args.device} | Timed Epochs: {args.num_timed_epochs} | Batch Size: {args.batch_size}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Hardware: {gpu_name} ({vram_gib:.2f} GiB VRAM)")
    print("=" * 80)

    results = []

    for cfg in LADDER_CONFIGS:
        data_path = repo_root / cfg["file"]
        if not data_path.exists():
            print(f"Skipping {cfg['name']}: file {data_path} not found.")
            continue

        for variant in ["baseline", "invariant"]:
            print(
                f"\n>>> Benchmarking [{variant.upper()}] on {cfg['name']} ({cfg['label']})..."
            )
            res = time_single_config(
                variant=variant,
                data_file=data_path,
                pretrained_path=pretrained_path,
                batch_size=args.batch_size,
                accum_steps=args.accum_steps,
                num_timed_epochs=args.num_timed_epochs,
                device=args.device,
            )
            res["ladder_name"] = cfg["name"]
            res["ladder_label"] = cfg["label"]
            results.append(res)
            print(
                f"    Mean epoch: {res['mean_epoch_time_sec']:.3f}s ± {res['std_epoch_time_sec']:.3f}s "
                f"| Fwd: {res['mean_fwd_time_sec']:.3f}s | Bwd: {res['mean_bwd_time_sec']:.3f}s "
                f"| Peak VRAM: {res['peak_vram_gib']:.2f} GiB "
                f"| Throughput: {res['throughput_samples_per_sec']:.1f} samples/s"
            )

    # Print Summary Table
    print("\n" + "=" * 94)
    print(f"{'Ladder Rung':<12}{'Lengths':<14}{'Variant':<11}{'TM Params':>11}{'Epoch (s)':>11}{'Throughput':>14}{'Peak VRAM':>11}{'40-Ep Est':>10}")
    print("-" * 94)
    for r in results:
        v_label = "Dense" if r["variant"] == "baseline" else "Invariant"
        l_str = f"{r['lengths'][0]}/{r['lengths'][1]}/{r['lengths'][2]}"
        print(
            f"{r['ladder_name']:<12}{l_str:<14}{v_label:<11}"
            f"{r['time_mixing_params']:>11,d}"
            f"{r['mean_epoch_time_sec']:>10.3f}s"
            f"{r['throughput_samples_per_sec']:>10.1f} smp/s"
            f"{r['peak_vram_gib']:>9.2f} GiB"
            f"{r['full_40_epoch_est_min']:>8.1f} min"
        )
    print("=" * 94)

    # Relative Comparison Table (Pricing the ladder)
    print("\n" + "=" * 80)
    print("LADDER PRICING & RELATIVE OVERHEAD COMPARISON")
    print(f"{'Ladder Rung':<12}{'Lengths':<14}{'Dense (s)':>11}{'Invariant (s)':>15}{'Ratio (Inv/Dense)':>18}{'Dense Params':>14}{'Inv Params':>12}")
    print("-" * 80)

    by_ladder = {}
    for r in results:
        by_ladder.setdefault(r["ladder_name"], {})[r["variant"]] = r

    for name, pair in by_ladder.items():
        if "baseline" in pair and "invariant" in pair:
            b = pair["baseline"]
            inv = pair["invariant"]
            ratio = inv["mean_epoch_time_sec"] / b["mean_epoch_time_sec"]
            l_str = f"{b['lengths'][0]}/{b['lengths'][1]}/{b['lengths'][2]}"
            print(
                f"{name:<12}{l_str:<14}"
                f"{b['mean_epoch_time_sec']:>10.3f}s"
                f"{inv['mean_epoch_time_sec']:>14.3f}s"
                f"{ratio:>17.2f}x"
                f"{b['time_mixing_params']:>14,d}"
                f"{inv['time_mixing_params']:>12,d}"
            )
    print("=" * 80)

    # Save to JSON
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": "c6f8d18e9222bd65165a3214820201dc51a35f19",
        "project_commit": get_git_commit(repo_root),
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "num_timed_epochs": args.num_timed_epochs,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved timing results to {output_path}")


if __name__ == "__main__":
    main()
