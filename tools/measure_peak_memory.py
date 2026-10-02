"""Measure peak CUDA memory allocated during forward and backward passes.

Verifies the memory footprint and tests the 12 GB OOM claim for CMU-MOSI
unaligned lengths (L=50, A=375, V=500) at batch size 16 vs batch size 64.

Usage:
    python tools/measure_peak_memory.py --batch_size 16
    python tools/measure_peak_memory.py --batch_size 64
    python tools/measure_peak_memory.py --all
"""

import argparse
import gc
from pathlib import Path
import subprocess
import sys
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lenmod.compat
from lenmod.config import make_hp
from lenmod.operator import swap_time_mixers
from src.model import PromptModel


def measure_single_batch(
    batch_size: int,
    variant: str = "baseline",
    len_l: int = 50,
    len_a: int = 375,
    len_v: int = 500,
    dim_l: int = 300,
    dim_a: int = 5,
    dim_v: int = 20,
    device_id: int = 0,
) -> dict:
    """Measure peak memory for a single batch configuration on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

    device = torch.device(f"cuda:{device_id}")
    device_prop = torch.cuda.get_device_properties(device)
    total_memory_gib = device_prop.total_memory / (1024 ** 3)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 1. Model creation
    hp = make_hp(
        L=len_l,
        A=len_a,
        V=len_v,
        orig_dims=(dim_l, dim_a, dim_v),
        batch_size=batch_size,
        use_cuda=True,
    )
    model = PromptModel(hp)
    if variant == "invariant":
        model = swap_time_mixers(model, operator="cross_attn", query_len=50)

    model = model.to(device)
    model.train()

    model_mem_gib = torch.cuda.memory_allocated(device) / (1024 ** 3)

    # 2. Fake inputs
    x_l = torch.randn(batch_size, len_l, dim_l, device=device)
    x_a = torch.randn(batch_size, len_a, dim_a, device=device)
    x_v = torch.randn(batch_size, len_v, dim_v, device=device)
    missing_mod = [i % 7 for i in range(batch_size)]

    fwd_peak_gib = float("nan")
    bwd_peak_gib = float("nan")
    status = "SUCCESS"
    error_msg = ""

    # 3. Forward pass
    try:
        torch.cuda.reset_peak_memory_stats(device)
        out = model(x_l, x_a, x_v, missing_mod)
        fwd_peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    except torch.cuda.OutOfMemoryError as e:
        status = "OOM_FORWARD"
        error_msg = str(e)
        return {
            "batch_size": batch_size,
            "variant": variant,
            "total_gpu_gib": total_memory_gib,
            "model_mem_gib": model_mem_gib,
            "fwd_peak_gib": fwd_peak_gib,
            "bwd_peak_gib": bwd_peak_gib,
            "status": status,
            "error": error_msg,
        }
    except Exception as e:
        status = "ERROR_FORWARD"
        error_msg = str(e)
        return {
            "batch_size": batch_size,
            "variant": variant,
            "total_gpu_gib": total_memory_gib,
            "model_mem_gib": model_mem_gib,
            "fwd_peak_gib": fwd_peak_gib,
            "bwd_peak_gib": bwd_peak_gib,
            "status": status,
            "error": error_msg,
        }

    # 4. Backward pass (simulating optimizer step gradient calculation)
    try:
        loss = out.sum()
        loss.backward()
        bwd_peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    except torch.cuda.OutOfMemoryError as e:
        status = "OOM_BACKWARD"
        error_msg = str(e)
    except Exception as e:
        status = "ERROR_BACKWARD"
        error_msg = str(e)

    return {
        "batch_size": batch_size,
        "variant": variant,
        "total_gpu_gib": total_memory_gib,
        "model_mem_gib": model_mem_gib,
        "fwd_peak_gib": fwd_peak_gib,
        "bwd_peak_gib": bwd_peak_gib,
        "status": status,
        "error": error_msg,
    }


def run_in_subprocess(
    batch_size: int,
    variant: str,
    len_l: int = 50,
    len_a: int = 375,
    len_v: int = 500,
) -> dict:
    """Run single configuration measurement in an isolated subprocess."""
    code = f"""
import sys
from pathlib import Path
repo_dir = '{str(REPO_ROOT)}'
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
import json
from tools.measure_peak_memory import measure_single_batch
res = measure_single_batch(
    batch_size={batch_size},
    variant='{variant}',
    len_l={len_l},
    len_a={len_a},
    len_v={len_v},
)
print('__RESULT__' + json.dumps(res))
"""
    cmd = [sys.executable, "-c", code]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            import json
            return json.loads(line[len("__RESULT__"):])

    # If it crashed without emitting result (e.g. fatal CUDA OOM / segfault)
    return {
        "batch_size": batch_size,
        "variant": variant,
        "total_gpu_gib": 12.0,
        "model_mem_gib": float("nan"),
        "fwd_peak_gib": float("nan"),
        "bwd_peak_gib": float("nan"),
        "status": "OOM_CRASH" if "out of memory" in proc.stderr.lower() else "CRASH",
        "error": proc.stderr.strip()[:300],
    }


def main():
    parser = argparse.ArgumentParser(description="Measure peak CUDA memory for unaligned MOSI.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (e.g. 16, 32, 64)")
    parser.add_argument("--variant", type=str, choices=["baseline", "invariant"], default="baseline")
    parser.add_argument("--all", action="store_true", help="Run full comparison matrix across batch 16 and batch 64")
    parser.add_argument("--ladder", action="store_true", help="Run peak memory measurement across all ladder rungs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA device not available.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print("=" * 78)
    print(f"CUDA MEMORY BENCHMARK")
    print(f"Device: {device_name} ({total_mem:.2f} GiB total VRAM)")
    print("=" * 78)

    if args.ladder:
        ladder_configs = [
            ("L50", (50, 50, 50)),
            ("L100", (50, 100, 100)),
            ("L200", (50, 200, 200)),
            ("Native", (50, 375, 500)),
        ]
        header = f"| {'Rung':<8} | {'Lengths':<14} | {'Variant':<10} | {'Batch':<6} | {'Fwd Peak':<11} | {'Bwd Peak':<11} | {'Status':<14} |"
        sep = "|" + "-"*10 + "|" + "-"*16 + "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*16 + "|"
        print(header)
        print(sep)

        for name, (l, a, v) in ladder_configs:
            l_str = f"{l}/{a}/{v}"
            for bsz in [16, 64]:
                for var in ["baseline", "invariant"]:
                    res = run_in_subprocess(bsz, var, len_l=l, len_a=a, len_v=v)
                    f_gib = f"{res['fwd_peak_gib']:.2f} GiB" if not str(res['fwd_peak_gib']) == "nan" else "N/A"
                    b_gib = f"{res['bwd_peak_gib']:.2f} GiB" if not str(res['bwd_peak_gib']) == "nan" else "N/A"
                    stat = res["status"]
                    print(f"| {name:<8} | {l_str:<14} | {var:<10} | {bsz:<6} | {f_gib:<11} | {b_gib:<11} | {stat:<14} |")
        print("=" * 78)
    elif args.all:
        configs = [
            ("baseline", 16),
            ("baseline", 64),
            ("invariant", 16),
            ("invariant", 64),
        ]
        header = f"| {'Variant':<10} | {'Batch':<6} | {'Model GiB':<10} | {'Fwd Peak GiB':<13} | {'Bwd Peak GiB':<13} | {'Status':<14} |"
        sep = "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*12 + "|" + "-"*15 + "|" + "-"*15 + "|" + "-"*16 + "|"
        print(header)
        print(sep)

        for var, bsz in configs:
            res = run_in_subprocess(bsz, var)
            m_gib = f"{res['model_mem_gib']:.3f}" if not str(res['model_mem_gib']) == "nan" else "N/A"
            f_gib = f"{res['fwd_peak_gib']:.3f}" if not str(res['fwd_peak_gib']) == "nan" else "N/A"
            b_gib = f"{res['bwd_peak_gib']:.3f}" if not str(res['bwd_peak_gib']) == "nan" else "N/A"
            stat = res["status"]
            print(f"| {var:<10} | {bsz:<6} | {m_gib:<10} | {f_gib:<13} | {b_gib:<13} | {stat:<14} |")
            if res.get("error"):
                print(f"  --> Note: {res['error'][:120]}...")

        print("=" * 78)
    else:
        res = measure_single_batch(args.batch_size, args.variant)
        print(f"Variant:          {res['variant']}")
        print(f"Batch Size:       {res['batch_size']}")
        print(f"Model Static Mem: {res['model_mem_gib']:.3f} GiB")
        print(f"Forward Peak:     {res['fwd_peak_gib']:.3f} GiB")
        print(f"Backward Peak:    {res['bwd_peak_gib']:.3f} GiB")
        print(f"Status:           {res['status']}")
        if res.get("error"):
            print(f"Error:            {res['error']}")
        print("=" * 78)


if __name__ == "__main__":
    main()
