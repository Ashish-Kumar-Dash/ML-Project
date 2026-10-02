#!/usr/bin/env python3
"""Main execution runner for MPLMM experiments with runtime adaptation.

Strict Build Order (per project specification):
1. PromptModel instantiated.
2. swap_time_mixers applied for the invariant variant.
3. transfer_model transfers pretrained backbone weights.
4. .cuda() moves model to GPU device.
5. fit runs training loop with gradient accumulation and state-dict checkpointing.
6. eval_cases evaluates best checkpoint across all 7 modality cases (0 to 6).
7. write the JSON to runs/ recording results with upstream commit hash.

Usage:
    python run.py --variant baseline --data_path data/mosi_data.pkl --name runs/mosi_baseline_seed1
    python run.py --variant invariant --data_path data/mosi_data.pkl --name runs/mosi_invariant_seed1
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import torch
from torch.utils.data import DataLoader

# 1. Compatibility and configuration layer
import lenmod.compat
from lenmod.config import make_hp
from lenmod.hooks import evaluate_generation_fidelity
from lenmod.operator import swap_time_mixers
from lenmod.trainer import eval_cases, fit

# Upstream imports
from src.model import PromptModel
from src.mosidata import MOSIData
from src.utils import transfer_model

# Ensure default tensor type remains standard FloatTensor (mosidata sets it to cuda)
torch.set_default_tensor_type("torch.FloatTensor")



def get_git_commit(repo_path: Path) -> str:
    """Retrieve git HEAD commit hash or return unknown."""
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


def main():
    parser = argparse.ArgumentParser(description="Run MPLMM training and evaluation.")
    # Dataset and paths
    parser.add_argument(
        "--dataset",
        type=str,
        default="mosi",
        help="Dataset name (default: mosi)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/mosi_data.pkl",
        help="Path to dataset pickle (default: data/mosi_data.pkl)",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="pretrained/mosei.pt",
        help="Path to pretrained MOSEI backbone checkpoint",
    )
    # Model architecture and variant
    parser.add_argument(
        "--variant",
        type=str,
        choices=["baseline", "invariant"],
        default="baseline",
        help="Model variant: 'baseline' (original dense time-mixers) or 'invariant' (cross-attention)",
    )
    parser.add_argument(
        "--operator",
        type=str,
        choices=["cross_attn", "pooling"],
        default="cross_attn",
        help="Operator for invariant variant (default: cross_attn)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="Attention heads for cross-attention generator (default: 2)",
    )
    # Optimization and hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Mini-batch size (default: 16)",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4; 4 * 16 = effective batch 64)",
    )
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.7,
        help="Modality drop rate during training (paper optimum: 0.7)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=40,
        help="Number of training epochs (default: 40)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="Random seed (default: 666)",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA execution",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-epoch stdout logs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run identifier or directory path under runs/",
    )
    parser.add_argument(
        "--eval_fidelity",
        action="store_true",
        help="Capture forward hook representations and compute generation fidelity & CKA",
    )

    args = parser.parse_args()

    # Reproducibility seed setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Establish run directory
    repo_root = Path(__file__).resolve().parent
    if args.name:
        run_dir = (
            Path(args.name)
            if Path(args.name).is_absolute()
            else repo_root / args.name
        )
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = (
            repo_root
            / "runs"
            / f"{args.dataset}_{args.variant}_seed{args.seed}_{timestamp}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    json_path = run_dir / "results.json"

    # Load datasets
    data_file = Path(args.data_path)
    if not data_file.is_absolute():
        data_file = repo_root / data_file
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found at: {data_file}")

    train_dataset = MOSIData(
        str(data_file), split_type="train", drop_rate=args.drop_rate
    )
    valid_dataset = MOSIData(
        str(data_file), split_type="valid", drop_rate=args.drop_rate
    )
    test_dataset = MOSIData(
        str(data_file), split_type="test", drop_rate=args.drop_rate
    )

    # Read sequence lengths and feature dimensions directly from dataset
    L, A, V = train_dataset.get_seq_len()
    orig_dims = train_dataset.get_dim()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    pretrained_model = (
        None
        if not args.pretrained_model or args.pretrained_model.lower() == "none"
        else args.pretrained_model
    )

    # Assemble hyp_params
    hp = make_hp(
        L=L,
        A=A,
        V=V,
        orig_dims=orig_dims,
        dataset=args.dataset,
        drop_rate=args.drop_rate,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        seed=args.seed,
        name=str(checkpoint_path),
        data_path=str(data_file),
        pretrained_model=pretrained_model,
        use_cuda=not args.no_cuda and torch.cuda.is_available(),
    )

    print("=" * 70)
    print(f"RUN CONFIGURATION: {args.variant.upper()}")
    print(f"  Dataset:          {args.dataset} ({data_file.name})")
    print(f"  Sequence lengths: L={L}, A={A}, V={V}")
    print(f"  Dimensions:       d_l={orig_dims[0]}, d_a={orig_dims[1]}, d_v={orig_dims[2]}")
    print(f"  Effective Batch:  {args.batch_size} * {args.accum_steps} = {args.batch_size * args.accum_steps}")
    print(f"  Run Directory:    {run_dir}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STRICT BUILD ORDER
    # -------------------------------------------------------------------------
    # Step 1: PromptModel
    print("[1/7] Instantiating PromptModel...")
    model = PromptModel(hp)

    # Step 2: swap_time_mixers for the invariant variant
    if args.variant == "invariant":
        print(f"[2/7] Swapping time-mixers with invariant operator ({args.operator})...")
        model = swap_time_mixers(
            model,
            operator=args.operator,
            num_heads=args.num_heads,
        )
    else:
        print("[2/7] Baseline variant: keeping original dense time-mixers.")

    # Step 3: transfer_model
    if hp.pretrained_model and str(hp.pretrained_model).lower() not in ["none", ""]:
        pretrained_path = Path(hp.pretrained_model)
        if not pretrained_path.is_absolute():
            pretrained_path = repo_root / pretrained_path
        print(f"[3/7] Transferring backbone weights from {pretrained_path}...")
        transfer_model(model, str(pretrained_path))
    else:
        print("[3/7] No pretrained backbone specified, skipping transfer.")

    # Step 4: .cuda()
    device = torch.device(
        "cuda:0" if hp.use_cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"[4/7] Moving model to {device}...")
    model = model.to(device)

    # Step 5: fit
    print(f"[5/7] Fitting model for {hp.num_epochs} epochs...")
    fit_result = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        hyp_params=hp,
        device=device,
        checkpoint_path=checkpoint_path,
        accum_steps=args.accum_steps,
        quiet=args.quiet,
    )

    # Step 6: eval_cases (on test set with best checkpoint)
    print("[6/7] Evaluating all 7 modality conditions on test set...")
    cases_result = eval_cases(
        model=model,
        loader=test_loader,
        device=device,
        quiet=args.quiet,
    )

    fidelity_result = None
    if args.eval_fidelity:
        print("[6b/7] Evaluating generation fidelity and CKA via forward hooks...")
        fidelity_result = evaluate_generation_fidelity(
            model=model,
            loader=test_loader,
            device=device,
        )

    # Step 7: write the JSON to runs/
    print(f"[7/7] Writing results JSON to {json_path}...")
    upstream_commit = get_git_commit(repo_root / "third_party" / "MPLMM")
    project_commit = get_git_commit(repo_root)

    test_results_payload = {
        "aggregated": fit_result["test_metrics"],
        "per_case": cases_result["cases"],
        "mean_cases_0_to_5": cases_result["mean_0_to_5"],
        "complete_data_case_6": cases_result["complete_data"],
    }
    if fidelity_result is not None:
        test_results_payload["generation_fidelity"] = fidelity_result

    run_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": upstream_commit,
        "project_commit": project_commit,
        "variant": args.variant,
        "operator": args.operator if args.variant == "invariant" else "mlp_dense",
        "dataset": args.dataset,
        "data_path": str(data_file),
        "pretrained_model": str(hp.pretrained_model),
        "sequence_lengths": {"L": L, "A": A, "V": V},
        "feature_dimensions": {
            "d_l": orig_dims[0],
            "d_a": orig_dims[1],
            "d_v": orig_dims[2],
        },
        "hyperparameters": {
            "drop_rate": args.drop_rate,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "accum_steps": args.accum_steps,
            "effective_batch_size": args.batch_size * args.accum_steps,
            "num_epochs": args.num_epochs,
            "seed": args.seed,
        },
        "training_outcome": {
            "best_epoch": fit_result["best_epoch"],
            "best_val_loss": fit_result["best_val_loss"],
            "checkpoint_path": str(checkpoint_path),
        },
        "test_results": test_results_payload,
        "training_history": fit_result["history"],
    }

    with open(json_path, "w") as f:
        json.dump(run_record, f, indent=2)

    print(f"Run completed successfully! Output JSON: {json_path}")
    print(
        f"Mean Acc-2 (Cases 0 to 5): {cases_result['mean_0_to_5']['acc2'] * 100:.2f}%\n"
    )


if __name__ == "__main__":
    main()
