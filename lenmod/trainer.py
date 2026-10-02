"""Trainer module for MPLMM models and length-invariant missing-modality generation.

Key design requirements:
1. `fit` and `evaluate` decoupled from upstream monolithic train.py.
2. Gradient accumulation: configurable (default accum=4 at batch_size=16,
   yielding effective batch size 64 matching upstream while saving GPU VRAM).
3. No DataParallel: direct, clean single-GPU execution.
4. State-dict checkpoints: saves and reloads `model.state_dict()` rather than
   pickling full module instances, avoiding serialization pitfalls.
5. Test-set reporting only: reports final metrics exclusively on the test split
   evaluated with the best validation checkpoint.
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import lenmod.compat  # noqa: F401 - registers upstream path & safe torch.load


def compute_metrics(
    preds_t: torch.Tensor, truths_t: torch.Tensor, exclude_zero: bool = True
) -> Dict[str, float]:
    """Compute standard MOSI / MOSEI regression and classification metrics."""
    preds = preds_t.view(-1).cpu().detach().numpy()
    truths = truths_t.view(-1).cpu().detach().numpy()

    mae = float(np.mean(np.abs(preds - truths)))

    # Correlation coefficient with guard against zero-variance slices
    if len(preds) > 1 and np.std(preds) > 1e-7 and np.std(truths) > 1e-7:
        corr = float(np.corrcoef(preds, truths)[0][1])
    else:
        corr = 0.0

    # Multiclass accuracies (rounded to nearest integer sentiment score)
    preds_a7 = np.clip(preds, -3.0, 3.0)
    truths_a7 = np.clip(truths, -3.0, 3.0)
    preds_a5 = np.clip(preds, -2.0, 2.0)
    truths_a5 = np.clip(truths, -2.0, 2.0)

    mult_a7 = float(
        np.sum(np.round(preds_a7) == np.round(truths_a7)) / float(len(truths))
    )
    mult_a5 = float(
        np.sum(np.round(preds_a5) == np.round(truths_a5)) / float(len(truths))
    )

    # Binary accuracy excluding zero labels (the canonical paper target 72.14)
    non_zeros = np.array(
        [i for i, e in enumerate(truths) if e != 0 or (not exclude_zero)]
    )
    if len(non_zeros) > 0:
        binary_truth = truths[non_zeros] > 0
        binary_preds = preds[non_zeros] > 0
        acc2 = float(accuracy_score(binary_truth, binary_preds))
        f1 = float(f1_score(binary_truth, binary_preds, average="weighted"))
    else:
        acc2 = 0.0
        f1 = 0.0

    # Binary accuracy including zero (non-negative classification)
    acc2_with_zero = float(accuracy_score(truths >= 0, preds >= 0))

    return {
        "mae": mae,
        "corr": corr,
        "acc2": acc2,
        "acc2_with_zero": acc2_with_zero,
        "f1": f1,
        "mult_acc_5": mult_a5,
        "mult_acc_7": mult_a7,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    forced_missing_code: Optional[int] = None,
) -> Tuple[float, Dict[str, float], torch.Tensor, torch.Tensor]:
    """Evaluate model on a DataLoader.

    Args:
        model: PyTorch model instance
        loader: DataLoader to evaluate
        criterion: Loss criterion (defaults to nn.L1Loss)
        device: Torch device (defaults to model device or cuda/cpu)
        forced_missing_code: Optional integer [0..6] to force specific missing modality
                             condition for all samples in the evaluation pass.

    Returns:
        Tuple of (average_loss, metrics_dict, all_predictions, all_ground_truths)
    """
    if criterion is None:
        criterion = nn.L1Loss()
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_truths = []

    with torch.no_grad():
        for batch_X, batch_Y, missing_mod in loader:
            text, audio, vision = (
                batch_X[0].to(device),
                batch_X[1].to(device),
                batch_X[2].to(device),
            )
            eval_attr = batch_Y.to(device).squeeze(-1)
            batch_sz = text.size(0)

            if forced_missing_code is not None:
                missing_mod = [forced_missing_code] * batch_sz

            preds = model(text, audio, vision, missing_mod)
            loss = criterion(preds, eval_attr)

            total_loss += loss.item() * batch_sz
            total_samples += batch_sz

            all_preds.append(preds.detach())
            all_truths.append(eval_attr.detach())

    avg_loss = total_loss / max(1, total_samples)
    cat_preds = torch.cat(all_preds, dim=0)
    cat_truths = torch.cat(all_truths, dim=0)

    metrics = compute_metrics(cat_preds, cat_truths, exclude_zero=True)
    metrics["loss"] = avg_loss

    return avg_loss, metrics, cat_preds, cat_truths


MODALITY_CASE_NAMES = {
    0: "L_missing ({a,v} present)",
    1: "A_missing ({l,v} present)",
    2: "V_missing ({l,a} present)",
    3: "LA_missing ({v} present)",
    4: "LV_missing ({a} present)",
    5: "AV_missing ({l} present)",
    6: "complete_data ({l,a,v} present)",
}


def eval_cases(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Evaluate model under all seven forced missing codes (0 to 6),

    computing per-case metrics and the mean across the six missing-modality cases (0 to 5).

    Args:
        model: PyTorch model instance
        loader: DataLoader (typically the test loader)
        criterion: Loss function
        device: Device to run evaluation on
        quiet: If True, suppress printed table

    Returns:
        Dictionary with:
            - 'cases': mapping of code [0..6] -> metrics dict
            - 'mean_0_to_5': dict of averaged metrics across cases 0 through 5
            - 'complete_data': metrics dict for case 6
    """
    if criterion is None:
        criterion = nn.L1Loss()
    if device is None:
        device = next(model.parameters()).device

    case_results = {}
    for code in range(7):
        loss, metrics, _, _ = evaluate(
            model,
            loader,
            criterion=criterion,
            device=device,
            forced_missing_code=code,
        )
        case_results[code] = metrics

    # Compute mean over the 6 missing-modality cases (0 to 5)
    metric_keys = ["acc2", "f1", "mae", "corr", "mult_acc_5", "mult_acc_7", "loss"]
    mean_0_to_5 = {}
    for k in metric_keys:
        values = [case_results[c][k] for c in range(6)]
        mean_0_to_5[k] = float(np.mean(values))

    if not quiet:
        print("\n" + "=" * 76)
        print("MODALITY EVALUATION SUMMARY (Forced Missing Codes 0 to 6)")
        print(f"{'Case':<32}{'Acc-2':>9}{'F1':>9}{'MAE':>9}{'Corr':>9}{'Acc-5':>8}")
        print("-" * 76)
        for code in range(6):
            name = f"Case {code}: {MODALITY_CASE_NAMES[code]}"
            m = case_results[code]
            print(
                f"{name:<32}{m['acc2']*100:>8.2f}%{m['f1']*100:>8.2f}%"
                f"{m['mae']:>9.4f}{m['corr']:>9.4f}{m['mult_acc_5']*100:>7.2f}%"
            )
        print("-" * 76)
        m_mean = mean_0_to_5
        print(
            f"{'MEAN (Cases 0 to 5)':<32}{m_mean['acc2']*100:>8.2f}%{m_mean['f1']*100:>8.2f}%"
            f"{m_mean['mae']:>9.4f}{m_mean['corr']:>9.4f}{m_mean['mult_acc_5']*100:>7.2f}%"
        )
        print("-" * 76)
        m6 = case_results[6]
        name6 = f"Case 6: {MODALITY_CASE_NAMES[6]}"
        print(
            f"{name6:<32}{m6['acc2']*100:>8.2f}%{m6['f1']*100:>8.2f}%"
            f"{m6['mae']:>9.4f}{m6['corr']:>9.4f}{m6['mult_acc_5']*100:>7.2f}%"
        )
        print("=" * 76 + "\n")

    return {
        "cases": case_results,
        "mean_0_to_5": mean_0_to_5,
        "complete_data": case_results[6],
    }



def fit(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    hyp_params: Any,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    accum_steps: int = 4,
    optimizer: Optional[optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Train model with gradient accumulation, save state-dict checkpoints,

    and report final performance strictly on the test set.

    Args:
        model: PyTorch model instance (PromptModel or MULTModel)
        train_loader: Training DataLoader (recommended batch_size=16)
        valid_loader: Validation DataLoader
        test_loader: Test DataLoader
        hyp_params: Namespace with hyperparameters (lr, num_epochs, clip, etc.)
        device: Torch execution device (never uses DataParallel)
        checkpoint_path: Where to save best state-dict checkpoint
        accum_steps: Gradient accumulation steps (default 4; 4 * 16 = effective batch 64)
        optimizer: Optional custom optimizer
        criterion: Optional custom loss function
        scheduler: Optional learning rate scheduler
        quiet: If True, suppresses per-epoch stdout printing

    Returns:
        Dictionary containing best_epoch, best_val_loss, test_metrics, and history.
    """
    if device is None:
        use_cuda = getattr(hyp_params, "use_cuda", torch.cuda.is_available())
        device = torch.device(
            "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
        )

    # Clean single-device placement, strictly no DataParallel
    model = model.to(device)

    # Initialize optimizer for trainable parameters only
    if optimizer is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt_name = getattr(hyp_params, "optim", "Adam")
        opt_cls = getattr(optim, opt_name, optim.Adam)
        lr = getattr(hyp_params, "lr", 1e-3)
        optimizer = opt_cls(trainable, lr=lr)

    # Initialize criterion
    if criterion is None:
        crit_name = getattr(hyp_params, "criterion", "L1Loss")
        crit_cls = getattr(nn, crit_name, nn.L1Loss)
        criterion = crit_cls()

    # Initialize scheduler (mode='min' on validation loss, without deprecated verbose)
    if scheduler is None:
        when = getattr(hyp_params, "when", 10)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=when, factor=0.1
        )

    # Setup checkpoint path
    if checkpoint_path is None:
        name = getattr(hyp_params, "name", None)
        checkpoint_path = Path(name) if name else Path("runs/checkpoint.pt")
    else:
        checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    num_epochs = getattr(hyp_params, "num_epochs", 40)
    clip_val = getattr(hyp_params, "clip", 0.8)

    best_val_loss = float("inf")
    best_epoch = -1
    history = []

    if not quiet:
        print(f"Starting training on {device} for {num_epochs} epochs...")
        print(f"Gradient accumulation: {accum_steps} steps (no DataParallel)")
        print(f"Checkpoint target: {checkpoint_path}")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()

        train_loss_sum = 0.0
        train_samples = 0

        for step, (batch_X, batch_Y, missing_mod) in enumerate(train_loader):
            text, audio, vision = (
                batch_X[0].to(device),
                batch_X[1].to(device),
                batch_X[2].to(device),
            )
            eval_attr = batch_Y.to(device).squeeze(-1)
            batch_sz = text.size(0)

            preds = model(text, audio, vision, missing_mod)
            loss = criterion(preds, eval_attr)

            # Accumulate scaled gradient
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            train_loss_sum += loss.item() * batch_sz
            train_samples += batch_sz

            # Step optimizer every accum_steps batches or at epoch boundary
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_val
                    )
                optimizer.step()
                optimizer.zero_grad()

        train_loss = train_loss_sum / max(1, train_samples)

        # Validation evaluation
        val_loss, val_metrics, _, _ = evaluate(
            model, valid_loader, criterion, device=device
        )
        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_duration = time.time() - epoch_start
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save lightweight state_dict checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

        history.append(
            {
                "epoch": epoch,
                "duration": epoch_duration,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc2": val_metrics["acc2"],
                "val_f1": val_metrics["f1"],
                "is_best": is_best,
            }
        )

        if not quiet:
            marker = " *" if is_best else ""
            print(
                f"Epoch {epoch:2d}/{num_epochs:2d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc-2: {val_metrics['acc2']:.4f} | "
                f"Time: {epoch_duration:.2f}s{marker}"
            )

    # -------------------------------------------------------------------------
    # Test-Set Reporting Only: reload best checkpoint and evaluate strictly on test
    # -------------------------------------------------------------------------
    if not quiet:
        print("\n" + "=" * 65)
        print(
            f"Reloading best checkpoint from Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})"
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    test_loss, test_metrics, _, _ = evaluate(
        model, test_loader, criterion, device=device
    )

    # Comprehensive evaluation across all 7 modality conditions
    test_cases_result = eval_cases(
        model, test_loader, criterion, device=device, quiet=quiet
    )

    if not quiet:
        print("=" * 65)
        print("FINAL TEST-SET EVALUATION RESULTS (Test Split Only)")
        print(f"  Test Loss:        {test_metrics['loss']:.4f}")
        print(f"  Binary Acc-2:     {test_metrics['acc2'] * 100:.2f}%")
        print(f"  Weighted F1:      {test_metrics['f1'] * 100:.2f}%")
        print(f"  MAE:              {test_metrics['mae']:.4f}")
        print(f"  Corr:             {test_metrics['corr']:.4f}")
        print(f"  Multiclass Acc-7: {test_metrics['mult_acc_7'] * 100:.2f}%")
        print(f"  Multiclass Acc-5: {test_metrics['mult_acc_5'] * 100:.2f}%")
        print(f"  Cases 0-5 Mean Acc-2: {test_cases_result['mean_0_to_5']['acc2'] * 100:.2f}%")
        print("=" * 65 + "\n")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(checkpoint_path),
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_cases": test_cases_result["cases"],
        "test_mean_0_to_5": test_cases_result["mean_0_to_5"],
        "history": history,
    }


__all__ = ["fit", "evaluate", "eval_cases", "compute_metrics", "MODALITY_CASE_NAMES"]

