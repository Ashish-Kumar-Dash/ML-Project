#!/usr/bin/env python3
"""Main execution entrypoint for running upstream MPLMM commands.

Provides runtime compatibility patching (torch.load weights_only=False,
ReduceLROnPlateau verbose removal, and path resolution) while keeping
third_party/MPLMM completely untouched and pristine.
"""

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 1. Compatibility layer: injects third_party/MPLMM into sys.path & wraps torch.load
import lenmod.compat

# 2. Patch ReduceLROnPlateau to safely ignore deprecated 'verbose' parameter in PyTorch >= 2.2
import torch
import torch.optim.lr_scheduler as lr_scheduler

_orig_reduce_lr_init = lr_scheduler.ReduceLROnPlateau.__init__


def _compat_reduce_lr_init(self, *args, **kwargs):
    kwargs.pop("verbose", None)
    return _orig_reduce_lr_init(self, *args, **kwargs)


lr_scheduler.ReduceLROnPlateau.__init__ = _compat_reduce_lr_init

# 3. Patch transfer_model to use our robust lenmod.transfer function
from src import utils as src_utils
from src import train as upstream_train
from lenmod.transfer import transfer_weights

def _compat_transfer(new_model, pretrained):
    model, _ = transfer_weights(new_model, pretrained, quiet=False)
    return model

src_utils.transfer_model = _compat_transfer
upstream_train.transfer_model = _compat_transfer

# 4. Patch train_model to ensure checkpoint directory exists and run both valid and test eval
from src.eval_metrics import eval_mosi, eval_mosei_senti

_orig_train_model = upstream_train.train_model


def _compat_train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    # Ensure checkpoint target directory exists
    if getattr(hyp_params, "name", None):
        ckpt_dir = Path(hyp_params.name).parent
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    result = _orig_train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

    # After upstream prints line 160 (which evaluates valid split test=False),
    # run test-split evaluation so the user gets true test performance
    if getattr(hyp_params, "name", None) and Path(hyp_params.name).exists():
        try:
            print("\n" + "=" * 60)
            print("==== TEST SPLIT EVALUATION (test=True) ====")
            model = torch.load(hyp_params.name, weights_only=False)
            model.eval()
            if hasattr(model, "missing_type_prompt"):
                assert (model.missing_type_prompt == 0).all(), (
                    "Assertion failed: model.missing_type_prompt is not all zeros after training!"
                )
                print("[PIPELINE CHECK] Asserted (model.missing_type_prompt == 0).all() — all zeros confirmed.")
            criterion = settings["criterion"]
            from lenmod.trainer import evaluate as lenmod_evaluate
            test_loss, metrics, results, truths = lenmod_evaluate(model, test_loader, criterion=criterion)
            print(f"Test Loss (MAE): {test_loss:.4f}")
            print(f"Binary Acc-2:    {metrics['acc2']*100:.2f}%")
            print(f"Weighted F1:     {metrics['f1']*100:.2f}%")
            print(f"Correlation:     {metrics['corr']:.4f}")
            print(f"Multiclass Acc-5:{metrics['mult_acc_5']*100:.2f}%")
            print(f"Multiclass Acc-7:{metrics['mult_acc_7']*100:.2f}%")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"Notice: post-training test evaluation encountered: {e}")

    return result


upstream_train.train_model = _compat_train_model


if __name__ == "__main__":
    # Execute upstream main logic directly
    import importlib.util
    upstream_main_path = REPO_ROOT / "third_party" / "MPLMM" / "main.py"
    spec = importlib.util.spec_from_file_location("__main__", str(upstream_main_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["__main__"] = mod
    spec.loader.exec_module(mod)
