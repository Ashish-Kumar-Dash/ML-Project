"""Runtime compatibility layer for upstream MPLMM.

1. Automatically ensures `third_party/MPLMM` is in `sys.path` so that
   upstream modules (`src.model`, `src.utils`, `modules`, etc.) are importable.
2. Wraps `torch.load` so that `weights_only=False` is used by default,
   allowing loading of full pickled PyTorch modules (e.g. MULTModel) across
   PyTorch >= 2.6 without modifying upstream code on disk.
"""

import functools
import inspect
from pathlib import Path
import sys
import torch

# 1. Register upstream path
REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_PATH = REPO_ROOT / "third_party" / "MPLMM"

if str(UPSTREAM_PATH) not in sys.path:
    # Prepend to ensure precedence for upstream modules
    sys.path.insert(0, str(UPSTREAM_PATH))

# 2. torch.load wrapper
_orig_torch_load = torch.load
_sig = inspect.signature(_orig_torch_load)
_supports_weights_only = "weights_only" in _sig.parameters


@functools.wraps(_orig_torch_load)
def safe_torch_load(f, *args, **kwargs):
    """Wrapped torch.load that defaults weights_only=False for full module pickles."""
    if _supports_weights_only and "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(f, *args, **kwargs)


def patch_torch_load():
    """Install safe_torch_load onto torch.load."""
    torch.load = safe_torch_load


def restore_torch_load():
    """Restore original torch.load."""
    torch.load = _orig_torch_load


# Apply monkey-patch on import to protect upstream utils.py and train.py calls
patch_torch_load()

torch_load = safe_torch_load

__all__ = [
    "UPSTREAM_PATH",
    "safe_torch_load",
    "torch_load",
    "patch_torch_load",
    "restore_torch_load",
]
