"""Checkpoint transfer utility for loading pretrained backbone weights.

Handles both pickled PyTorch modules and state_dict checkpoints,
supporting safe loading across PyTorch versions, excluding dimension-dependent
projection and classification layers, and freezing transferred backbone parameters.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

EXCLUDED_KEYS = {
    "proj_l.weight",
    "proj_a.weight",
    "proj_v.weight",
    "out_layer.weight",
    "out_layer.bias",
}


def load_checkpoint_dict(checkpoint: Union[str, Path, nn.Module, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Extract a state dictionary from a file path, nn.Module, or dict."""
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        return checkpoint

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Safe torch.load supporting both whole pickled models and state dicts
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, nn.Module):
        return obj.state_dict()
    elif isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"]
        elif "model_state_dict" in obj:
            return obj["model_state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        return obj
    else:
        raise ValueError(f"Unrecognized checkpoint object type: {type(obj)}")


def transfer_weights(
    new_model: nn.Module,
    pretrained: Union[str, Path, nn.Module, Dict[str, Any]],
    excluded_keys: Optional[set] = None,
    quiet: bool = False,
) -> Tuple[nn.Module, List[str]]:
    """Transfer pretrained backbone weights into new_model and freeze transferred parameters.

    Args:
        new_model: Target model (e.g. PromptModel)
        pretrained: Path to checkpoint, state_dict dict, or nn.Module instance
        excluded_keys: Set of key names to exclude from transfer (defaults to EXCLUDED_KEYS)
        quiet: If True, suppress stdout printing of missing keys

    Returns:
        Tuple of (new_model, list_of_missing_or_excluded_keys)
    """
    if excluded_keys is None:
        excluded_keys = EXCLUDED_KEYS

    pretrain_dict = load_checkpoint_dict(pretrained)
    new_dict = new_model.state_dict()

    state_dict_to_load = {}
    missing_keys: List[str] = []

    for k, v in pretrain_dict.items():
        # Strip optional "module." prefix from DataParallel saves
        clean_k = k[7:] if k.startswith("module.") else k
        if clean_k in new_dict and clean_k not in excluded_keys:
            state_dict_to_load[clean_k] = v
        else:
            missing_keys.append(clean_k)
            if not quiet:
                print(f"Missing key(s) in state_dict :{clean_k}")

    # Load compatible parameters into new_model
    new_dict.update(state_dict_to_load)
    new_model.load_state_dict(new_dict)

    # Freeze transferred parameters
    for name, param in new_model.named_parameters():
        if name in state_dict_to_load:
            param.requires_grad = False

    return new_model, missing_keys
