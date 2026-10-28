r"""Operator ablations for the MPLMM generation module (end-term arms C and D).

Arm C (``frozen_dense``): the upstream dense MLPLayer time-mixers are kept exactly
as instantiated and frozen at random initialisation, so not one of their parameters
ever receives a gradient. If accuracy is unchanged, the dense parameters do no work.

Arm D (``pool_fixed``): the nine time-mixing layers are replaced by a fixed adaptive
average pool over the time axis followed by the same GELU that MLPLayer applies.
Zero parameters. This is the floor: no learned time mixing at all.

Both arms reuse TIME_MIXING_LAYERS from lenmod.operator, so the layer set is
identical to the invariant arm's. Every check asserts rather than warns.
"""

import hashlib
from typing import Optional

import torch
from torch import nn

from lenmod.operator import TIME_MIXING_LAYERS


class FixedAvgPoolTimeMixer(nn.Module):
    r"""Zero-parameter drop-in for the upstream MLPLayer time-mixer (arm D).

    Input  (batch, in_len, embed_dim)
    Output (batch, out_len, embed_dim)

    MLPLayer is Conv1d(in_len -> out_len, kernel_size=1) applied after a transpose,
    i.e. a learned dense map over the TIME axis followed by GELU. This class
    replaces the learned map with a fixed adaptive average pool over the same axis
    and keeps the GELU, so the only thing removed is the learned time mixing.
    """

    def __init__(self, out_len: int, embed_dim: int = 30) -> None:
        super().__init__()
        self.out_len = int(out_len)
        self.embed_dim = int(embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(self.out_len)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_len, d) -> (B, d, in_len) -> pool -> (B, out_len, d)
        pooled = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.act(pooled)


def swap_fixed_pool(model: nn.Module, embed_dim: Optional[int] = None) -> nn.Module:
    """Replace the nine time-mixing layers with zero-parameter average pooling."""
    if embed_dim is None:
        embed_dim = getattr(model, "prompt_dim", 30)

    modality_lens = {
        "l": getattr(model, "llen", 50),
        "a": getattr(model, "alen", 50),
        "v": getattr(model, "vlen", 50),
    }

    installed = 0
    for layer_name, mod in TIME_MIXING_LAYERS.items():
        assert hasattr(model, layer_name), f"time-mixing layer absent: {layer_name}"
        mixer = FixedAvgPoolTimeMixer(
            out_len=modality_lens[mod], embed_dim=embed_dim
        )
        n_params = sum(p.numel() for p in mixer.parameters())
        assert n_params == 0, (
            f"{layer_name}: arm D mixer must have zero parameters, has {n_params}"
        )
        setattr(model, layer_name, mixer)
        installed += 1

    assert installed == 9, f"expected 9 time-mixing layers, installed {installed}"
    return model


def freeze_time_mixers(model: nn.Module) -> int:
    """Freeze the nine DENSE time-mixing layers in place (arm C).

    Returns the number of parameters frozen. Asserts that each layer really is the
    upstream MLPLayer (it carries ``.conv``), so arm C cannot silently freeze an
    invariant or pooling operator instead.
    """
    frozen_params = 0
    layers = 0
    for layer_name in TIME_MIXING_LAYERS:
        assert hasattr(model, layer_name), f"time-mixing layer absent: {layer_name}"
        layer = getattr(model, layer_name)
        assert hasattr(layer, "conv"), (
            f"{layer_name} is {type(layer).__name__}, not the upstream MLPLayer; "
            "arm C must freeze the DENSE operator"
        )
        n_params = sum(p.numel() for p in layer.parameters())
        assert n_params > 0, f"{layer_name} has no parameters to freeze"
        for p in layer.parameters():
            p.requires_grad = False
        frozen_params += n_params
        layers += 1

    assert layers == 9, f"expected 9 time-mixing layers, froze {layers}"
    return frozen_params


def time_mixer_fingerprint(model: nn.Module) -> str:
    """sha256 over the nine time-mixing layers' parameter bytes, in a fixed order."""
    hasher = hashlib.sha256()
    for layer_name in sorted(TIME_MIXING_LAYERS):
        layer = getattr(model, layer_name)
        for param_name, param in sorted(layer.named_parameters()):
            hasher.update(f"{layer_name}.{param_name}".encode())
            hasher.update(param.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


__all__ = [
    "FixedAvgPoolTimeMixer",
    "swap_fixed_pool",
    "freeze_time_mixers",
    "time_mixer_fingerprint",
]
