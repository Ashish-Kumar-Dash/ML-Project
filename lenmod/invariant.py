r"""Length-invariant missing-modality generation module and runtime layer swapper.

As identified in the project SOP:
The original MPLMM time-mixing layer (MLPLayer) is Conv1d(k=1) applied after a
transpose, making it a dense linear map whose parameter count scales quadratically
with input sequence lengths:
    params = \ell_{out} * (\ell_p + \sum \ell_{in}) + \ell_{out}
At CMU-MOSI unaligned lengths (50/375/500), the 9 layers explode to 972,175 parameters.

This module provides length-invariant replacements whose parameter counts do NOT
depend on the input sequence lengths (\ell_{in}):
1. CrossAttentionTimeMixer (Primary candidate): Cross-attention over the concatenated
   input sequence against a fixed learned query set of size \ell_{out}.
2. PoolingTimeMixer (Ablation candidate): Adaptive pooling along the time axis
   followed by linear feature projection.
"""

from typing import Dict, Optional
import torch
from torch import nn


class CrossAttentionTimeMixer(nn.Module):
    r"""Length-invariant generator using cross-attention against learned queries.

    Input arrives transposed as (batch, \ell_{in}, prompt_dim).
    Learned queries have shape (1, \ell_{out}, prompt_dim).
    Cross-attention attends over all input positions, decoupling parameters
    from \ell_{in} while allowing every input token to contribute.
    Returns tensor of shape (batch, \ell_{out}, prompt_dim).
    """

    def __init__(
        self,
        out_len: int,
        embed_dim: int = 30,
        num_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_len = out_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Learned target queries of length \ell_{out}
        self.query = nn.Parameter(torch.randn(1, out_len, embed_dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, in_seq_len, embed_dim)
        batch_sz = x.size(0)
        q = self.query.expand(batch_sz, -1, -1)
        out, _ = self.mha(query=q, key=x, value=x)
        return self.act(out)


class PoolingTimeMixer(nn.Module):
    """Ablation baseline: Adaptive pooling to target length followed by projection."""

    def __init__(self, out_len: int, embed_dim: int = 30):
        super().__init__()
        self.out_len = out_len
        self.pool = nn.AdaptiveAvgPool1d(out_len)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, in_seq_len, embed_dim)
        # Pool across time axis (dim 1)
        x_perm = x.permute(0, 2, 1)  # (batch, embed_dim, in_seq_len)
        pooled = self.pool(x_perm).permute(0, 2, 1)  # (batch, out_len, embed_dim)
        return self.act(self.proj(pooled))


TIME_MIXING_LAYERS = {
    "l_avp": "l",
    "a_lvp": "a",
    "v_alp": "v",
    "l_ap": "l",
    "l_vp": "l",
    "a_lp": "a",
    "a_vp": "a",
    "v_ap": "v",
    "v_lp": "v",
}


def swap_time_mixers(
    model: nn.Module,
    operator: str = "cross_attn",
    num_heads: int = 2,
    embed_dim: Optional[int] = None,
) -> nn.Module:
    """Dynamically replace the 9 time-mixing layers on a PromptModel instance

    with length-invariant operators at runtime.

    Args:
        model: PromptModel instance
        operator: 'cross_attn' (primary) or 'pooling' (ablation)
        num_heads: Number of attention heads for cross_attn
        embed_dim: Embedding dimension (defaults to model.prompt_dim)

    Returns:
        The mutated PromptModel instance with length-invariant generators.
    """
    if embed_dim is None:
        embed_dim = getattr(model, "prompt_dim", 30)

    modality_lens = {
        "l": getattr(model, "llen", 50),
        "a": getattr(model, "alen", 50),
        "v": getattr(model, "vlen", 50),
    }

    for layer_name, mod in TIME_MIXING_LAYERS.items():
        out_len = modality_lens[mod]
        if operator == "cross_attn":
            mixer = CrossAttentionTimeMixer(
                out_len=out_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        elif operator == "pooling":
            mixer = PoolingTimeMixer(
                out_len=out_len,
                embed_dim=embed_dim,
            )
        else:
            raise ValueError(f"Unknown generator operator: {operator}")

        setattr(model, layer_name, mixer)

    return model


__all__ = [
    "CrossAttentionTimeMixer",
    "PoolingTimeMixer",
    "swap_time_mixers",
    "TIME_MIXING_LAYERS",
]
