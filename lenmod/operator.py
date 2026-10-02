r"""Length-invariant missing-modality generation operator and layer swapper.

Addresses the core architectural limitation identified in the SOP:
The original MPLMM time-mixing layer (MLPLayer) is Conv1d(k=1) applied after a
transpose, making it a dense linear map over time whose parameter count scales
bilinearly with input sequence length:
    params = \ell_{out} * (\ell_p + \sum \ell_{in}) + \ell_{out}
At CMU-MOSI unaligned lengths (50/375/500), the 9 layers total 972,175 parameters.

LengthInvariant decouples parameter count from input length (\ell_{in}) using
cross-attention over all available input tokens against a fixed, learned query
set of size \ell_{out}.
"""

from typing import Dict, Optional, Tuple
import torch
from torch import nn


class LengthInvariant(nn.Module):
    r"""Length-invariant generator operator.

    Receives concatenated prompt and available modality features of shape
    (batch, in_seq_len, embed_dim) and generates target modality representations
    of shape (batch, out_len, embed_dim).

    Parameter count is strictly O(\ell_{out} \times d_p + 4 d_p^2), completely
    independent of the input sequence length \ell_{in}.
    """

    def __init__(
        self,
        out_len: int,
        embed_dim: int = 30,
        num_heads: int = 2,
        query_len: int = 50,
        dropout: float = 0.0,
        operator: str = "cross_attn",
    ):
        super().__init__()
        self.out_len = out_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_len = query_len
        self.operator = operator

        if operator == "cross_attn":
            # Canonical learned query tokens of fixed length [1, query_len, d_p]
            # Parameter count is strictly constant regardless of in_len or out_len
            self.query = nn.Parameter(torch.randn(1, query_len, embed_dim) * 0.02)
            self.mha = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.act = nn.GELU()
        elif operator == "pooling":
            # Ablation baseline: Adaptive pooling along temporal axis
            self.pool = nn.AdaptiveAvgPool1d(out_len)
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_seq_len, embed_dim).

        Returns:
            Output tensor of shape (batch, out_len, embed_dim).
        """
        if self.operator == "cross_attn":
            batch_sz = x.size(0)
            if self.out_len == self.query_len:
                q = self.query
            else:
                # Interpolate query along time axis to match out_len
                q = torch.nn.functional.interpolate(
                    self.query.transpose(1, 2),
                    size=self.out_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            q = q.expand(batch_sz, -1, -1)
            out, _ = self.mha(query=q, key=x, value=x)
            return self.act(out)
        elif self.operator == "pooling":
            # Transpose time to last dimension for 1D pooling: (B, embed_dim, in_seq_len)
            x_perm = x.permute(0, 2, 1)
            pooled = self.pool(x_perm).permute(0, 2, 1)  # (B, out_len, embed_dim)
            return self.act(self.proj(pooled))



# The nine length-dependent time-mixing layers in upstream PromptModel
TIME_MIXING_LAYERS: Dict[str, str] = {
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
    query_len: int = 50,
    embed_dim: Optional[int] = None,
) -> nn.Module:
    """Dynamically replace the nine time-mixing layers on a PromptModel instance

    with LengthInvariant operators at runtime.

    Args:
        model: Instantiated upstream PromptModel
        operator: Generator operator ('cross_attn' or 'pooling')
        num_heads: Number of attention heads for cross-attention
        query_len: Canonical query length for constant parameter count (default 50)
        embed_dim: Embedding dimension (defaults to model.prompt_dim)

    Returns:
        The mutated model instance with LengthInvariant operators installed.
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
        mixer = LengthInvariant(
            out_len=out_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            query_len=query_len,
            operator=operator,
        )
        setattr(model, layer_name, mixer)

    return model



__all__ = [
    "LengthInvariant",
    "swap_time_mixers",
    "TIME_MIXING_LAYERS",
]
