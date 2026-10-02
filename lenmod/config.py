"""Configuration builder for MPLMM models and training runs."""

import argparse
from typing import Optional, Sequence, Tuple
import torch

# Standard MOSI feature dimensions: GloVe 300, COVAREP 5, Facet 20
DEFAULT_MOSI_DIMS = (300, 5, 20)


def make_hp(
    L: int,
    A: int,
    V: int,
    orig_dims: Optional[Sequence[int]] = None,
    dataset: str = "mosi",
    drop_rate: float = 0.7,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_epochs: int = 40,
    seed: int = 666,
    proj_dim: int = 30,
    layers: int = 5,
    num_heads: int = 5,
    prompt_dim: int = 30,
    prompt_length: int = 16,
    output_dim: int = 1,
    **kwargs,
) -> argparse.Namespace:
    """Create a complete hyp_params Namespace configured for sequence lengths (L, A, V).

    Args:
        L: Sequence length for language (text)
        A: Sequence length for audio
        V: Sequence length for vision
        orig_dims: Tuple of (d_l, d_a, d_v) raw/extracted feature dimensions.
                   Defaults to (300, 5, 20) for MOSI.
        dataset: Dataset identifier ('mosi', 'mosei', 'sims', 'iemocap')
        drop_rate: Modality drop rate during prompt tuning (paper optimum is 0.7)
        batch_size: Training batch size
        lr: Initial learning rate
        num_epochs: Maximum epochs
        seed: Random seed
        proj_dim: Projection dimension of MulT backbone
        layers: Number of transformer cross-attention layers
        num_heads: Number of attention heads
        prompt_dim: Embedding dimension of prompt tokens
        prompt_length: Sequence length of generative prompt tokens
        output_dim: Output dimension (1 for continuous sentiment)
        **kwargs: Any additional hyperparameters to attach to the Namespace.

    Returns:
        argparse.Namespace configured with all required attributes for
        MULTModel, PromptModel, and training routines.
    """
    if orig_dims is None:
        if dataset.lower() == "mosei":
            orig_dims = (300, 74, 35)
        else:
            orig_dims = DEFAULT_MOSI_DIMS

    orig_d_l, orig_d_a, orig_d_v = orig_dims
    seq_len = (int(L), int(A), int(V))

    hp = argparse.Namespace(
        # Sequence lengths and input dimensions
        seq_len=seq_len,
        orig_d_l=orig_d_l,
        orig_d_a=orig_d_a,
        orig_d_v=orig_d_v,
        orig_dims=orig_dims,
        # Backbone architecture
        proj_dim=proj_dim,
        layers=layers,
        nlevels=layers,
        num_heads=num_heads,
        attn_mask=True,
        attn_dropout=0.1,
        attn_dropout_a=0.1,
        attn_dropout_v=0.1,
        relu_dropout=0.1,
        res_dropout=0.1,
        out_dropout=0.1,
        embed_dropout=0.25,
        output_dim=output_dim,
        # Prompt tuning hyperparameters
        prompt_dim=prompt_dim,
        prompt_length=prompt_length,
        drop_rate=drop_rate,
        # Training / optimization
        dataset=dataset.lower(),
        batch_size=batch_size,
        lr=lr,
        clip=0.8,
        when=10,
        optim="Adam",
        num_epochs=num_epochs,
        log_interval=30,
        seed=seed,
        criterion="L1Loss" if output_dim == 1 else "CrossEntropyLoss",
        use_cuda=torch.cuda.is_available(),
        pretrained_model=None,
        name=None,
        data_path=None,
    )

    # Attach any user-provided overrides
    for k, v in kwargs.items():
        setattr(hp, k, v)

    return hp


__all__ = ["make_hp", "DEFAULT_MOSI_DIMS"]
