"""lenmod: Length-invariant missing-modality modeling package."""

from lenmod import compat
from lenmod.config import make_hp
from lenmod.operator import (
    LengthInvariant,
    swap_time_mixers,
)
from lenmod.invariant import (
    CrossAttentionTimeMixer,
    PoolingTimeMixer,
)
from lenmod.hooks import (
    GenerationHookManager,
    cosine_similarity_fidelity,
    evaluate_generation_fidelity,
    linear_cka,
)
from lenmod.trainer import (
    MODALITY_CASE_NAMES,
    compute_metrics,
    eval_cases,
    evaluate,
    fit,
)

__all__ = [
    "compat",
    "make_hp",
    "fit",
    "evaluate",
    "eval_cases",
    "compute_metrics",
    "MODALITY_CASE_NAMES",
    "swap_time_mixers",
    "LengthInvariant",
    "CrossAttentionTimeMixer",
    "PoolingTimeMixer",
    "GenerationHookManager",
    "cosine_similarity_fidelity",
    "evaluate_generation_fidelity",
    "linear_cka",
]




