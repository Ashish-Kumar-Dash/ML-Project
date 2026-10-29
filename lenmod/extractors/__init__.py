"""Extractors subpackage for foundation feature representations."""

from lenmod.extractors.clip_extractor import (
    CLIP_MODELS,
    CLIPVisualExtractor,
    download_clip_weights,
    get_clip_model_path,
)

__all__ = [
    "CLIPVisualExtractor",
    "CLIP_MODELS",
    "get_clip_model_path",
    "download_clip_weights",
]

