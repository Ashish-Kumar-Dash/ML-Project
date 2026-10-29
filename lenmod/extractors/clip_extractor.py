"""CLIP Visual Extractor for Multimodal Sequences.

Extracts high-dimensional foundation visual features from video frames using frozen
OpenAI CLIP Vision Transformer backbones (ViT-B/32 or ViT-L/14).

Adheres to:
- SOP Section IV.D, Item 2: CLIP visual extractor plus registration.
- SOP Table I: Rung L2 Visual Dimension d_v = 512 (ViT-B/32) or d_v = 768 (ViT-L/14).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import urllib.request
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Standard CLIP normalization parameters (OpenAI CLIP)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Registered model metadata and official OpenAI weight URLs
CLIP_MODELS: Dict[str, Dict[str, Union[str, int]]] = {
    "ViT-B/32": {
        "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "sha256": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
        "filename": "ViT-B-32.pt",
        "dim": 512,
        "input_resolution": 224,
    },
    "ViT-L/14": {
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "sha256": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
        "filename": "ViT-L-14.pt",
        "dim": 768,
        "input_resolution": 224,
    },
}


def get_default_cache_dir() -> Path:
    """Return default CLIP model cache directory (~/.cache/clip)."""
    cache_dir = Path(os.environ.get("CLIP_CACHE_DIR", Path.home() / ".cache" / "clip")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def verify_file_sha256(path: Union[str, Path], expected_sha: str) -> bool:
    """Verify SHA256 checksum of a local file."""
    p = Path(path)
    if not p.is_file():
        return False
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower() == expected_sha.lower()


def download_clip_weights(model_name: str, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Download pretrained TorchScript weights for specified CLIP model if not already cached."""
    if model_name not in CLIP_MODELS:
        raise ValueError(
            f"Unsupported CLIP model '{model_name}'. Available models: {list(CLIP_MODELS.keys())}"
        )

    info = CLIP_MODELS[model_name]
    target_dir = Path(cache_dir).expanduser() if cache_dir else get_default_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / str(info["filename"])
    expected_sha = str(info["sha256"])

    if target_path.exists():
        if verify_file_sha256(target_path, expected_sha):
            return target_path
        warnings.warn(
            f"Cached weights at {target_path} failed SHA256 verification. Re-downloading...",
            UserWarning,
        )
        target_path.unlink(missing_ok=True)

    url = str(info["url"])
    print(f"Downloading CLIP {model_name} from {url} to {target_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "lenmod-clip-extractor/1.0"})
    with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_file:
        while chunk := resp.read(1024 * 1024):
            out_file.write(chunk)

    if not verify_file_sha256(target_path, expected_sha):
        target_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded weights for {model_name} failed SHA256 verification!")

    print(f"Successfully cached {model_name} at {target_path}.")
    return target_path


def get_clip_model_path(model_name: str, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Retrieve path to cached CLIP model weights, downloading if necessary."""
    return download_clip_weights(model_name, cache_dir)


class CLIPVisualExtractor(nn.Module):
    """Frozen CLIP Vision Transformer feature extractor.

    Takes a batch or sequence of RGB frames and maps them to high-dimensional
    normalized embeddings using the visual trunk of OpenAI CLIP.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        normalize_embeddings: bool = True,
    ):
        super().__init__()
        if model_name not in CLIP_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. Choose from: {list(CLIP_MODELS.keys())}"
            )

        self.model_name = model_name
        self.feature_dim = int(CLIP_MODELS[model_name]["dim"])
        self.input_resolution = int(CLIP_MODELS[model_name]["input_resolution"])
        self.normalize_embeddings = normalize_embeddings

        if device is None:
            self.target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.target_device = torch.device(device)

        # Download or locate weights
        weights_path = get_clip_model_path(model_name, cache_dir)

        # Load TorchScript model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            full_model = torch.jit.load(str(weights_path), map_location=self.target_device)

        if not hasattr(full_model, "visual"):
            raise AttributeError(f"Model at {weights_path} does not expose a '.visual' submodule.")

        self.visual = full_model.visual
        self.visual.eval()

        # Freeze all parameters
        for p in self.visual.parameters():
            p.requires_grad = False

        # Register normalization buffers
        self.register_buffer(
            "clip_mean",
            torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def preprocess_frames(
        self, frames: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Preprocess RGB frames to standard CLIP normalized input format.

        Args:
            frames: (T, H, W, 3) or (B, T, H, W, 3) in uint8 [0, 255] or float [0, 1].

        Returns:
            Preprocessed tensor (N, 3, 224, 224) on target device, dtype=torch.half (or float on CPU).
        """
        if isinstance(frames, np.ndarray):
            tensor_frames = torch.from_numpy(frames if frames.flags.writeable else frames.copy())
        else:
            tensor_frames = frames

        # Flatten any leading dimensions so tensor is (N, H, W, C)
        orig_shape = tensor_frames.shape
        if len(orig_shape) == 3:  # (H, W, C) single frame
            tensor_frames = tensor_frames.unsqueeze(0)
        elif len(orig_shape) > 4:  # e.g. (B, T, H, W, C)
            tensor_frames = tensor_frames.reshape(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])

        # Permute (N, H, W, C) -> (N, C, H, W)
        if tensor_frames.shape[-1] == 3:
            tensor_frames = tensor_frames.permute(0, 3, 1, 2)

        # Scale to [0, 1] float
        if tensor_frames.dtype == torch.uint8:
            tensor_frames = tensor_frames.float() / 255.0
        else:
            tensor_frames = tensor_frames.float()

        # Bicubic resize to (224, 224) with antialiasing
        res = self.input_resolution
        resized = TF.resize(
            tensor_frames,
            [res, res],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        )
        cropped = TF.center_crop(resized, [res, res])

        # Normalization
        mean = self.clip_mean.to(cropped.device)
        std = self.clip_std.to(cropped.device)
        normalized = (cropped - mean) / std

        # TorchScript CLIP expects half precision on CUDA
        if self.target_device.type == "cuda":
            return normalized.to(device=self.target_device, dtype=torch.half)
        return normalized.to(device=self.target_device, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        """Forward pass through frozen visual trunk.

        Args:
            preprocessed_images: (N, 3, 224, 224) tensor on target device.

        Returns:
            (N, feature_dim) feature representations as float32.
        """
        features = self.visual(preprocessed_images)
        features = features.float()  # Convert FP16 output back to FP32

        if self.normalize_embeddings:
            features = F.normalize(features, p=2, dim=-1)

        return features

    @torch.no_grad()
    def extract_frames(
        self,
        frames: Union[np.ndarray, torch.Tensor],
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """Extract embeddings from raw RGB video frames in memory-bounded chunks.

        Args:
            frames: (T, H, W, 3) array/tensor of video frames.
            chunk_size: Number of frames per forward chunk to limit peak VRAM.

        Returns:
            (T, feature_dim) extracted feature tensor on CPU (float32).
        """
        if len(frames) == 0:
            return torch.empty((0, self.feature_dim), dtype=torch.float32)

        preprocessed = self.preprocess_frames(frames)
        n_frames = preprocessed.shape[0]

        outputs = []
        for start_idx in range(0, n_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, n_frames)
            chunk = preprocessed[start_idx:end_idx]
            chunk_embeds = self.forward(chunk)
            outputs.append(chunk_embeds.cpu())

        return torch.cat(outputs, dim=0)

    @staticmethod
    def resample_temporal(
        embeddings: Union[torch.Tensor, np.ndarray],
        target_len: int,
        pad_front: bool = False,
    ) -> np.ndarray:
        """Resample time-series representations along the temporal axis.

        Uses 1D adaptive average pooling to resample (T, D) -> (target_len, D).
        Matches the binning semantics in tools/pool_resample.py.

        Args:
            embeddings: (T, D) sequence embeddings.
            target_len: Desired sequence length V.
            pad_front: If True and T < target_len, zero-pads the front instead of stretching.

        Returns:
            (target_len, D) resampled numpy array in float32.
        """
        if isinstance(embeddings, np.ndarray):
            emb_tensor = torch.from_numpy(embeddings).float()
        else:
            emb_tensor = embeddings.float().cpu()

        t_cur, d_feat = emb_tensor.shape

        if t_cur == 0:
            return np.zeros((target_len, d_feat), dtype=np.float32)

        if pad_front and t_cur < target_len:
            pad_rows = target_len - t_cur
            padded = torch.cat([torch.zeros(pad_rows, d_feat), emb_tensor], dim=0)
            return padded.numpy()

        if t_cur == target_len:
            return emb_tensor.numpy()

        # Reshape (T, D) -> (1, D, T) for 1D pooling
        reshaped = emb_tensor.transpose(0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool1d(reshaped, output_size=target_len)
        # Reshape (1, D, target_len) -> (target_len, D)
        out = pooled.squeeze(0).transpose(0, 1)
        return out.numpy()

