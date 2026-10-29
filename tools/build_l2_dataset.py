#!/usr/bin/env python3
"""Build L2 Synthetic Orthogonal Nuisance Stressor Dataset for CMU-MOSI.

Constructs the synthetic L2 rung of the feature-abstraction ladder (SOP Section IV, Table I):
- Audio: randomly up-projected COVAREP features (5 -> 768) + orthogonal nuisance noise
- Vision: randomly up-projected Facet features (20 -> 512) + orthogonal nuisance noise
- Text: GloVe-300 embeddings (d_l = 300, sequence length L = 50) held strictly identical to L1
- Labels & IDs: Strictly identical to canonical L1 mosi_data.pkl

NOTE: This dataset represents a controlled high-dimensional orthogonal nuisance stressor.
For genuine pretrained Vision Transformer feature extraction (CLIP ViT-B/32 or ViT-L/14),
use `tools/extract_clip_features.py` and `tools/merge_multimodal_features.py`.

Usage:
    python tools/build_l2_dataset.py --input data/mosi_data.pkl --output data/mosi_data_l2.pkl
"""

import argparse
from pathlib import Path
import pickle
import numpy as np


def create_orthonormal_basis(d_in: int, d_out: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate an orthonormal projection matrix U in R^{d_in x d_out} (U @ U^T = I)."""
    # Generate random Gaussian matrix and perform QR decomposition
    raw = rng.randn(d_out, d_in)
    q, _ = np.linalg.qr(raw)
    U = q[:, :d_in].T  # (d_in, d_out)
    return U.astype(np.float32)


def create_orthogonal_complement_projector(U: np.ndarray) -> np.ndarray:
    """Construct projection matrix P_perp = I - U^T @ (U @ U^T)^-1 @ U into orthogonal complement."""
    # Since U has orthonormal rows (U @ U^T = I), P_perp = I - U^T @ U
    d_out = U.shape[1]
    I = np.eye(d_out, dtype=np.float32)
    P_perp = I - U.T @ U
    return P_perp.astype(np.float32)


def generate_smooth_temporal_noise(
    n_samples: int,
    seq_len: int,
    dim: int,
    P_perp: np.ndarray,
    rng: np.random.RandomState,
    noise_std: float = 1.0,
) -> np.ndarray:
    """Generate temporally smooth nuisance noise projected into the orthogonal complement of U."""
    raw_noise = rng.randn(n_samples, seq_len, dim).astype(np.float32) * noise_std
    # Apply 1D temporal Gaussian/moving-average smoothing to reflect sequential token dynamics
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    smoothed = np.zeros_like(raw_noise)
    for t in range(seq_len):
        t_prev = max(0, t - 1)
        t_next = min(seq_len - 1, t + 1)
        smoothed[:, t, :] = (
            raw_noise[:, t_prev, :] * kernel[0]
            + raw_noise[:, t, :] * kernel[1]
            + raw_noise[:, t_next, :] * kernel[2]
        )
    # Project into orthogonal complement of the affect subspace
    flat = smoothed.reshape(-1, dim)
    proj_noise = flat @ P_perp
    return proj_noise.reshape(n_samples, seq_len, dim)


def build_l2_dataset(
    input_path: str,
    output_path: str,
    dim_a: int = 768,
    dim_v: int = 512,
    seed: int = 42,
    noise_std: float = 0.8,
) -> None:
    """Construct and serialize the L2 dataset pickle."""
    in_file = Path(input_path).resolve()
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading reference L1 dataset from: {in_file}")
    with open(in_file, "rb") as f:
        l1_data = pickle.load(f, encoding="latin1")

    rng = np.random.RandomState(seed)

    # 1. Construct Orthonormal Subspace Bases
    # Audio: 5 -> 768
    U_a = create_orthonormal_basis(d_in=5, d_out=dim_a, rng=rng)
    P_perp_a = create_orthogonal_complement_projector(U_a)

    # Visual: 20 -> 512
    U_v = create_orthonormal_basis(d_in=20, d_out=dim_v, rng=rng)
    P_perp_v = create_orthogonal_complement_projector(U_v)

    l2_data = {}
    splits = ["train", "valid", "test"]

    for split in splits:
        print(f"\nProcessing split: {split}...")
        split_dict = l1_data[split]

        l1_audio = np.asarray(split_dict["audio"], dtype=np.float32)   # (N, 50, 5)
        l1_vision = np.asarray(split_dict["vision"], dtype=np.float32) # (N, 50, 20)
        l1_text = np.asarray(split_dict["text"], dtype=np.float32)     # (N, 50, 300)
        l1_labels = np.asarray(split_dict["labels"], dtype=np.float32) # (N, 1, 1)
        l1_id = split_dict.get("id")

        n_samples, seq_len, _ = l1_audio.shape

        # Expand Audio: (N, 50, 5) @ (5, 768) -> (N, 50, 768) + Z_a
        affect_audio = l1_audio @ U_a
        noise_audio = generate_smooth_temporal_noise(
            n_samples, seq_len, dim_a, P_perp_a, rng, noise_std=noise_std
        )
        l2_audio = (affect_audio + noise_audio).astype(np.float32)

        # Expand Vision: (N, 50, 20) @ (20, 512) -> (N, 50, 512) + Z_v
        affect_vision = l1_vision @ U_v
        noise_vision = generate_smooth_temporal_noise(
            n_samples, seq_len, dim_v, P_perp_v, rng, noise_std=noise_std
        )
        l2_vision = (affect_vision + noise_vision).astype(np.float32)

        # Assert zero NaNs or infs
        assert not np.isnan(l2_audio).any(), f"NaN found in {split} audio"
        assert not np.isnan(l2_vision).any(), f"NaN found in {split} vision"
        assert not np.isinf(l2_audio).any(), f"Inf found in {split} audio"
        assert not np.isinf(l2_vision).any(), f"Inf found in {split} vision"

        # Verify shapes
        assert l2_audio.shape == (n_samples, 50, dim_a), f"Bad audio shape: {l2_audio.shape}"
        assert l2_vision.shape == (n_samples, 50, dim_v), f"Bad vision shape: {l2_vision.shape}"
        assert l1_text.shape == (n_samples, 50, 300), f"Bad text shape: {l1_text.shape}"
        assert l1_labels.shape == (n_samples, 1, 1), f"Bad labels shape: {l1_labels.shape}"

        print(f"  Audio shape:  {l2_audio.shape} (synthetic up-projected 5->768 + nuisance noise)")
        print(f"  Vision shape: {l2_vision.shape} (synthetic up-projected 20->512 + nuisance noise)")
        print(f"  Text shape:   {l1_text.shape} (GloVe-300 preserved)")
        print(f"  Labels shape: {l1_labels.shape} ([-3, 3] preserved)")

        l2_data[split] = {
            "audio": l2_audio,
            "vision": l2_vision,
            "text": l1_text,
            "labels": l1_labels,
        }
        if l1_id is not None:
            l2_data[split]["id"] = l1_id

    print(f"\nWriting L2 dataset pickle to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(l2_data, f, protocol=4)

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"L2 dataset creation complete: {out_file.name} ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build L2 Foundation Feature Dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/mosi_data.pkl",
        help="Path to canonical L1 aligned MOSI pickle (default: data/mosi_data.pkl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/mosi_data_l2.pkl",
        help="Path to target L2 pickle (default: data/mosi_data_l2.pkl)",
    )
    parser.add_argument("--dim_a", type=int, default=768, help="Target audio dimension (default: 768)")
    parser.add_argument("--dim_v", type=int, default=512, help="Target vision dimension (default: 512)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--noise_std", type=float, default=0.8, help="Nuisance variance standard deviation")

    args = parser.parse_args()
    build_l2_dataset(
        input_path=args.input,
        output_path=args.output,
        dim_a=args.dim_a,
        dim_v=args.dim_v,
        seed=args.seed,
        noise_std=args.noise_std,
    )


if __name__ == "__main__":
    main()
