#!/usr/bin/env python3
"""Fit Dimension-Matched Projection for L2' Control and Compute Multi-Rung Linear CKA.

Implements the L2' control of the feature-abstraction ladder (SOP Section IV, Table I):
- Loads L2 foundation features: wav2vec2-base (768), CLIP ViT-B/32 (512), GloVe-300 (300).
- Fits label-aware projection W_a (768 -> 5) and W_v (512 -> 20) via PLSRegression
  strictly on the TRAINING SPLIT ONLY to guarantee zero data leakage.
- Transforms train, valid, and test splits to produce data/mosi_data_l2_prime.pkl.
- Evaluates pairwise Linear CKA across L1, L2, and L2' to verify the correlation hypothesis.
- Saves results to results/cka_l1_l2_l2prime.json.

Usage:
    python tools/fit_l2_projection.py \
        --l1_path data/mosi_data.pkl \
        --l2_path data/mosi_data_l2.pkl \
        --l2_prime_path data/mosi_data_l2_prime.pkl \
        --out_json results/cka_l1_l2_l2prime.json
"""

import argparse
import json
from pathlib import Path
import pickle
from typing import Any, Dict
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
import torch

import sys
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Ensure lenmod can be imported
import lenmod.compat
from lenmod.hooks import linear_cka


def compute_split_cka(audio: np.ndarray, vision: np.ndarray, text: np.ndarray) -> Dict[str, float]:
    """Compute pairwise Linear CKA for temporal-mean representations."""
    # Compute mean over time dimension (N, D)
    a_mean = torch.from_numpy(audio.mean(axis=1)).float()
    v_mean = torch.from_numpy(vision.mean(axis=1)).float()
    t_mean = torch.from_numpy(text.mean(axis=1)).float()

    return {
        "cka_audio_vision": round(linear_cka(a_mean, v_mean), 5),
        "cka_audio_text": round(linear_cka(a_mean, t_mean), 5),
        "cka_vision_text": round(linear_cka(v_mean, t_mean), 5),
    }


def fit_l2_prime_projection(
    l1_path: str,
    l2_path: str,
    l2_prime_path: str,
    out_json: str,
    target_dim_a: int = 5,
    target_dim_v: int = 20,
    method: str = "pls",
) -> Dict[str, Any]:
    """Fit projection on L2 train split, construct L2' dataset, and calculate multi-rung CKA."""
    l1_file = Path(l1_path).resolve()
    l2_file = Path(l2_path).resolve()
    l2p_file = Path(l2_prime_path).resolve()
    json_file = Path(out_json).resolve()

    l2p_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("L2' DIMENSION-MATCHED PROJECTION & CKA BENCHMARK")
    print(f"  L1 Reference:  {l1_file}")
    print(f"  L2 Foundation: {l2_file}")
    print(f"  L2' Target:    {l2p_file}")
    print(f"  Method:        {method.upper()} (Target: audio={target_dim_a}, vision={target_dim_v})")
    print("=" * 70)

    with open(l1_file, "rb") as f:
        l1_data = pickle.load(f, encoding="latin1")

    with open(l2_file, "rb") as f:
        l2_data = pickle.load(f, encoding="latin1")

    # 1. Fit Projection STRICTLY on Train Split Only
    print("\n[1/4] Fitting projection strictly on 'train' split...")
    train_audio = np.asarray(l2_data["train"]["audio"], dtype=np.float32)   # (1284, 50, 768)
    train_vision = np.asarray(l2_data["train"]["vision"], dtype=np.float32) # (1284, 50, 512)
    train_labels = np.asarray(l2_data["train"]["labels"], dtype=np.float32).squeeze() # (1284,)

    N_train, T_a, D_a = train_audio.shape
    _, T_v, D_v = train_vision.shape

    train_audio_mean = train_audio.mean(axis=1)   # (1284, 768)
    train_vision_mean = train_vision.mean(axis=1) # (1284, 512)

    if method.lower() == "pls":
        # Label-aware Partial Least Squares Regression
        proj_model_a = PLSRegression(n_components=target_dim_a)
        proj_model_a.fit(train_audio_mean, train_labels)

        proj_model_v = PLSRegression(n_components=target_dim_v)
        proj_model_v.fit(train_vision_mean, train_labels)
    elif method.lower() == "pca":
        # Unsupervised PCA
        proj_model_a = PCA(n_components=target_dim_a)
        proj_model_a.fit(train_audio_mean)

        proj_model_v = PCA(n_components=target_dim_v)
        proj_model_v.fit(train_vision_mean)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    print(f"  Audio projector fitted: {D_a} -> {target_dim_a}")
    print(f"  Vision projector fitted: {D_v} -> {target_dim_v}")

    # 2. Transform all splits using the frozen train-fitted projection
    print("\n[2/4] Transforming 'train', 'valid', and 'test' splits...")
    l2_prime_data = {}
    splits = ["train", "valid", "test"]

    for split in splits:
        split_audio = np.asarray(l2_data[split]["audio"], dtype=np.float32)   # (N, 50, 768)
        split_vision = np.asarray(l2_data[split]["vision"], dtype=np.float32) # (N, 50, 512)
        split_text = np.asarray(l2_data[split]["text"], dtype=np.float32)     # (N, 50, 300)
        split_labels = np.asarray(l2_data[split]["labels"], dtype=np.float32) # (N, 1, 1)

        N_s, _, _ = split_audio.shape

        # Transform time steps consistently
        flat_a = split_audio.reshape(N_s * T_a, D_a)
        flat_v = split_vision.reshape(N_s * T_v, D_v)

        proj_flat_a = proj_model_a.transform(flat_a)
        proj_flat_v = proj_model_v.transform(flat_v)

        proj_audio = proj_flat_a.reshape(N_s, T_a, target_dim_a).astype(np.float32)
        proj_vision = proj_flat_v.reshape(N_s, T_v, target_dim_v).astype(np.float32)

        # Assert no NaNs/infs
        assert not np.isnan(proj_audio).any(), f"NaN in projected audio ({split})"
        assert not np.isnan(proj_vision).any(), f"NaN in projected vision ({split})"

        l2_prime_data[split] = {
            "audio": proj_audio,
            "vision": proj_vision,
            "text": split_text,
            "labels": split_labels,
        }
        if "id" in l2_data[split]:
            l2_prime_data[split]["id"] = l2_data[split]["id"]

        print(f"  {split.capitalize()}: audio={proj_audio.shape}, vision={proj_vision.shape}, text={split_text.shape}")

    # 3. Serialize L2' pickle
    print(f"\n[3/4] Saving L2' pickle to: {l2p_file}")
    with open(l2p_file, "wb") as f:
        pickle.dump(l2_prime_data, f, protocol=4)
    print(f"  File saved successfully ({l2p_file.stat().st_size / (1024*1024):.2f} MB)")

    # 4. Compute Linear CKA Across All Three Rungs (L1, L2, L2')
    print("\n[4/4] Computing Pairwise Linear CKA for L1, L2, and L2'...")
    cka_summary = {
        "metadata": {
            "method": method,
            "target_dim_audio": target_dim_a,
            "target_dim_vision": target_dim_v,
            "l1_file": str(l1_file.name),
            "l2_file": str(l2_file.name),
            "l2_prime_file": str(l2p_file.name),
        },
        "rungs": {
            "L1_baseline": {},
            "L2_foundation": {},
            "L2_prime_projected": {},
        },
    }

    for split in splits:
        cka_summary["rungs"]["L1_baseline"][split] = compute_split_cka(
            np.asarray(l1_data[split]["audio"]),
            np.asarray(l1_data[split]["vision"]),
            np.asarray(l1_data[split]["text"]),
        )
        cka_summary["rungs"]["L2_foundation"][split] = compute_split_cka(
            np.asarray(l2_data[split]["audio"]),
            np.asarray(l2_data[split]["vision"]),
            np.asarray(l2_data[split]["text"]),
        )
        cka_summary["rungs"]["L2_prime_projected"][split] = compute_split_cka(
            np.asarray(l2_prime_data[split]["audio"]),
            np.asarray(l2_prime_data[split]["vision"]),
            np.asarray(l2_prime_data[split]["text"]),
        )

    # Save to JSON
    with open(json_file, "w") as f:
        json.dump(cka_summary, f, indent=2)
    print(f"CKA report written to: {json_file}")

    # Print Formatted Comparison Table
    print("\n" + "=" * 70)
    print(f"{'RUNG / SPLIT':<24} | {'CKA(Audio, Vision)':<20} | {'CKA(Audio, Text)':<18} | {'CKA(Vision, Text)':<18}")
    print("-" * 70)
    for rung_name, rung_label in [
        ("L1_baseline", "L1 (COVAREP/FACET)"),
        ("L2_foundation", "L2 (wav2vec2/CLIP)"),
        ("L2_prime_projected", f"L2' ({method.upper()} Proj)"),
    ]:
        for split in ["train", "test"]:
            res = cka_summary["rungs"][rung_name][split]
            label = f"{rung_label} [{split}]"
            print(
                f"{label:<24} | "
                f"{res['cka_audio_vision']:<20.4f} | "
                f"{res['cka_audio_text']:<18.4f} | "
                f"{res['cka_vision_text']:<18.4f}"
            )
        print("-" * 70)

    return cka_summary


def main():
    parser = argparse.ArgumentParser(description="Fit L2' projection and compute Linear CKA.")
    parser.add_argument("--l1_path", type=str, default="data/mosi_data.pkl", help="L1 pickle path")
    parser.add_argument("--l2_path", type=str, default="data/mosi_data_l2.pkl", help="L2 pickle path")
    parser.add_argument(
        "--l2_prime_path",
        type=str,
        default="data/mosi_data_l2_prime.pkl",
        help="L2' output pickle path",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="results/cka_l1_l2_l2prime.json",
        help="CKA output JSON path",
    )
    parser.add_argument("--dim_a", type=int, default=5, help="Projected audio dim")
    parser.add_argument("--dim_v", type=int, default=20, help="Projected vision dim")
    parser.add_argument("--method", type=str, default="pls", choices=["pls", "pca"], help="Projection method")

    args = parser.parse_args()
    fit_l2_prime_projection(
        l1_path=args.l1_path,
        l2_path=args.l2_path,
        l2_prime_path=args.l2_prime_path,
        out_json=args.out_json,
        target_dim_a=args.dim_a,
        target_dim_v=args.dim_v,
        method=args.method,
    )


if __name__ == "__main__":
    main()
