#!/usr/bin/env python3
"""Multimodal Dataset Pickle Packaging Utility.

Merges newly extracted foundation visual features (e.g. CLIP ViT-B/32 or ViT-L/14)
with canonical text (GloVe-300), acoustic representations, sentiment labels, and IDs
from a reference dataset pickle (e.g. CMU-MOSI mosi_data.pkl).

Adheres to:
- SOP Section IV.D, Item 5: Pickle merge utility.
- Strict data schema parity with canonical MulT / MPLMM format:
  {
      "train": {"vision": (1284, V, d_v), "audio": ..., "text": ..., "labels": ..., "id": ...},
      "valid": {"vision": (229, V, d_v), ...},
      "test":  {"vision": (686, V, d_v), ...}
  }

Usage:
    python tools/merge_multimodal_features.py \
        --reference_pkl data/mosi_data.pkl \
        --vision_pkl data/mosi_clip_features_v50.pkl \
        --output_path data/mosi_data_clip_v50.pkl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
import pickle
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

# Ensure ML-Project root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


EXPECTED_SPLIT_COUNTS = {
    "train": 1284,
    "valid": 229,
    "test": 686,
}


def merge_datasets(
    reference_pkl: Path,
    vision_pkl: Path,
    output_path: Path,
    audio_pkl: Optional[Path] = None,
    verify_dataloader: bool = True,
    require_ids: bool = True,
    allow_synthetic: bool = False,
) -> Path:
    """Merge new visual features into reference multimodal dataset pickle."""
    print(f"\n========================================================")
    print(f"Merging Multimodal Features into Dataset Pickle")
    print(f"Reference dataset: {reference_pkl}")
    print(f"Visual features:   {vision_pkl}")
    if audio_pkl:
        print(f"Audio features:    {audio_pkl}")
    print(f"Target output:     {output_path}")
    print(f"========================================================\n")

    if not reference_pkl.is_file():
        raise FileNotFoundError(f"Reference dataset pickle not found: {reference_pkl}")
    if not vision_pkl.is_file():
        raise FileNotFoundError(f"Visual features pickle not found: {vision_pkl}")

    with open(reference_pkl, "rb") as f:
        ref_data = pickle.load(f, encoding="latin1")

    with open(vision_pkl, "rb") as f:
        vis_payload = pickle.load(f)

    # vis_payload may contain {"train": ..., "valid": ..., "test": ..., "meta": ...}
    vis_meta = vis_payload.get("meta", {}) if isinstance(vis_payload, dict) else {}
    if vis_meta.get("synthetic") and not allow_synthetic:
        raise ValueError(f"{vision_pkl} was produced in synthetic mode (meta.synthetic=True); refusing to merge it")

    audio_payload: Optional[Dict[str, Any]] = None
    if audio_pkl and audio_pkl.is_file():
        with open(audio_pkl, "rb") as f:
            audio_payload = pickle.load(f)

    merged_data: Dict[str, Dict[str, Any]] = {}
    splits = ["train", "valid", "test"]

    for split in splits:
        if split not in ref_data:
            raise KeyError(f"Required split '{split}' missing from reference dataset {reference_pkl}")
        if split not in vis_payload:
            raise KeyError(f"Required split '{split}' missing from visual features {vision_pkl}")

        ref_split = ref_data[split]
        vis_split = vis_payload[split]

        ref_n = len(ref_split["labels"])
        vis_n = len(vis_split)

        print(f"--- Processing Split '{split}' ---")
        print(f"  Reference samples: {ref_n}")
        print(f"  Visual samples:    {vis_n}")

        if vis_n != ref_n:
            raise ValueError(
                f"Sample count mismatch for split '{split}': reference has {ref_n}, "
                f"visual features has {vis_n}!"
            )

        vis_ids = (vis_payload.get("ids") or {}).get(split)
        if vis_ids is None:
            if require_ids:
                raise ValueError(f"{vision_pkl} carries no segment IDs for split '{split}'; sample order cannot be verified")
        else:
            ref_ids = [r[0].decode("utf-8") if isinstance(r[0], bytes) else str(r[0]) for r in ref_split["id"]]
            if list(vis_ids) != ref_ids:
                n_bad = sum(a != b for a, b in zip(vis_ids, ref_ids))
                raise ValueError(f"Segment ID order mismatch in split '{split}': {n_bad} positions differ")
            print(f"  Segment IDs:       {len(vis_ids)} match the reference order")

        # Standard split count check if standard MOSI
        if ref_n in EXPECTED_SPLIT_COUNTS.values():
            expected = EXPECTED_SPLIT_COUNTS.get(split)
            if expected is not None and ref_n != expected:
                print(f"  [Notice] Split '{split}' sample count {ref_n} != canonical {expected}")

        # Vision processing & validation
        vision_arr = np.asarray(vis_split, dtype=np.float32)
        assert not np.isnan(vision_arr).any(), f"NaNs detected in {split} visual features!"
        assert not np.isinf(vision_arr).any(), f"Infs detected in {split} visual features!"

        # Text processing (strictly preserve GloVe-300 from reference)
        text_arr = np.asarray(ref_split["text"], dtype=np.float32)
        assert not np.isnan(text_arr).any(), f"NaNs detected in {split} text features!"

        # Audio processing (custom audio payload or preserve reference)
        if audio_payload and split in audio_payload:
            audio_arr = np.asarray(audio_payload[split], dtype=np.float32)
        else:
            audio_arr = np.asarray(ref_split["audio"], dtype=np.float32)
            # Replace -inf with 0 if present (canonical MulT behavior)
            audio_arr[audio_arr == -np.inf] = 0.0

        assert not np.isnan(audio_arr).any(), f"NaNs detected in {split} audio features!"
        assert not np.isinf(audio_arr).any(), f"Infs detected in {split} audio features!"

        # Labels & IDs (strictly preserved)
        labels_arr = np.asarray(ref_split["labels"], dtype=np.float32)
        id_arr = ref_split.get("id")

        print(f"  Final Shapes -> Vision: {vision_arr.shape}, Audio: {audio_arr.shape}, "
              f"Text: {text_arr.shape}, Labels: {labels_arr.shape}")

        split_dict: Dict[str, Any] = {
            "vision": vision_arr,
            "audio": audio_arr,
            "text": text_arr,
            "labels": labels_arr,
        }
        if id_arr is not None:
            split_dict["id"] = id_arr

        merged_data[split] = split_dict

    # Write output pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(merged_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    sidecar = {"vision": vis_meta, "vision_pkl": str(vision_pkl), "reference_pkl": str(reference_pkl),
               "audio_source": str(audio_pkl) if audio_payload else "reference pickle audio (unchanged)",
               "text_source": "reference pickle text (unchanged)"}
    with open(str(output_path) + ".meta.json", "w") as f:
        json.dump(sidecar, f, indent=1, default=str)

    print(f"\nSuccessfully serialized merged pickle to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.2f} MB")

    if verify_dataloader:
        print("\n--- Verifying Compatibility with MOSIData & PyTorch DataLoader ---")
        # Ensure default tensor type remains FloatTensor on CPU
        torch.set_default_tensor_type("torch.FloatTensor")
        
        # Add third_party/MPLMM to sys.path to load MOSIData
        mplmm_path = repo_root / "third_party" / "MPLMM"
        if str(mplmm_path) not in sys.path:
            sys.path.insert(0, str(mplmm_path))

        try:
            from src.mosidata import MOSIData
            from torch.utils.data import DataLoader

            for s_name in ["train", "valid", "test"]:
                dataset = MOSIData(str(output_path), split_type=s_name, drop_rate=0.0, full_data=True)
                seq_lens = dataset.get_seq_len()
                dims = dataset.get_dim()
                lbl_info = dataset.get_lbl_info()
                print(f"  [{s_name}] MOSIData initialized: len={len(dataset)}, "
                      f"seq_lens (text, audio, vision)={seq_lens}, dims={dims}, lbl_info={lbl_info}")

            # Test a mini-batch DataLoader iteration
            train_set = MOSIData(str(output_path), split_type="train", drop_rate=0.5, full_data=False)
            loader = DataLoader(train_set, batch_size=16, shuffle=False)
            batch = next(iter(loader))
            (text_b, audio_b, vision_b), labels_b, missing_mode_b = batch
            print(f"  Mini-batch test: vision_batch={vision_b.shape}, audio_batch={audio_b.shape}, "
                  f"text_batch={text_b.shape}, labels={labels_b.shape}, missing_modes={missing_mode_b.shape}")
            print("  MOSIData compatibility verified successfully!\n")
        except Exception as e:
            print(f"  [Warning] Loader validation warning: {e}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge extracted visual foundation features into reference dataset pickle."
    )
    parser.add_argument(
        "--reference_pkl",
        type=Path,
        default=repo_root / "data" / "mosi_data.pkl",
        help="Canonical reference dataset pickle (with GloVe-300 text and labels).",
    )
    parser.add_argument(
        "--vision_pkl",
        type=Path,
        required=True,
        help="Extracted CLIP visual features pickle from tools/extract_clip_features.py.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Target output path for the merged dataset pickle.",
    )
    parser.add_argument(
        "--audio_pkl",
        type=Path,
        default=None,
        help="Optional acoustic features pickle (if replacing COVAREP with wav2vec2).",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip MOSIData loader verification step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_datasets(
        reference_pkl=args.reference_pkl,
        vision_pkl=args.vision_pkl,
        output_path=args.output_path,
        audio_pkl=args.audio_pkl,
        verify_dataloader=not args.no_verify,
    )


if __name__ == "__main__":
    main()

