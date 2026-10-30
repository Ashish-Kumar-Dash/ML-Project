#!/usr/bin/env python3
"""CLI Orchestrator for CLIP Visual Feature Extraction on CMU-MOSI.

Extracts high-dimensional foundation visual features from raw video files or frame
directories for the CMU-MOSI dataset, using frozen CLIP Vision Transformers
(ViT-B/32 or ViT-L/14) and 1D adaptive temporal pooling.

Adheres to:
- SOP Section IV.D, Item 2: CLIP visual extractor plus registration.
- SOP Table I: Rung L2 Visual Dimension d_v = 512 (ViT-B/32) or d_v = 768 (ViT-L/14).

Usage:
    # Synthetic verification mode (validates end-to-end pipeline without external media):
    python tools/extract_clip_features.py --synthetic_check --limit 10

    # Production run on raw video directory:
    python tools/extract_clip_features.py \
        --video_dir /path/to/mosi_videos \
        --reference_pkl data/mosi_data.pkl \
        --output_path data/mosi_clip_features_v50.pkl \
        --model ViT-B/32 \
        --seq_len 50
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Ensure ML-Project root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from lenmod.extractors import CLIPVisualExtractor, CLIP_MODELS


VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".webm", ".flv", ".mov"]


def resolve_video_path(
    video_dir: Path, segment_id: str, video_id: str
) -> Optional[Tuple[Path, bool]]:
    """Locate video file corresponding to a segment or video ID.

    Returns:
        (path, is_pre_segmented) where is_pre_segmented is True if the file
        corresponds to segment_id directly, False if it corresponds to video_id.
    """
    # 1. Check if individual segment file exists: {video_dir}/{segment_id}.ext
    for ext in VIDEO_EXTENSIONS:
        p = video_dir / f"{segment_id}{ext}"
        if p.is_file():
            return p, True

    # 2. Check in subfolder: {video_dir}/{video_id}/{segment_id}.ext
    for ext in VIDEO_EXTENSIONS:
        p = video_dir / video_id / f"{segment_id}{ext}"
        if p.is_file():
            return p, True

    # 2b. MMSA layout: {video_dir}/{video_id}/{clip_index}.ext with segment_id == f"{video_id}_{clip_index}"
    if "_" in segment_id:
        clip_index = segment_id.rsplit("_", 1)[1]
        for ext in VIDEO_EXTENSIONS:
            p = video_dir / video_id / f"{clip_index}{ext}"
            if p.is_file():
                return p, True

    # 3. Check if full video file exists: {video_dir}/{video_id}.ext
    for ext in VIDEO_EXTENSIONS:
        p = video_dir / f"{video_id}{ext}"
        if p.is_file():
            return p, False

    return None


def create_synthetic_video(
    video_path: Path, duration_sec: float = 5.0, fps: int = 30, width: int = 320, height: int = 240
) -> None:
    """Generate synthetic video with motion patterns using FFmpeg testsrc."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_sec:.2f}:size={width}x{height}:rate={fps}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-loglevel",
        "error",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)


def extract_frames_ffmpeg(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    fps: int = 30,
    width: Optional[int] = 224,
    height: Optional[int] = 224,
) -> np.ndarray:
    """Extract and resize video frames within [start_sec, end_sec] via FFmpeg pipe.

    Returns:
        (T, height, width, 3) uint8 numpy array of RGB frames.
    """
    duration = max(0.04, end_sec - start_sec)
    if not (width and height):
        # Native resolution: no resize at decode; the extractor applies CLIP's own preprocessing.
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        width, height = (int(v) for v in probe.stdout.strip().split("x")[:2])
    cmd = [
        "ffmpeg",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-vf",
        f"fps={fps},scale={width}:{height}" if (width and height) else f"fps={fps}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-loglevel",
        "error",
        "-",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_bytes, stderr = proc.communicate()

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg decoding failed for {video_path}: {err_msg}")

    frame_bytes = width * height * 3
    n_frames = len(raw_bytes) // frame_bytes

    if n_frames == 0:
        # Never substitute a black frame: a silent fallback would turn a missing clip into plausible features.
        raise RuntimeError(
            f"FFmpeg decoded 0 frames from {video_path} in [{start_sec:.3f}, {end_sec:.3f}] s"
        )

    frames = np.frombuffer(raw_bytes[: n_frames * frame_bytes], dtype=np.uint8).copy()
    return frames.reshape((n_frames, height, width, 3))


def process_segment(
    segment_info: Tuple[str, float, float],
    video_dir: Optional[Path],
    extractor: CLIPVisualExtractor,
    seq_len: int,
    pad_front: bool,
    fps: int,
    batch_size: int,
    synthetic_mode: bool,
    temp_synthetic_dir: Optional[Path] = None,
    return_frames: bool = False,
):
    """Extract and temporally pool CLIP features for a single segment."""
    segment_id, t_start, t_end = segment_info
    video_id = segment_id.rsplit("_", 1)[0] if "_" in segment_id else segment_id

    frames: Optional[np.ndarray] = None

    if synthetic_mode:
        # In synthetic mode, generate frames directly or use a temporary synthetic clip
        assert temp_synthetic_dir is not None
        synth_file = temp_synthetic_dir / f"{video_id}.mp4"
        if not synth_file.exists():
            create_synthetic_video(synth_file, duration_sec=max(10.0, t_end + 2.0), fps=fps)
        frames = extract_frames_ffmpeg(synth_file, t_start, t_end, fps=fps)
    else:
        assert video_dir is not None
        resolved = resolve_video_path(video_dir, segment_id, video_id)
        if resolved is None:
            raise FileNotFoundError(
                f"Could not find video file for segment '{segment_id}' (video '{video_id}') in {video_dir}"
            )
        video_path, is_pre_segmented = resolved
        if is_pre_segmented:
            # Pre-segmented clip: extract from start 0 to clip duration
            # Pre-segmented clip: decode the whole file at native resolution (the clip already spans [t_start, t_end]).
            frames = extract_frames_ffmpeg(video_path, 0.0, 1.0e4, fps=fps, width=None, height=None)
        else:
            # Full raw video: slice from t_start to t_end
            frames = extract_frames_ffmpeg(video_path, t_start, t_end, fps=fps, width=None, height=None)

    # Extract frame embeddings via frozen CLIP backbone
    frame_embeds = extractor.extract_frames(frames, chunk_size=batch_size)

    # Resample / pool along temporal axis
    pooled = extractor.resample_temporal(
        frame_embeds, target_len=seq_len, pad_front=pad_front
    )
    if return_frames:
        return pooled, frame_embeds
    return pooled


def extract_dataset(
    reference_pkl: Path,
    output_path: Path,
    video_dir: Optional[Path],
    model_name: str = "ViT-B/32",
    seq_len: int = 50,
    unaligned: bool = False,
    batch_size: int = 64,
    device: Optional[str] = None,
    fps: int = 30,
    limit: Optional[int] = None,
    synthetic_check: bool = False,
    native_out: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Execute CLIP feature extraction over all splits in reference dataset."""
    print(f"\n========================================================")
    print(f"Starting CLIP Feature Extraction: {model_name}")
    print(f"Reference dataset: {reference_pkl}")
    print(f"Output path:       {output_path}")
    print(f"Target seq_len:    {seq_len} (unaligned/pad_front={unaligned})")
    print(f"Mode:              {'SYNTHETIC TEST' if synthetic_check else 'REAL VIDEO'}")
    print(f"========================================================\n")

    with open(reference_pkl, "rb") as f:
        ref_data = pickle.load(f, encoding="latin1")

    extractor = CLIPVisualExtractor(
        model_name=model_name,
        device=device,
        normalize_embeddings=True,
    )
    print(f"Loaded {model_name} (dim={extractor.feature_dim}) on {extractor.target_device}")

    temp_synthetic_dir: Optional[Path] = None
    if synthetic_check:
        temp_synthetic_dir = Path("/tmp/mosi_synthetic_videos")
        temp_synthetic_dir.mkdir(parents=True, exist_ok=True)

    extracted_dict: Dict[str, np.ndarray] = {}
    ids_dict: Dict[str, List[str]] = {}
    nframes_dict: Dict[str, List[int]] = {}
    native_frames: Dict[str, list] = {s: [] for s in ("train", "valid", "test")}
    total_start_time = time.time()

    splits = ["train", "valid", "test"]
    for split in splits:
        if split not in ref_data:
            continue

        raw_ids = ref_data[split]["id"]
        n_total = len(raw_ids)
        n_process = min(n_total, limit) if limit is not None else n_total

        print(f"\n--- Processing split '{split}' ({n_process} / {n_total} samples) ---")
        split_features = np.zeros((n_process, seq_len, extractor.feature_dim), dtype=np.float32)
        split_ids: List[str] = []
        split_nframes: List[int] = []

        split_start = time.time()
        for idx in range(n_process):
            raw_id, raw_s, raw_e = raw_ids[idx]
            seg_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            t_start = float(raw_s)
            t_end = float(raw_e)

            feats = process_segment(
                segment_info=(seg_id, t_start, t_end),
                video_dir=video_dir,
                extractor=extractor,
                seq_len=seq_len,
                pad_front=unaligned,
                fps=fps,
                batch_size=batch_size,
                synthetic_mode=synthetic_check,
                temp_synthetic_dir=temp_synthetic_dir,
                return_frames=True,
            )
            feats, frame_embeds = feats

            split_features[idx] = feats
            split_ids.append(seg_id)
            split_nframes.append(int(frame_embeds.shape[0]))
            if native_out is not None:
                native_frames[split].append(frame_embeds.numpy().astype(np.float16))

            if (idx + 1) % 50 == 0 or (idx + 1) == n_process:
                elapsed = time.time() - split_start
                rate = (idx + 1) / elapsed
                print(f"[{split}] {idx + 1}/{n_process} clips ({rate:.1f} clips/s, elapsed: {elapsed:.1f}s)")

        # Verify no NaN or Inf
        assert not np.isnan(split_features).any(), f"NaN values detected in extracted features for {split}!"
        assert not np.isinf(split_features).any(), f"Inf values detected in extracted features for {split}!"

        extracted_dict[split] = split_features
        ids_dict[split] = split_ids
        nframes_dict[split] = split_nframes
        print(f"Completed split '{split}': shape {split_features.shape}, dtype {split_features.dtype}")

    total_elapsed = time.time() - total_start_time
    print(f"\nExtraction complete in {total_elapsed:.2f}s!")

    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": model_name,
        "feature_dim": extractor.feature_dim,
        "seq_len": seq_len,
        "unaligned": unaligned,
        "fps": fps,
        "synthetic": synthetic_check,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata["video_dir"] = str(video_dir) if video_dir else None
    metadata["preprocessing"] = "ffmpeg native-resolution decode; CLIP official bicubic shorter-side resize to 224 + centre crop; L2-normalised"
    metadata["n_frames"] = {s: {"min": int(min(v)), "median": float(np.median(v)), "max": int(max(v))} for s, v in nframes_dict.items() if v}
    payload = {**extracted_dict, "ids": ids_dict, "n_frames": nframes_dict, "meta": metadata}

    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Serialized extracted visual features to: {output_path} ({output_path.stat().st_size / 1e6:.2f} MB)")
    if native_out is not None:
        native_out = Path(native_out)
        native_out.parent.mkdir(parents=True, exist_ok=True)
        with open(native_out, "wb") as f:
            pickle.dump({"frames": native_frames, "ids": ids_dict, "meta": metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Serialized native per-frame features to: {native_out} ({native_out.stat().st_size / 1e6:.2f} MB)")
    return extracted_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CLIP foundation visual features for CMU-MOSI video segments."
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=None,
        help="Directory containing raw MOSI video files (.mp4 / .avi).",
    )
    parser.add_argument(
        "--reference_pkl",
        type=Path,
        default=repo_root / "data" / "mosi_data.pkl",
        help="Reference dataset pickle containing split segment IDs and timestamps.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=repo_root / "data" / "mosi_clip_features_v50.pkl",
        help="Path where extracted feature pickle will be serialized.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        choices=list(CLIP_MODELS.keys()),
        help="CLIP model trunk (ViT-B/32 for 512-dim, ViT-L/14 for 768-dim).",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=50,
        help="Target sequence length after temporal pooling (e.g. 50, 100, 200, 500).",
    )
    parser.add_argument(
        "--unaligned",
        action="store_true",
        help="If set, pad front with zeros instead of stretching when T < seq_len (for unaligned V=500).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Chunk size for batched visual transformer inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda:0 or cpu).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Framerate for FFmpeg video frame extraction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of segments per split (for testing/validation).",
    )
    parser.add_argument(
        "--native_out",
        type=Path,
        default=None,
        help="Optional path for native-rate per-frame embeddings (fp16), for re-pooling to any ladder length.",
    )
    parser.add_argument(
        "--synthetic_check",
        action="store_true",
        help="Generate synthetic video streams on the fly to test pipeline without raw videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.synthetic_check and args.video_dir is None:
        print(
            "ERROR: Either --video_dir must be provided pointing to video files, "
            "or --synthetic_check must be specified for headless validation."
        )
        sys.exit(1)

    extract_dataset(
        reference_pkl=args.reference_pkl,
        output_path=args.output_path,
        video_dir=args.video_dir,
        model_name=args.model,
        seq_len=args.seq_len,
        unaligned=args.unaligned,
        batch_size=args.batch_size,
        device=args.device,
        fps=args.fps,
        limit=args.limit,
        synthetic_check=args.synthetic_check,
        native_out=args.native_out,
    )


if __name__ == "__main__":
    main()
