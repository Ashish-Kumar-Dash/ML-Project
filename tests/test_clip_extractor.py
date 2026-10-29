"""Unit and Integration Tests for CLIP Visual Extractor Pipeline.

Covers:
- test_clip_model_loader: Model trunk instantiation, parameter freezing, feature dims (512, 768).
- test_spatial_preprocessing: Bicubic resize, center crop, normalization across arbitrary resolutions.
- test_chunked_forward_pass: Chunked batched forward inference on CUDA/CPU with unit norm check.
- test_temporal_ladder_resampler: 1D adaptive average pooling across ladder rungs (50, 100, 200) and front padding.
- test_ffmpeg_video_slicer: In-memory frame extraction from video intervals via FFmpeg pipe.
- test_e2e_pickle_merge_and_dataloader: Full synthetic pipeline to MOSIData and PyTorch DataLoader iteration.
"""

from pathlib import Path
import pickle
import sys
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

# Ensure ML-Project root and third_party/MPLMM are on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

mplmm_dir = repo_root / "third_party" / "MPLMM"
if str(mplmm_dir) not in sys.path:
    sys.path.insert(0, str(mplmm_dir))

from lenmod.extractors import CLIPVisualExtractor, CLIP_MODELS
from tools.extract_clip_features import create_synthetic_video, extract_frames_ffmpeg
from tools.merge_multimodal_features import merge_datasets


@pytest.fixture(scope="module")
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def test_clip_model_loader(device):
    """Verify loading, visual isolation, parameter freezing, and output dimensions."""
    # Test ViT-B/32 (primary SOP specification: d_v = 512)
    ext_b32 = CLIPVisualExtractor(model_name="ViT-B/32", device=device)
    assert ext_b32.feature_dim == 512
    assert hasattr(ext_b32, "visual")
    for param in ext_b32.visual.parameters():
        assert not param.requires_grad

    # Test ViT-L/14 (extended specification: d_v = 768)
    ext_l14 = CLIPVisualExtractor(model_name="ViT-L/14", device=device)
    assert ext_l14.feature_dim == 768
    for param in ext_l14.visual.parameters():
        assert not param.requires_grad


def test_spatial_preprocessing(device):
    """Verify preprocessing transforms arbitrary frame sizes to (N, 3, 224, 224)."""
    extractor = CLIPVisualExtractor(model_name="ViT-B/32", device=device)

    # 1. Test from uint8 numpy array with non-square aspect ratio
    raw_np = np.random.randint(0, 256, (3, 480, 640, 3), dtype=np.uint8)
    preproc = extractor.preprocess_frames(raw_np)
    assert preproc.shape == (3, 3, 224, 224)
    assert not torch.isnan(preproc).any()

    # 2. Test from float torch tensor
    raw_tensor = torch.rand((2, 300, 300, 3), dtype=torch.float32)
    preproc_tensor = extractor.preprocess_frames(raw_tensor)
    assert preproc_tensor.shape == (2, 3, 224, 224)


def test_chunked_forward_pass(device):
    """Verify chunked batched inference maintains memory bound, zero NaNs, and unit L2 norms."""
    extractor = CLIPVisualExtractor(model_name="ViT-B/32", device=device, normalize_embeddings=True)

    # Simulate 35 frames with chunk_size 8 (triggers multiple internal chunks)
    dummy_frames = np.random.randint(0, 256, (35, 240, 320, 3), dtype=np.uint8)
    embeds = extractor.extract_frames(dummy_frames, chunk_size=8)

    assert embeds.shape == (35, 512)
    assert embeds.dtype == torch.float32
    assert not torch.isnan(embeds).any()
    assert not torch.isinf(embeds).any()

    # Verify L2 normalization: norms should be identically 1.0 (within float tolerance)
    norms = torch.norm(embeds, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_temporal_ladder_resampler():
    """Verify 1D adaptive pooling across sequence lengths and front-padding behavior."""
    # 1. Identity case: T = V
    x_50 = np.random.randn(50, 512).astype(np.float32)
    res_identity = CLIPVisualExtractor.resample_temporal(x_50, target_len=50)
    assert np.allclose(res_identity, x_50, atol=1e-5)

    # 2. Downsampling: T=120 -> V=50
    x_120 = np.random.randn(120, 512).astype(np.float32)
    res_down = CLIPVisualExtractor.resample_temporal(x_120, target_len=50)
    assert res_down.shape == (50, 512)

    # 3. Upsampling / Stretching: T=25 -> V=100
    x_25 = np.random.randn(25, 512).astype(np.float32)
    res_up = CLIPVisualExtractor.resample_temporal(x_25, target_len=100)
    assert res_up.shape == (100, 512)

    # 4. Front-padding mode (for unaligned V=500)
    res_pad = CLIPVisualExtractor.resample_temporal(x_25, target_len=100, pad_front=True)
    assert res_pad.shape == (100, 512)
    assert np.all(res_pad[:75] == 0.0)
    assert np.allclose(res_pad[75:], x_25)


def test_ffmpeg_video_slicer(tmp_path):
    """Verify slicing and decoding sub-intervals from video using FFmpeg pipe."""
    video_file = tmp_path / "synthetic_clip.mp4"
    # Create 3-second 30fps test video
    create_synthetic_video(video_file, duration_sec=3.0, fps=30, width=320, height=240)
    assert video_file.is_file()

    # Slice interval [1.0, 2.0] -> expected ~30 frames
    frames = extract_frames_ffmpeg(video_file, start_sec=1.0, end_sec=2.0, fps=30, width=224, height=224)
    assert isinstance(frames, np.ndarray)
    assert frames.ndim == 4
    assert frames.shape[1:] == (224, 224, 3)
    assert frames.dtype == np.uint8
    assert 28 <= frames.shape[0] <= 32


def test_e2e_pickle_merge_and_dataloader(tmp_path, device):
    """Verify end-to-end integration: extract features, merge pickle, load via MOSIData."""
    ref_pkl = repo_root / "data" / "fake_mosi.pkl"
    if not ref_pkl.exists():
        pytest.skip(f"Reference fake_mosi not found at {ref_pkl}")

    with open(ref_pkl, "rb") as f:
        ref_data = pickle.load(f)

    # Generate visual embeddings matching fake_mosi splits
    vis_payload = {}
    for split in ["train", "valid", "test"]:
        n = len(ref_data[split]["labels"])
        vis_payload[split] = np.random.randn(n, 50, 512).astype(np.float32)

    vis_pkl_path = tmp_path / "vis_features.pkl"
    with open(vis_pkl_path, "wb") as f:
        pickle.dump(vis_payload, f)

    merged_out_path = tmp_path / "merged_dataset.pkl"
    merge_datasets(
        reference_pkl=ref_pkl,
        vision_pkl=vis_pkl_path,
        output_path=merged_out_path,
        verify_dataloader=False,
    )
    assert merged_out_path.is_file()

    # Verify loading via upstream MOSIData
    from src.mosidata import MOSIData

    torch.set_default_tensor_type("torch.FloatTensor")
    dataset = MOSIData(str(merged_out_path), split_type="train", drop_rate=0.5, full_data=False)
    assert len(dataset) == len(ref_data["train"]["labels"])

    text_len, audio_len, vision_len = dataset.get_seq_len()
    text_dim, audio_dim, vision_dim = dataset.get_dim()

    assert vision_len == 50
    assert vision_dim == 512
    assert text_dim == 300
    assert audio_dim == 5

    # Test DataLoader batch iteration
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    (text_b, audio_b, vision_b), labels_b, modes_b = batch

    assert vision_b.shape == (8, 50, 512)
    assert not torch.isnan(vision_b).any()

