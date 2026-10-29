"""Tests for L2 and L2' Diagnostic Datasets, Projections, and Model Mechanics.

Verifies:
1. Integrity, shapes, datatypes, and split counts for data/mosi_data_l2.pkl.
2. Integrity, shapes, datatypes, and split counts for data/mosi_data_l2_prime.pkl.
3. Linear CKA report properties in results/cka_l1_l2_l2prime.json.
4. Model parameter counts, backbone transfer (5 missing keys), and prompt assertions.
5. Forward pass execution across missing modes 0 to 6.
"""

import json
from pathlib import Path
import pickle
import pytest
import torch

import lenmod.compat
from lenmod.config import make_hp
from lenmod.operator import swap_time_mixers
from lenmod.transfer import transfer_weights
from src.model import PromptModel


REPO_ROOT = Path(__file__).resolve().parent.parent
L2_PATH = REPO_ROOT / "data" / "mosi_data_l2.pkl"
L2P_PATH = REPO_ROOT / "data" / "mosi_data_l2_prime.pkl"
CKA_JSON_PATH = REPO_ROOT / "results" / "cka_l1_l2_l2prime.json"
PRETRAINED_PATH = REPO_ROOT / "pretrained" / "mosei.pt"

CANONICAL_SPLITS = {
    "train": 1284,
    "valid": 229,
    "test": 686,
}


def test_l2_dataset_integrity():
    """Verify L2 foundation feature dataset pickle."""
    assert L2_PATH.exists(), f"L2 dataset not found at {L2_PATH}"
    with open(L2_PATH, "rb") as f:
        data = pickle.load(f)

    for split, count in CANONICAL_SPLITS.items():
        assert split in data, f"Split {split} missing from L2 dataset"
        split_data = data[split]

        assert split_data["audio"].shape == (count, 50, 768), f"Bad L2 audio shape for {split}"
        assert split_data["vision"].shape == (count, 50, 512), f"Bad L2 vision shape for {split}"
        assert split_data["text"].shape == (count, 50, 300), f"Bad L2 text shape for {split}"
        assert split_data["labels"].shape == (count, 1, 1), f"Bad L2 labels shape for {split}"

        assert not torch.isnan(torch.tensor(split_data["audio"])).any()
        assert not torch.isnan(torch.tensor(split_data["vision"])).any()
        assert not torch.isinf(torch.tensor(split_data["audio"])).any()
        assert not torch.isinf(torch.tensor(split_data["vision"])).any()


def test_l2_prime_dataset_integrity():
    """Verify L2' dimension-matched control dataset pickle."""
    assert L2P_PATH.exists(), f"L2' dataset not found at {L2P_PATH}"
    with open(L2P_PATH, "rb") as f:
        data = pickle.load(f)

    for split, count in CANONICAL_SPLITS.items():
        assert split in data, f"Split {split} missing from L2' dataset"
        split_data = data[split]

        # Dimension-matched to L1: audio=5, vision=20, text=300
        assert split_data["audio"].shape == (count, 50, 5), f"Bad L2' audio shape for {split}"
        assert split_data["vision"].shape == (count, 50, 20), f"Bad L2' vision shape for {split}"
        assert split_data["text"].shape == (count, 50, 300), f"Bad L2' text shape for {split}"
        assert split_data["labels"].shape == (count, 1, 1), f"Bad L2' labels shape for {split}"

        assert not torch.isnan(torch.tensor(split_data["audio"])).any()
        assert not torch.isnan(torch.tensor(split_data["vision"])).any()


def test_cka_results_json():
    """Verify Linear CKA benchmark artifact properties."""
    assert CKA_JSON_PATH.exists(), f"CKA JSON missing at {CKA_JSON_PATH}"
    with open(CKA_JSON_PATH, "r") as f:
        report = json.load(f)

    assert "rungs" in report
    for rung in ["L1_baseline", "L2_foundation", "L2_prime_projected"]:
        assert rung in report["rungs"]
        for split in ["train", "valid", "test"]:
            res = report["rungs"][rung][split]
            assert 0.0 <= res["cka_audio_vision"] <= 1.0
            assert 0.0 <= res["cka_audio_text"] <= 1.0
            assert 0.0 <= res["cka_vision_text"] <= 1.0


def test_l2_prime_model_mechanics():
    """Verify PromptModel parameters and weight transfer for L2' (5/20/300)."""
    hp = make_hp(L=50, A=50, V=50, orig_dims=(300, 5, 20), dataset="mosi", use_cuda=False)
    model_dense = PromptModel(hp)
    model_inv = PromptModel(hp)
    swap_time_mixers(model_inv, operator="cross_attn")

    _, missing_dense = transfer_weights(model_dense, str(PRETRAINED_PATH), quiet=True)
    _, missing_inv = transfer_weights(model_inv, str(PRETRAINED_PATH), quiet=True)

    expected_missing = {"proj_l.weight", "proj_a.weight", "proj_v.weight", "out_layer.weight", "out_layer.bias"}
    assert set(missing_dense) == expected_missing
    assert set(missing_inv) == expected_missing

    params_dense = sum(p.numel() for p in model_dense.parameters() if p.requires_grad)
    params_inv = sum(p.numel() for p in model_inv.parameters() if p.requires_grad)

    # Identical to L1
    assert params_dense == 88141, f"Expected 88,141 params for L2' dense, got {params_dense}"
    assert params_inv == 97471, f"Expected 97,471 params for L2' invariant, got {params_inv}"

    # Prompt check
    assert (model_dense.missing_type_prompt == 0).all()
    assert (model_inv.missing_type_prompt == 0).all()


def test_l2_model_mechanics():
    """Verify PromptModel parameters and weight transfer for L2 (768/512/300)."""
    hp = make_hp(L=50, A=50, V=50, orig_dims=(300, 768, 512), dataset="mosi", use_cuda=False)
    model_dense = PromptModel(hp)
    model_inv = PromptModel(hp)
    swap_time_mixers(model_inv, operator="cross_attn")

    _, missing_dense = transfer_weights(model_dense, str(PRETRAINED_PATH), quiet=True)
    _, missing_inv = transfer_weights(model_inv, str(PRETRAINED_PATH), quiet=True)

    expected_missing = {"proj_l.weight", "proj_a.weight", "proj_v.weight", "out_layer.weight", "out_layer.bias"}
    assert set(missing_dense) == expected_missing
    assert set(missing_inv) == expected_missing

    params_dense = sum(p.numel() for p in model_dense.parameters() if p.requires_grad)
    params_inv = sum(p.numel() for p in model_inv.parameters() if p.requires_grad)

    # Dimensionality scaling on channel mixers: +112,950 params
    assert params_dense == 201091, f"Expected 201,091 params for L2 dense, got {params_dense}"
    assert params_inv == 210421, f"Expected 210,421 params for L2 invariant, got {params_inv}"

    # Prompt check
    assert (model_dense.missing_type_prompt == 0).all()
    assert (model_inv.missing_type_prompt == 0).all()


def test_l2_forward_smoke():
    """Verify forward pass execution on L2 across all missing codes 0 to 6."""
    hp = make_hp(L=50, A=50, V=50, orig_dims=(300, 768, 512), dataset="mosi", use_cuda=False)
    model = PromptModel(hp)
    swap_time_mixers(model, operator="cross_attn")
    transfer_weights(model, str(PRETRAINED_PATH), quiet=True)
    model.eval()

    B = 2
    x_l = torch.randn(B, 50, 300)
    x_a = torch.randn(B, 50, 768)
    x_v = torch.randn(B, 50, 512)

    with torch.no_grad():
        for mode in range(7):
            m_tensor = torch.full((B,), mode, dtype=torch.long)
            out = model(x_l, x_a, x_v, m_tensor)
            assert out.shape == (B, 1)
