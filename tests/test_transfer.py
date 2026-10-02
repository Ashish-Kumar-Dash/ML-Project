"""Unit tests for lenmod.transfer module."""

from pathlib import Path
import unittest
import torch

import lenmod.compat
from lenmod.config import make_hp
from lenmod.transfer import transfer_weights, load_checkpoint_dict
from src.model import PromptModel

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_PATH = REPO_ROOT / "pretrained" / "mosei.pt"


class TestTransferWeights(unittest.TestCase):
    @unittest.skipUnless(PRETRAINED_PATH.exists(), "pretrained/mosei.pt required")
    def test_transfer_from_file_and_parameter_freeze(self):
        hp = make_hp(L=50, A=50, V=50, orig_dims=(300, 5, 20))
        model = PromptModel(hp)

        model, missing_keys = transfer_weights(model, PRETRAINED_PATH, quiet=True)

        # Must report exactly the 5 dimension-dependent keys
        expected_missing = [
            "proj_l.weight",
            "proj_a.weight",
            "proj_v.weight",
            "out_layer.weight",
            "out_layer.bias",
        ]
        self.assertEqual(len(missing_keys), 5)
        self.assertEqual(sorted(missing_keys), sorted(expected_missing))

        # Trainable parameters on aligned MOSI must be exactly 88,141
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(trainable, 88141)

    @unittest.skipUnless(PRETRAINED_PATH.exists(), "pretrained/mosei.pt required")
    def test_transfer_from_state_dict(self):
        # Test loading from state dict directly (our trainer format)
        raw_dict = load_checkpoint_dict(PRETRAINED_PATH)
        self.assertIsInstance(raw_dict, dict)

        hp = make_hp(L=50, A=50, V=50, orig_dims=(300, 5, 20))
        model = PromptModel(hp)

        # Wrap in {"state_dict": ...}
        wrapped = {"epoch": 40, "state_dict": raw_dict}
        model, missing_keys = transfer_weights(model, wrapped, quiet=True)

        self.assertEqual(len(missing_keys), 5)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(trainable, 88141)


if __name__ == "__main__":
    unittest.main()
