import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

import lenmod.compat
from lenmod.config import make_hp
from lenmod.hooks import (
    GenerationHookManager,
    cosine_similarity_fidelity,
    evaluate_generation_fidelity,
    linear_cka,
)
from lenmod.operator import swap_time_mixers
from src.model import PromptModel


class MockDataset:
    """Mock dataset mimicking MOSIData split format."""

    def __init__(self, n_samples: int = 4, L: int = 50, A: int = 50, V: int = 50):
        self.text = torch.randn(n_samples, L, 300)
        self.audio = torch.randn(n_samples, A, 5)
        self.vision = torch.randn(n_samples, V, 20)
        self.labels = torch.randn(n_samples, 1, 1)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {
            "text": self.text[idx],
            "audio": self.audio[idx],
            "vision": self.vision[idx],
            "labels": self.labels[idx],
        }


class TestGenerationHooks(unittest.TestCase):
    def test_linear_cka_properties(self):
        # 1. Identical representations must yield CKA = 1.0
        X = torch.randn(20, 30)
        self.assertAlmostEqual(linear_cka(X, X), 1.0, places=5)

        # 2. Invariant to isotropic scaling
        self.assertAlmostEqual(linear_cka(X, 10.0 * X), 1.0, places=5)

        # 3. Invariant to orthogonal rotation/projection Q
        Q, _ = torch.linalg.qr(torch.randn(30, 30))
        self.assertAlmostEqual(linear_cka(X, torch.matmul(X, Q)), 1.0, places=5)

        # 4. Independent representations with large N yield low CKA (< 0.1)
        torch.manual_seed(42)
        X_rand = torch.randn(500, 30)
        Y_rand = torch.randn(500, 30)
        self.assertLess(linear_cka(X_rand, Y_rand), 0.1)

    def test_cosine_similarity_fidelity(self):
        X = torch.randn(10, 30, 50)
        res_ident = cosine_similarity_fidelity(X, X, dim=1)
        self.assertAlmostEqual(res_ident["mean_cosine_sim"], 1.0, places=5)
        self.assertAlmostEqual(res_ident["flattened_cosine_sim"], 1.0, places=5)

        res_neg = cosine_similarity_fidelity(X, -X, dim=1)
        self.assertAlmostEqual(res_neg["mean_cosine_sim"], -1.0, places=5)
        self.assertAlmostEqual(res_neg["flattened_cosine_sim"], -1.0, places=5)

    def test_generation_hook_manager_lifecycle(self):
        hp = make_hp(50, 50, 50)
        model = PromptModel(hp)
        model.eval()

        # Before context manager: no active hooks
        with GenerationHookManager(model) as hook_mgr:
            self.assertGreater(len(hook_mgr.hook_handles), 0)

            # Run forward pass for case 0 (language missing, l_avp active)
            x_l = torch.randn(2, 50, 300)
            x_a = torch.randn(2, 50, 5)
            x_v = torch.randn(2, 50, 20)
            missing = [0, 0]

            with torch.no_grad():
                model(x_l, x_a, x_v, missing)

            out_lavp = hook_mgr.get_layer_output("l_avp")
            self.assertIsNotNone(out_lavp)
            self.assertEqual(out_lavp.size(0), 2)
            self.assertEqual(out_lavp.size(1), 50)  # llen
            self.assertEqual(out_lavp.size(2), 30)  # prompt_dim

        # After context manager: all hooks removed
        self.assertEqual(len(hook_mgr.hook_handles), 0)

    def test_evaluate_generation_fidelity_baseline_and_invariant(self):
        device = torch.device("cpu")
        mock_loader = DataLoader(MockDataset(n_samples=4), batch_size=2)

        for variant in ["baseline", "invariant"]:
            hp = make_hp(50, 50, 50)
            model = PromptModel(hp)
            if variant == "invariant":
                model = swap_time_mixers(model, operator="cross_attn", query_len=50)

            metrics = evaluate_generation_fidelity(model, mock_loader, device)

            # Check that all 6 missing modality cases (0 to 5) are evaluated
            for code in range(6):
                self.assertIn(code, metrics["cases"])
                case_data = metrics["cases"][code]
                for mod, mod_data in case_data["modalities"].items():
                    self.assertIn("cosine_similarity", mod_data)
                    self.assertIn("linear_cka", mod_data)
                    self.assertFalse(torch.isnan(torch.tensor(mod_data["cosine_similarity"])))
                    self.assertFalse(torch.isnan(torch.tensor(mod_data["linear_cka"])))

            self.assertIn("mean_generation_fidelity", metrics)
            self.assertIn("cosine_similarity", metrics["mean_generation_fidelity"])
            self.assertIn("linear_cka", metrics["mean_generation_fidelity"])


if __name__ == "__main__":
    unittest.main()
