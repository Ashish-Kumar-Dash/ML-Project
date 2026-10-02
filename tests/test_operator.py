import unittest
import torch
import torch.nn as nn

import lenmod.compat
from lenmod.config import make_hp
from lenmod.operator import LengthInvariant, swap_time_mixers, TIME_MIXING_LAYERS
from src.model import MLPLayer, PromptModel


class TestLengthInvariantOperator(unittest.TestCase):
    def setUp(self):
        # The four canonical sequence length configurations to test
        self.four_lengths = [
            (50, 50, 50),
            (50, 100, 100),
            (50, 200, 200),
            (50, 375, 500),
        ]

    def test_constant_parameter_count_across_four_lengths(self):
        """Requirement 1: Verify constant parameter count across all four lengths:

        (50/50/50, 50/100/100, 50/200/200, 50/375/500).
        """
        param_totals = []
        layer_param_counts = {name: [] for name in TIME_MIXING_LAYERS}

        for L, A, V in self.four_lengths:
            model = PromptModel(make_hp(L, A, V))
            swap_time_mixers(model, operator="cross_attn", query_len=50)

            total_swapped_params = 0
            for name in TIME_MIXING_LAYERS:
                layer = getattr(model, name)
                n_p = sum(p.numel() for p in layer.parameters())
                layer_param_counts[name].append(n_p)
                total_swapped_params += n_p

            param_totals.append(total_swapped_params)

        # 1. Total parameters across all 9 layers must be strictly identical
        first_total = param_totals[0]
        for idx, (L, A, V) in enumerate(self.four_lengths):
            self.assertEqual(
                param_totals[idx],
                first_total,
                f"Parameter count mismatch at ({L}/{A}/{V}): {param_totals[idx]} != {first_total}",
            )

        # 2. Each individual layer must have identical parameter count across all four lengths
        for name in TIME_MIXING_LAYERS:
            counts = layer_param_counts[name]
            self.assertEqual(
                len(set(counts)),
                1,
                f"Layer {name} parameter counts vary across lengths: {counts}",
            )

        # Exactly 46,980 parameters across all 9 layers (9 * 5,220)
        self.assertEqual(first_total, 46980)

    def test_forward_pass_all_seven_codes_all_four_lengths(self):
        """Requirement 2: Verify forward pass runs cleanly for all seven codes (0 to 6)

        across all four sequence length configurations (4 x 7 = 28 checks).
        """
        batch_sz = 2
        for L, A, V in self.four_lengths:
            model = PromptModel(make_hp(L, A, V))
            swap_time_mixers(model, operator="cross_attn", query_len=50)
            model.eval()

            # Synthetic inputs with matched sequence lengths
            x_l = torch.randn(batch_sz, L, 300)
            x_a = torch.randn(batch_sz, A, 5)
            x_v = torch.randn(batch_sz, V, 20)

            with torch.no_grad():
                for code in range(7):
                    missing_mod = [code] * batch_sz
                    out = model(x_l, x_a, x_v, missing_mod)
                    self.assertEqual(
                        out.shape,
                        (batch_sz, 1),
                        f"Failed shape check for lengths ({L}/{A}/{V}), code {code}: got {out.shape}",
                    )
                    self.assertFalse(
                        torch.isnan(out).any(),
                        f"Detected NaN for lengths ({L}/{A}/{V}), code {code}",
                    )

    def test_gradients_reach_every_new_parameter(self):
        """Requirement 3: Verify that gradients reach EVERY new parameter across

        all 9 swapped time-mixing layers during backpropagation.
        """
        # Test on the longest sequence configuration
        L, A, V = 50, 375, 500
        model = PromptModel(make_hp(L, A, V))
        swap_time_mixers(model, operator="cross_attn", query_len=50)
        model.train()

        # Batch of 6 samples so that missing codes 0 through 5 are all active
        batch_sz = 6
        x_l = torch.randn(batch_sz, L, 300)
        x_a = torch.randn(batch_sz, A, 5)
        x_v = torch.randn(batch_sz, V, 20)
        missing_mod = list(range(6))

        out = model(x_l, x_a, x_v, missing_mod)
        loss = out.sum()
        loss.backward()

        unreached_parameters = []
        for layer_name in TIME_MIXING_LAYERS:
            layer = getattr(model, layer_name)
            for param_name, param in layer.named_parameters():
                full_param_name = f"{layer_name}.{param_name}"
                if param.grad is None:
                    unreached_parameters.append(f"{full_param_name} (grad is None)")
                elif torch.isnan(param.grad).any():
                    unreached_parameters.append(f"{full_param_name} (grad contains NaN)")
                elif param.grad.abs().sum().item() == 0.0:
                    unreached_parameters.append(f"{full_param_name} (grad is strictly 0)")

        self.assertEqual(
            unreached_parameters,
            [],
            f"Gradients failed to reach the following parameters:\n{unreached_parameters}",
        )

    def test_layer_replacement_type(self):
        """Verify swap_time_mixers replaces all 9 time-mixers while leaving channel-mixers as MLPLayer."""
        model = PromptModel(make_hp(50, 50, 50))
        swap_time_mixers(model, operator="cross_attn")

        for name in TIME_MIXING_LAYERS:
            self.assertIsInstance(getattr(model, name), LengthInvariant)

        # Channel mixers must stay MLPLayer
        for name in ["l2a", "l2v", "v2a", "v2l", "a2v", "a2l"]:
            self.assertIsInstance(getattr(model, name), MLPLayer)

    def test_pooling_ablation_variant(self):
        """Verify the pooling ablation operator forward pass."""
        op = LengthInvariant(out_len=50, embed_dim=30, operator="pooling")
        for in_len in [50, 100, 375, 500]:
            x = torch.randn(2, in_len, 30)
            out = op(x)
            self.assertEqual(out.shape, (2, 50, 30))


if __name__ == "__main__":
    unittest.main()
