import unittest
import torch
import lenmod.compat
from lenmod.config import make_hp
from src.model import PromptModel


class TestDeadPrompts(unittest.TestCase):
    def test_zero_gradient_on_missing_type_prompts(self):
        """Verify that missing_type_prompt and coupling matrices (m_a, m_v, m_l)

        have identically zero gradients after backward pass due to 0 x 0 initialization.
        """
        hp = make_hp(50, 50, 50)
        model = PromptModel(hp)
        model.train()

        B = 7
        x_l = torch.randn(B, 50, 300)
        x_a = torch.randn(B, 50, 5)
        x_v = torch.randn(B, 50, 20)
        missing_mod = list(range(7))

        out = model(x_l, x_a, x_v, missing_mod)
        loss = out.sum()
        loss.backward()

        # The first four must be exactly 0 due to zero-initialization bilinear product
        self.assertTrue(
            (model.missing_type_prompt.grad == 0).all().item(),
            "missing_type_prompt gradient is not all zeros!",
        )
        self.assertTrue(
            (model.m_a.grad == 0).all().item(),
            "m_a gradient is not all zeros!",
        )
        self.assertTrue(
            (model.m_v.grad == 0).all().item(),
            "m_v gradient is not all zeros!",
        )
        self.assertTrue(
            (model.m_l.grad == 0).all().item(),
            "m_l gradient is not all zeros!",
        )

        # Generative prompt and additive missing-signal prompts must have non-zero gradients
        self.assertFalse(
            (model.generative_prompt.grad == 0).all().item(),
            "generative_prompt unexpectedly has zero gradient!",
        )
        self.assertFalse(
            (model.promptl_m.grad == 0).all().item(),
            "promptl_m unexpectedly has zero gradient!",
        )


if __name__ == "__main__":
    unittest.main()
