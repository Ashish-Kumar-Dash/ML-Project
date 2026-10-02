import unittest
import torch
import lenmod.compat
from lenmod.config import make_hp
from src.model import PromptModel


class TestForwardSmoke(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.hp = make_hp(50, 50, 50)
        self.model = PromptModel(self.hp)
        self.model.eval()

        # Random inputs with aligned MOSI dimensions
        self.x_l = torch.randn(self.batch_size, 50, 300)
        self.x_a = torch.randn(self.batch_size, 50, 5)
        self.x_v = torch.randn(self.batch_size, 50, 20)

    def test_forward_all_missing_codes(self):
        """Verify forward pass produces valid (B, 1) predictions for all codes 0 to 6."""
        with torch.no_grad():
            for code in range(7):
                missing_mod = [code] * self.batch_size
                out = self.model(self.x_l, self.x_a, self.x_v, missing_mod)
                self.assertEqual(
                    out.shape,
                    (self.batch_size, 1),
                    f"Expected shape ({self.batch_size}, 1), got {out.shape} for code {code}",
                )
                self.assertFalse(
                    torch.isnan(out).any(),
                    f"Detected NaN in output for missing code {code}",
                )


if __name__ == "__main__":
    unittest.main()
