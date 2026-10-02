import tempfile
import unittest
import torch
from torch.utils.data import DataLoader, Dataset

import lenmod
from lenmod.config import make_hp
from src.model import PromptModel


class MockDataset(Dataset):
    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.x_l = torch.randn(n_samples, 50, 300)
        self.x_a = torch.randn(n_samples, 50, 5)
        self.x_v = torch.randn(n_samples, 50, 20)
        self.y = torch.randn(n_samples, 1, 1)
        self.modes = torch.randint(0, 7, (n_samples,)).tolist()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (self.x_l[idx], self.x_a[idx], self.x_v[idx]), self.y[idx], self.modes[idx]


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.train_loader = DataLoader(MockDataset(32), batch_size=16, shuffle=False)
        self.valid_loader = DataLoader(MockDataset(16), batch_size=16, shuffle=False)
        self.test_loader = DataLoader(MockDataset(16), batch_size=16, shuffle=False)
        self.hp = make_hp(50, 50, 50, num_epochs=2, lr=1e-3)
        self.model = PromptModel(self.hp)

    def test_fit_and_evaluate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = f"{tmpdir}/model_test.pt"
            result = lenmod.fit(
                model=self.model,
                train_loader=self.train_loader,
                valid_loader=self.valid_loader,
                test_loader=self.test_loader,
                hyp_params=self.hp,
                checkpoint_path=ckpt_path,
                accum_steps=4,
                quiet=True,
            )

            # 1. State dict checkpoint verification
            ckpt = torch.load(ckpt_path, weights_only=False)
            self.assertIn("state_dict", ckpt)
            self.assertIn("epoch", ckpt)
            self.assertIn("val_loss", ckpt)

            # 2. Test metrics verification
            self.assertIn("test_metrics", result)
            metrics = result["test_metrics"]
            self.assertIn("mae", metrics)
            self.assertIn("acc2", metrics)
            self.assertIn("f1", metrics)
            self.assertIn("loss", metrics)

    def test_evaluate_forced_missing_code(self):
        loss, metrics, preds, truths = lenmod.evaluate(
            model=self.model,
            loader=self.test_loader,
            forced_missing_code=0,
        )
        self.assertEqual(len(preds), 16)
        self.assertIn("acc2", metrics)

    def test_eval_cases(self):
        res = lenmod.eval_cases(self.model, self.test_loader, quiet=True)
        # Verify all 7 cases exist
        self.assertEqual(set(res["cases"].keys()), set(range(7)))
        # Verify mean calculation over cases 0 to 5
        expected_mean_acc = sum(res["cases"][c]["acc2"] for c in range(6)) / 6.0
        self.assertAlmostEqual(
            res["mean_0_to_5"]["acc2"], expected_mean_acc, places=5
        )
        # Verify complete data is present
        self.assertEqual(res["complete_data"], res["cases"][6])


if __name__ == "__main__":
    unittest.main()

