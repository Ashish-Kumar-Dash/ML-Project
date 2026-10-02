import unittest
import torch

from tools.measure_peak_memory import measure_single_batch


class TestPeakMemory(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for memory measurement tests")
    def test_batch16_unaligned_fits_in_memory(self):
        """Verify that batch size 16 at unaligned lengths (50/375/500) fits comfortably

        within 4.5 GiB VRAM on CUDA.
        """
        res = measure_single_batch(
            batch_size=16,
            variant="baseline",
            len_l=50,
            len_a=375,
            len_v=500,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertLess(res["bwd_peak_gib"], 4.5)
        self.assertGreater(res["bwd_peak_gib"], 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for memory measurement tests")
    def test_batch64_unaligned_triggers_oom(self):
        """Empirically confirm that batch size 64 at unaligned lengths (50/375/500)

        exceeds the 12 GB VRAM capacity and triggers an OOM.
        """
        from tools.measure_peak_memory import run_in_subprocess
        res = run_in_subprocess(batch_size=64, variant="baseline")
        self.assertIn("OOM", res["status"])


if __name__ == "__main__":
    unittest.main()
