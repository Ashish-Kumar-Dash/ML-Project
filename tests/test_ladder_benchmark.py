"""Unit test verifying the sequence ladder timing benchmark and artifacts."""

import json
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestLadderTimingBenchmark(unittest.TestCase):
    def test_ladder_benchmark_artifact_exists_and_valid(self):
        json_path = REPO_ROOT / "runs" / "ladder_timing_benchmark.json"
        self.assertTrue(
            json_path.exists(),
            f"Expected benchmark artifact at {json_path}",
        )

        with open(json_path, "r") as f:
            data = json.load(f)

        self.assertIn("results", data)
        results = data["results"]
        self.assertEqual(len(results), 8, "Expected 8 configurations (4 lengths x 2 variants)")

        # Verify invariant parameter constancy and dense parameter explosion
        inv_counts = [r["time_mixing_params"] for r in results if r["variant"] == "invariant"]
        self.assertTrue(
            all(c == 46980 for c in inv_counts),
            f"Invariant parameters must be strictly 46,980, got {inv_counts}",
        )

        dense_counts = {r["ladder_name"]: r["time_mixing_params"] for r in results if r["variant"] == "baseline"}
        self.assertEqual(dense_counts["L50"], 37650)
        self.assertEqual(dense_counts["L100"], 92750)
        self.assertEqual(dense_counts["L200"], 262950)
        self.assertEqual(dense_counts["Native"], 972175)

        # Verify timings are positive and reasonable
        for r in results:
            self.assertGreater(r["mean_epoch_time_sec"], 0.0)
            self.assertGreater(r["throughput_samples_per_sec"], 10.0)
            self.assertLess(r["peak_vram_gib"], 12.0)


if __name__ == "__main__":
    unittest.main()
