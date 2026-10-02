import os
import subprocess
import unittest
from pathlib import Path


class TestUpstreamClean(unittest.TestCase):
    def setUp(self):
        # Resolve repo root directory (ML-Project)
        self.repo_root = Path(__file__).resolve().parent.parent
        self.upstream_dir = self.repo_root / "third_party" / "MPLMM"

    def test_upstream_directory_exists(self):
        self.assertTrue(
            self.upstream_dir.is_dir(),
            f"Expected upstream directory to exist at {self.upstream_dir}",
        )

    def test_upstream_is_clean(self):
        """Verify git status --porcelain on upstream submodule is completely empty."""
        result = subprocess.run(
            ["git", "-C", str(self.upstream_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"git status command failed with stderr: {result.stderr}",
        )
        self.assertEqual(
            result.stdout.strip(),
            "",
            f"Upstream submodule at {self.upstream_dir} has uncommitted or untracked changes:\n{result.stdout}",
        )


if __name__ == "__main__":
    unittest.main()
