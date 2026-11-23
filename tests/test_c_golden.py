"""Golden trajectory test of the C++ ModularCMAES (TASKS.md, T0.1); see tests/golden.py."""

import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestGoldenTrajectories(unittest.TestCase):
    def test_trajectories_match_fixture(self):
        # A fresh interpreter: other tests change module-level state (constants, tolerances, seeds).
        # Every driver (step, ask/tell from Python, vectorized) must match the same fixture.
        result = subprocess.run(
            [sys.executable, "-m", "tests.golden", "check", "--drivers", "all"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if output.startswith("SKIP"):
            self.skipTest(output.strip())
        self.assertEqual(result.returncode, 0, output)


if __name__ == "__main__":
    unittest.main()
