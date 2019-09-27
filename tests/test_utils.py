import unittest
import numpy as np
from src import utils


class TestUtils(unittest.TestCase):
    def test_scale_with_treshold(self):
        threshold = 5
        z = np.ones(20)
        new_z = utils._scale_with_threshold(z.copy(), threshold)
        new_z_norm = np.linalg.norm(new_z)
        self.assertNotEqual(
            (z == new_z).all(), True
        )
        self.assertNotEqual(
            np.linalg.norm(z), new_z_norm
        )
        self.assertGreater(new_z_norm, threshold)

    def test_correct_bounds(self):
        x = np.ones(5) * np.array([2, 4, 6, -7, 3])
        ub, lb = 5, -5
        new_x = utils._correct_bounds(x.copy(), ub, lb)
        self.assertNotEqual(
            (x == new_x).all(), True
        )
        self.assertGreaterEqual(
            np.min(new_x), -5
        )
        self.assertLessEqual(
            np.max(new_x), 5
        )
        self.assertEqual(
            (x[[0, 1, 4]] == new_x[[0, 1, 4]]).all(), True
        )

    def test_ert(self):
        evals = [5000, 45000, 1000, 100, 10]
        budget = 10000
        ert, ert_sd = utils.ert(evals, budget)
        self.assertAlmostEqual(
            ert, 12777.5
        )
        self.assertAlmostEqual(
            ert_sd, 17484.642861665
        )
        ert, ert_sd = utils.ert([500000], budget)
        self.assertEqual(ert, float("inf"))
        self.assertEqual(ert_sd, 0)
        ert, ert_sd = utils.ert([], budget)
        self.assertEqual(ert, float("inf"))
        self.assertEqual(ert_sd, 0)


if __name__ == "__main__":
    unittest.main()
