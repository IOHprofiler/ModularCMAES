import unittest

import numpy as np

from modcma.c_maes import utils


class TestUtils(unittest.TestCase):
    def test_rng(self):
        utils.set_seed(10)
        u1 = utils.random_uniform()
        n1 = utils.random_normal()
        # TODO: this is needed because the normal generator caches data
        utils.random_normal()
        utils.set_seed(10)
        u2 = utils.random_uniform()
        n2 = utils.random_normal()
        self.assertEqual(u1, u2)
        self.assertEqual(n1, n2)

    def test_ert(self):
        running_times = np.array([10, 9])
        ert, n_succes = utils.compute_ert(
            running_times.astype(int), 10
        )   
        self.assertEqual(ert, 19)
        self.assertEqual(n_succes, 1)

        running_times = np.array([10, 10])
        ert, n_succes = utils.compute_ert(
            running_times.astype(int), 10
        )   
        self.assertEqual(ert, float("inf"))
        self.assertEqual(n_succes, 0)

        running_times = np.array([9, 9])
        ert, n_succes = utils.compute_ert(
            running_times.astype(int), 10
        )   
        self.assertEqual(ert, 9)
        self.assertEqual(n_succes, 2)



if __name__ == "__main__":
    unittest.main()

