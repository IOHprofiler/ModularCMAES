import unittest

import numpy as np

from modcma.c_maes import selection, Parameters, parameters, options


class TestSelection(unittest.TestCase):
    """Test case for sampling methods of C++ version Modular CMA-ES."""

    def setUp(self):
        settings = parameters.Settings(1, lambda0=4)
        self.p = Parameters(settings)

    def test_select(self):
        selector = selection.Strategy(self.p.settings.modules)
        self.p.pop.f = -np.arange(4)
        selector.select(self.p)
        self.assertTrue(np.all(self.p.pop.f == -np.arange(4)[::-1]))
        self.assertEqual(self.p.stats.global_best.y, -3)

    def test_elitsm(self):
        self.p.settings.modules.elitist = True
        selector = selection.Strategy(self.p.settings.modules)
        self.p.stats.t = 1
        self.p.pop.f = -np.arange(4)
        self.p.old_pop.f = -np.arange(1, 5)

        selector.select(self.p)
        self.assertEqual(self.p.stats.global_best.y, -4)
        self.assertTrue(np.all(self.p.pop.f == np.array([-4., -3., -3., -2.])))

    def test_pairwise(self):
        self.p.settings.modules.mirrored = options.Mirror.PAIRWISE

        selector = selection.Strategy(self.p.settings.modules)
        self.p.pop.f = -np.arange(4)
        selector.select(self.p)

        self.assertTrue(np.all(self.p.pop.f == np.array([-3., -1., float("inf"), float("inf")])))
        self.assertEqual(self.p.stats.global_best.y, -3)


if __name__ == "__main__":
    unittest.main()
