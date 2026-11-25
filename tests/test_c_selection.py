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
        # (mu + lambda): only the mu = 2 parents (the first, best columns of the sorted old_pop)
        # compete with the offspring, not all lambda of the previous generation
        self.p.settings.modules.elitist = True
        selector = selection.Strategy(self.p.settings.modules)
        self.p.stats.t = 1
        self.p.pop.f = -np.arange(4)
        self.p.old_pop.f = np.array([-4.0, -3.0, -2.5, -2.4])

        selector.select(self.p)
        self.assertEqual(self.p.mu, 2)
        self.assertEqual(self.p.stats.global_best.y, -4)
        self.assertTrue(np.all(self.p.pop.f == np.array([-4.0, -3.0, -3.0, -2.0])))
        # old_pop keeps all its columns (MSR and PSR compare with it)
        self.assertEqual(self.p.old_pop.n, 4)

    def test_elitist_parents_take_the_current_sigma(self):
        # The parents' steps are recomputed with the generation's sigma (the framework paper,
        # Eq. 1-3), not with the step size each was sampled with
        self.p.settings.modules.elitist = True
        selector = selection.Strategy(self.p.settings.modules)
        self.p.stats.t = 1
        self.p.mutation.sigma = 0.5
        self.p.pop.f = np.arange(4.0)
        self.p.old_pop.f = -np.arange(4.0, 0, -1)
        self.p.old_pop.X = np.full((1, 4), 3.0)
        self.p.old_pop.S = np.full((1, 4), 7.0)

        selector.select(self.p)
        m = self.p.adaptation.m
        np.testing.assert_allclose(self.p.pop.Y[:, :2], (3.0 - m[0]) / 0.5)
        np.testing.assert_allclose(self.p.pop.S[:, :2], 0.5)

    def test_pairwise(self):
        self.p.settings.modules.mirrored = options.Mirror.PAIRWISE

        selector = selection.Strategy(self.p.settings.modules)
        self.p.pop.f = -np.arange(4)
        selector.select(self.p)

        self.assertTrue(
            np.all(self.p.pop.f == np.array([-3.0, -1.0, float("inf"), float("inf")]))
        )
        self.assertEqual(self.p.stats.global_best.y, -3)


if __name__ == "__main__":
    unittest.main()
