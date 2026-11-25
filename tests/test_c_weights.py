"""The recombination weights, in particular the NES/xNES utilities (RecombinationWeights.UTILITY)."""

import math
import unittest

import numpy as np

import modcma


def sphere(x):
    return float(np.sum((x - 1) ** 2))


def utilities(n):
    a = np.maximum(0.0, math.log(n / 2 + 1) - np.log(np.arange(1, n + 1)))
    return a / a.sum() - 1.0 / n


class TestUtilityWeights(unittest.TestCase):
    def test_utilities(self):
        for lam in (4, 6, 10, 13, 24, 100):
            es = modcma.ModularCMAES(modcma.settings_from_dict(5, lambda0=lam, weights="UTILITY"))
            w = es.p.weights
            np.testing.assert_allclose(w.weights, utilities(lam), atol=1e-15)
            self.assertAlmostEqual(float(np.sum(w.weights)), 0.0, places=14)
            self.assertTrue(w.utility)
            self.assertAlmostEqual(w.mueff_mean, 1 / np.sum(utilities(lam) ** 2))
            # the positive weights (for SA, LPXNES, ...) are the non-negative part, normalized
            self.assertAlmostEqual(float(np.sum(w.positive)), 1.0)
            self.assertTrue(np.all(w.positive >= 0))

    def test_the_active_update_is_set(self):
        settings = modcma.settings_from_dict(5, weights="UTILITY")
        self.assertTrue(settings.modules.active)
        none = modcma.settings_from_dict(5, weights="UTILITY", matrix_adaptation="NONE")
        self.assertFalse(none.modules.active)

    def test_natural_gradient_with_active_uses_the_utilities(self):
        # CMA-ES's negative weights broke the natural gradient (dev/overhaul-2.0.md F2)
        settings = modcma.settings_from_dict(5, matrix_adaptation="NATURAL_GRADIENT", active=True)
        self.assertEqual(settings.modules.weights, modcma.options.RecombinationWeights.UTILITY)
        other = modcma.settings_from_dict(5, matrix_adaptation="MATRIX", active=True)
        self.assertEqual(other.modules.weights, modcma.options.RecombinationWeights.DEFAULT)

    def test_the_mean_uses_all_points(self):
        # m <- m + sum_k u_k (x_k - m) over the lambda sorted points
        es = modcma.ModularCMAES(modcma.settings_from_dict(5, weights="UTILITY", seed=1))
        es.mutate(sphere)
        es.select()
        m_old = np.array(es.p.adaptation.m)
        X = np.array(es.p.pop.X)  # (d, lambda), sorted by f
        es.recombine()
        expected = m_old + (X - m_old[:, None]) @ utilities(X.shape[1])
        np.testing.assert_allclose(es.p.adaptation.m, expected, atol=1e-12)

    def test_sequential_selection_uses_the_utilities_of_the_evaluated_points(self):
        w = modcma.ModularCMAES(modcma.settings_from_dict(5, lambda0=10, weights="UTILITY")).p.weights
        for n in (6, 8, 10):
            u = w.for_population(n)
            np.testing.assert_allclose(u, utilities(n), atol=1e-15)

    def test_xnes_sigma_grows_on_a_linear_function(self):
        # NATURAL_GRADIENT + XNES + UTILITY is xNES; with the active weights of CMA-ES instead
        # sigma shrinks on a linear function (dev/overhaul-2.0.md F2)
        growth = []
        for seed in range(10):
            settings = modcma.settings_from_dict(
                2, sigma0=1e-3, budget=10**9, seed=seed, matrix_adaptation="NATURAL_GRADIENT",
                ssa="XNES", weights="UTILITY", lb=np.full(2, -np.inf), ub=np.full(2, np.inf),
            )
            es = modcma.ModularCMAES(settings)
            for _ in range(30):
                es.step(lambda x: float(x[0]))
            growth.append(math.log(es.p.mutation.sigma / 1e-3))
        self.assertGreater(np.mean(growth), 0.5)

    def test_every_matrix_adaptation_converges(self):
        for option in ("COVARIANCE", "SEPARABLE", "MATRIX", "CHOLESKY", "CMSA",
                       "COVARIANCE_NO_EIGV", "NATURAL_GRADIENT", "NONE"):
            settings = modcma.settings_from_dict(
                5, matrix_adaptation=option, weights="UTILITY", x0=np.full(5, -1.0),
                budget=10_000, target=1e-8, seed=1,
            )
            es = modcma.ModularCMAES(settings)
            es.run(sphere)
            self.assertLessEqual(es.p.stats.global_best.y, 1e-8, option)


if __name__ == "__main__":
    unittest.main()
