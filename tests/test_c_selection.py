import unittest

import numpy as np

from modcma.c_maes import (
    selection, Parameters, Population, parameters, options, ModularCMAES, settings_from_dict
)


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
        # #75 F9: the losers of the pairs (0, 1) and (2, 3) keep their values, after the winners
        self.p.settings.modules.mirrored = options.Mirror.PAIRWISE

        selector = selection.Strategy(self.p.settings.modules)
        self.p.pop.f = -np.arange(4)
        selector.select(self.p)

        np.testing.assert_array_equal(self.p.pop.f, [-3.0, -1.0, -2.0, 0.0])
        self.assertEqual(self.p.pop.n_ranked, 2)
        self.assertEqual(self.p.stats.global_best.y, -3)
        self.assertEqual(self.p.stats.current_avg, -1.5)

    def test_pairwise_sort(self):
        # The winners and the unpaired last offspring by f, then the parents (after n_offspring,
        # ranked), then the losers by f; a NaN loses its pair, a tie loses the first
        pop = Population(1, 7)
        pop.f = np.array([3.0, 1.0, np.nan, 5.0, 2.0, 2.0, 0.5])
        pop.X = np.arange(7.0).reshape(1, 7)
        selection.Pairwise().sort(pop, 6)
        np.testing.assert_array_equal(pop.f, [0.5, 1.0, 2.0, 5.0, 2.0, 3.0, np.nan])
        np.testing.assert_array_equal(pop.X[0], [6, 1, 5, 3, 4, 0, 2])
        self.assertEqual(pop.n_ranked, 4)
        selection.NoPairwise().sort(pop, 6)
        self.assertEqual(pop.n_ranked, 7)
        np.testing.assert_array_equal(pop.f[:3], [0.5, 1.0, 2.0])

    def test_pairwise_elitism(self):
        # The parents are ranked with the winners; the losers come last and are cut first
        self.p.settings.modules.mirrored = options.Mirror.PAIRWISE
        self.p.settings.modules.elitist = True
        selector = selection.Strategy(self.p.settings.modules)
        self.p.stats.t = 1
        self.p.pop.f = np.array([1.0, 2.0, 3.0, 4.0])
        self.p.old_pop.f = np.array([0.5, 2.5, 6.0, 7.0])
        self.p.old_pop.n_ranked = 2
        selector.select(self.p)
        np.testing.assert_array_equal(self.p.pop.f, [0.5, 1.0, 2.5, 3.0])
        self.assertEqual(self.p.pop.n_ranked, 4)



class TestPairwiseValues(unittest.TestCase):
    """#75, F9: with pairwise selection every f stays finite, so MSR, the mean f and Stagnation
    read real values (the losers had f = +inf)"""

    def test_values_are_finite(self):
        es = ModularCMAES(settings_from_dict(5, mirrored="PAIRWISE", ssa="MSR",
                                             restart_strategy="RESTART", seed=1))
        stagnation = next(c for c in es.p.criteria.items if c.name == "Stagnation")
        for _ in range(30):
            es.step(lambda x: float(np.sum(x**2)))
            self.assertTrue(np.all(np.isfinite(es.p.pop.f)))
            self.assertEqual(es.p.pop.n_ranked, es.p.pop.n // 2)
            self.assertTrue(np.isfinite(es.p.stats.current_avg))
            # the ranked points first, sorted; then the losers, sorted
            r = es.p.pop.n_ranked
            self.assertTrue(np.all(np.diff(es.p.pop.f[:r]) >= 0))
            self.assertTrue(np.all(np.diff(es.p.pop.f[r:]) >= 0))
        self.assertTrue(np.all(np.isfinite(stagnation.median_fitnesses)))

    def test_msr_grows_sigma_on_a_linear_function(self):
        # Before #75, MSR counted the winners against lambda / 2 and sigma shrank every
        # generation, whatever f did
        es = ModularCMAES(settings_from_dict(5, mirrored="PAIRWISE", ssa="MSR", sigma0=0.1,
                                             lb=np.full(5, -1e9), ub=np.full(5, 1e9), seed=2))
        for _ in range(30):
            es.step(lambda x: float(x[0]))
        self.assertGreater(es.p.mutation.sigma, 1.0)

    def test_settings_default_mu_of_the_even_lambda(self):
        # PAIRWISE raises an odd default lambda0 (d = 3: 7) to 8; the default mu0 is half of that
        # (it was half of 7, 3); a given mu0 is kept
        s = settings_from_dict(3, mirrored="PAIRWISE")
        self.assertEqual((s.lambda0, s.mu0), (8, 4))
        s = settings_from_dict(3, mirrored="PAIRWISE", mu0=3)
        self.assertEqual((s.lambda0, s.mu0), (8, 3))
        s = settings_from_dict(3, mirrored="PAIRWISE", ssa="SA")
        self.assertEqual((s.lambda0, s.mu0), (8, 2))
        s = settings_from_dict(3, mirrored="MIRRORED")
        self.assertEqual((s.lambda0, s.mu0), (7, 3))


if __name__ == "__main__":
    unittest.main()
