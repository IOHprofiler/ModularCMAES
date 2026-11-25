"""The design decisions of #83: where the modules now follow the framework paper and the papers
of the modules (elitism, MSR, PSR, SR, the IPOP limit, pairwise selection with active)."""

import math
import unittest

import numpy as np

from modcma import c_maes
from modcma.c_maes import options, parameters, Parameters, Population


def make(d=4, **modules):
    settings = c_maes.settings_from_dict(d, verbose=False, seed=1, **modules)
    return Parameters(settings)


def population(f):
    pop = Population(1, len(f))
    pop.f = np.asarray(f, dtype=float)
    return pop


class TestMSR(unittest.TestCase):
    def test_compares_with_the_03_quantile_of_the_previous_generation(self):
        p = make(ssa="MSR", lambda0=10)
        msr = p.mutation
        self.assertIsInstance(msr, c_maes.mutation.MSR)
        self.assertEqual(msr.quantile, 0.3)
        # j = 0.3 * 10 = 3 (0-based): the 4th best of the previous generation, 3.0. Six of the
        # offspring (-1, 0, 1, 2, 2.5, 2.9) are better, four are not; with the median (4.5) the
        # counts would be 6 vs 8.
        old = population(np.arange(10.0))
        new = population([-1, 0, 1, 2, 2.5, 2.9, 3, 5, 6, 7])
        sigma = msr.sigma
        msr.adapt(p.weights, p.adaptation, new, old, p.stats, 10)
        k = 6
        z = (2 / 10) * (k - 11 / 2)
        s = p.weights.cs * z
        self.assertAlmostEqual(msr.s, s)
        self.assertAlmostEqual(msr.sigma, sigma * math.exp(s / (2 - 2 / 4)))

    def test_damping_at_d1(self):
        # 2 - 2/d is 0 at d = 1: sigma would become 0 or infinite
        p = make(d=1, ssa="MSR")
        self.assertEqual(p.weights.damps, 1.0)
        self.assertEqual(make(d=4, ssa="MSR").weights.damps, 1.5)


class TestPSR(unittest.TestCase):
    def test_constants_of_loshchilov_2014(self):
        p = make(ssa="PSR")
        self.assertEqual(p.weights.cs, 0.3)
        self.assertEqual(p.weights.damps, 1.0)
        self.assertEqual(p.mutation.success_ratio, 0.25)


class TestSR(unittest.TestCase):
    def test_success_against_the_previous_generation(self):
        # A generation whose best beats the previous generation's best is a success, even when
        # it does not beat the best of the run
        p = make(d=2, lambda0=4, ssa="SR")
        p.pop.f = np.array([-100.0, 9.0, 9.0, 9.0])  # the best of the run
        p.selection.select(p)

        ratio = p.stats.success_ratio
        p.old_pop.f = np.array([5.0, 6.0, 7.0, 8.0])
        p.pop.f = np.array([4.0, 9.0, 9.0, 9.0])
        p.selection.select(p)
        self.assertEqual(p.stats.current_best.y, -100.0)
        self.assertAlmostEqual(p.stats.success_ratio, (1 - 1 / 12) * ratio + 1 / 12)

        ratio = p.stats.success_ratio
        p.old_pop.f = np.array([4.0, 6.0, 7.0, 8.0])
        p.pop.f = np.array([4.5, 9.0, 9.0, 9.0])
        p.selection.select(p)
        self.assertAlmostEqual(p.stats.success_ratio, (1 - 1 / 12) * ratio)

    def test_tpa_points_do_not_count(self):
        # Only selection updates the success ratio, once per generation
        es = c_maes.ModularCMAES(c_maes.settings_from_dict(3, ssa="TPA", seed=2, verbose=False))
        ratio = es.p.stats.success_ratio
        es.step(lambda x: float(np.sum(x**2)))
        self.assertIn(es.p.stats.success_ratio,
                      ((1 - 1 / 12) * ratio, (1 - 1 / 12) * ratio + 1 / 12))


class TestIPOP(unittest.TestCase):
    def test_limit_is_2_9_lambda0(self):
        p = make(d=40, lambda0=4, restart_strategy="IPOP")
        ipop = p.restart_strategy
        self.assertEqual(ipop.max_lambda_factor, 512)
        sizes = []
        for _ in range(12):
            ipop.update(p)
            sizes.append(p.lamb)
        self.assertEqual(max(sizes), 512 * 4)
        self.assertEqual(sizes[-1], 512 * 4)
        # (d lambda0)^2 = 25600 would allow more
        self.assertLess(max(sizes), (40 * 4) ** 2)

    def test_non_power_factor_stops_at_the_limit(self):
        p = make(d=5, lambda0=6, restart_strategy="IPOP")
        p.restart_strategy.ipop_factor = 3.0
        for _ in range(10):
            p.restart_strategy.update(p)
        self.assertEqual(p.lamb, 512 * 6)


class TestPairwiseActive(unittest.TestCase):
    def test_losers_get_no_negative_weights(self):
        p = make(lambda0=10, mu0=3, active=True, mirrored="PAIRWISE")
        w = p.weights
        self.assertTrue(w.pairwise)
        full = w.for_population(10)
        self.assertTrue(np.all(full[3:] < 0))
        # In the update only the 5 winners are ranked: the losers (f = +inf, sorted last) get 0
        pop = Population(4, 10)
        pop.f = np.array([0, 1, 2, 3, 4] + [np.inf] * 5, dtype=float)
        p.pop = pop
        c_before = p.adaptation.C.copy()
        p.adaptation.adapt_matrix(p.weights, p.settings.modules, p.pop, p.mu, p.settings, p.stats)
        self.assertFalse(np.allclose(c_before, p.adaptation.C))

    def test_with_default_mu_active_has_no_effect(self):
        # mu = lambda / 2 winners: every negative weight would go to a loser
        rng = np.random.default_rng(1)
        results = []
        for active in (False, True):
            p = make(lambda0=10, active=active, mirrored="PAIRWISE")
            p.pop = Population(4, 10)
            p.pop.Y = rng.standard_normal((4, 10)) if not results else results[0][1]
            p.pop.f = np.array([0, 1, 2, 3, 4] + [np.inf] * 5, dtype=float)
            p.adaptation.adapt_matrix(p.weights, p.settings.modules, p.pop, p.mu, p.settings, p.stats)
            results.append((p.adaptation.C.copy(), p.pop.Y.copy()))
        np.testing.assert_allclose(results[0][0], results[1][0])


class TestElitismRuns(unittest.TestCase):
    def test_elitist_run_converges(self):
        # (mu + lambda) on the sphere: the best value never gets worse
        es = c_maes.ModularCMAES(c_maes.settings_from_dict(5, elitist=True, budget=3000, seed=3,
                                                          verbose=False))
        best = []
        while not es.break_conditions():
            es.step(lambda x: float(np.sum(x**2)))
            best.append(es.p.stats.current_best.y)
        self.assertTrue(all(b2 <= b1 for b1, b2 in zip(best, best[1:])))
        self.assertLess(best[-1], 1e-8)


if __name__ == "__main__":
    unittest.main()
