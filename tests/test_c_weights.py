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
        for lam in (2, 6, 10, 13, 24, 100):
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


class TestCSALearningRates(unittest.TestCase):
    """cs and damps of CSA: Hansen, The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772v2,
    Table 1, Eq. (55), without pycma's cap of cs at 1/2 (#138)."""

    @staticmethod
    def tutorial(d, lam):
        mu = lam // 2
        w = math.log((lam + 1) / 2) - np.log(np.arange(1, mu + 1))
        mueff = w.sum() ** 2 / (w**2).sum()
        cs = (mueff + 2) / (d + mueff + 5)
        damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        return cs, damps

    def test_defaults_follow_the_tutorial(self):
        # d = 2, lambda = 8: cs = 0.4792 (pycma 0.5, its cap; modcma 1.1 and 1.2 0.6053);
        # d = 5, lambda = 512: cs = 0.94, above pycma's cap
        for d, lam in ((2, 8), (2, 6), (5, 8), (10, 10), (20, 12), (5, 512), (2, 100)):
            w = modcma.ModularCMAES(modcma.Settings(d, lambda0=lam)).p.weights
            cs, damps = self.tutorial(d, lam)
            self.assertAlmostEqual(w.cs, cs, places=14, msg=(d, lam))
            self.assertAlmostEqual(w.damps, damps, places=14, msg=(d, lam))
        self.assertAlmostEqual(self.tutorial(2, 8)[0], 0.4792, places=4)
        self.assertGreater(self.tutorial(5, 512)[0], 0.94)

    def test_ipop_recomputes_them_for_the_larger_population(self):
        def rastrigin(x):
            return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

        settings = modcma.settings_from_dict(5, restart_strategy="IPOP", budget=20_000, seed=1)
        es = modcma.ModularCMAES(settings)
        es.run(rastrigin)
        self.assertGreater(es.p.lamb, 8)
        cs, damps = self.tutorial(5, es.p.lamb)
        self.assertAlmostEqual(es.p.weights.cs, cs, places=14)
        self.assertAlmostEqual(es.p.weights.damps, damps, places=14)

    def test_other_rules_keep_their_cs(self):
        d = 10
        for ssa, cs in (("SR", 2 / (d + 2)), ("MXNES", 1.0), ("TPA", 0.3)):
            w = modcma.ModularCMAES(modcma.settings_from_dict(d, ssa=ssa)).p.weights
            self.assertAlmostEqual(w.cs, cs, msg=ssa)

    def test_an_explicit_cs_is_kept(self):
        w = modcma.ModularCMAES(modcma.Settings(5, cs=0.7)).p.weights
        self.assertEqual(w.cs, 0.7)
        self.assertAlmostEqual(w.damps, 1.7)


class TestXNESFamily(unittest.TestCase):
    """The rates of XNES, MXNES and LPXNES: Krause, Glasmachers & Igel (2017), Table 3 (#72)."""

    def test_xnes_sa_rate(self):
        # c_sigma = mueff / (2 ln(d) sqrt(d)), ln(max(2, d)) at d = 1 (F27; it was 0.01)
        for d in (1, 2, 10, 40):
            w = modcma.ModularCMAES(modcma.settings_from_dict(d, ssa="XNES")).p.weights
            expected = w.mueff / (2 * math.log(max(2, d)) * math.sqrt(d))
            self.assertAlmostEqual(w.cs, expected, places=14, msg=d)

    def test_lpxnes_at_d_1(self):
        # beta = ln 2 / (sqrt(d) ln d) was inf at d = 1 (F16)
        settings = modcma.settings_from_dict(1, ssa="LPXNES", budget=2000, target=1e-8, seed=1)
        es = modcma.ModularCMAES(settings)
        self.assertEqual(es.p.weights.beta, 1.0)
        es.run(lambda x: float(x[0] ** 2))
        self.assertLess(es.p.stats.global_best.y, 1e-8)

    def test_the_path_has_a_rate_of_at_most_one(self):
        # LPXNES's cs = 9 mueff / (10 sqrt(d)) = 2.37 at d = 2, lambda = 12: as the rate of ps it
        # made sqrt(cs (2 - cs) mueff) NaN. The rate of ps is min(cs, 1) (cs_path)
        for ma in ("COVARIANCE", "MATRIX", "SEPARABLE"):
            settings = modcma.settings_from_dict(
                2, ssa="LPXNES", matrix_adaptation=ma, lambda0=12, budget=2000, seed=1,
                always_compute_eigv=True,
            )
            es = modcma.ModularCMAES(settings)
            w = es.p.weights
            self.assertGreater(w.cs, 2)
            self.assertEqual(w.cs_path, 1.0)
            self.assertAlmostEqual(w.sqrt_cs_mueff, math.sqrt(w.mueff))
            for _ in range(10):
                es.step(sphere)
                self.assertTrue(np.all(np.isfinite(es.p.adaptation.ps)), ma)
        for ssa in ("CSA", "TPA", "MSR", "PSR", "SR", "XNES", "MXNES", "LPXNES"):
            w = modcma.ModularCMAES(modcma.settings_from_dict(20, ssa=ssa)).p.weights
            self.assertEqual(w.cs_path, w.cs, ssa)

    def test_natural_gradient_with_lpxnes_converges(self):
        # NATURAL_GRADIENT gave LPXNES xNES's rate, and the run did not converge (F3); LPXNES is
        # slow on its own (one step size per coordinate, F1): about 40000 evaluations here
        settings = modcma.settings_from_dict(
            10, ssa="LPXNES", matrix_adaptation="NATURAL_GRADIENT", budget=80_000, target=1e-8,
            seed=1,
        )
        es = modcma.ModularCMAES(settings)
        es.run(sphere)
        self.assertLess(es.p.stats.global_best.y, 1e-8)

    def test_convex_weights_with_utility(self):
        # XNES (without NATURAL_GRADIENT) and LPXNES weight the mu best with `positive`, which
        # sums to 1 (Krause et al. 2017, Sec. 2), also with UTILITY
        settings = modcma.settings_from_dict(4, lambda0=8, ssa="XNES", weights="UTILITY", seed=3)
        es = modcma.ModularCMAES(settings)
        es.mutate(sphere)
        es.select()
        es.recombine()
        p = es.p
        sigma = p.mutation.sigma
        w = np.asarray(p.weights.positive)
        z = np.asarray(p.pop.Z)[:, : len(w)]
        expected = sigma * math.exp(
            p.weights.cs / 2 * float(w @ (np.sum(z**2, axis=0) - 4))
        )
        es.adapt()
        self.assertAlmostEqual(p.mutation.sigma, expected, places=12)


def pycma_weights(mu, lam, d, c1, cmu):
    """The weights of pycma 4.6.0 with CMA_mu=mu and popsize lam, for the learning rates c1, cmu.

    pycma: RecombinationWeights(2 mu) (options_parameters.py:1010-1013), or (lam) for the default
    mu = lam // 2; finalize_negative_weights(d, c1, cmu); and the weights of a population of lam,
    RecombinationWeights.__call__(lam) (recombination_weights.py:191-199). This reimplementation
    agrees with pycma's class to 2.2e-16 for the cases of test_matches_pycma (checked 2026-11-27).
    """
    n = lam if mu == lam // 2 else 2 * mu
    raw = math.log((n + 1) / 2) - np.log(np.arange(1, n + 1))
    k = int(np.sum(raw > 0))
    pos, neg = raw[:k] / raw[:k].sum(), raw[k:] / -raw[k:].sum()
    mueff = 1 / np.sum(pos**2)
    mueff_neg = neg.sum() ** 2 / np.sum(neg**2)
    total = min(1 + c1 / cmu, (1 - c1 - cmu) / (d * cmu), 1 + 2 * mueff_neg / (mueff + 2))
    neg = neg * total
    if lam > n:
        neg = np.concatenate([np.zeros(lam - n), neg])
    elif lam < n:
        neg = neg[len(neg) - (lam - k):]
    return np.concatenate([pos, neg])


class TestDefaultWeightsForMu(unittest.TestCase):
    """DEFAULT weights with mu != lambda / 2 (#73, F8)."""

    def test_matches_pycma(self):
        d = 10
        for mu, lam in ((3, 10), (6, 10), (5, 10), (2, 20), (5, 11), (7, 10), (4, 7)):
            w = modcma.ModularCMAES(modcma.Settings(d, mu0=mu, lambda0=lam)).p.weights
            expected = pycma_weights(mu, lam, d, w.c1, w.cmu)
            np.testing.assert_allclose(w.weights, expected, atol=1e-14, err_msg=str((mu, lam)))

    def test_negative_weights_are_not_positive(self):
        # lambda 10, mu 6: the four "negative" weights were all positive
        for lam in range(2, 31):
            for mu in range(1, lam + 1):
                w = modcma.ModularCMAES(modcma.Settings(5, mu0=mu, lambda0=lam)).p.weights
                self.assertTrue(np.all(w.positive > 0), (mu, lam))
                self.assertTrue(np.all(w.negative <= 0), (mu, lam))
                self.assertAlmostEqual(float(np.sum(w.positive)), 1.0, places=14)

    def test_the_default_mu_is_unchanged(self):
        # mu = lambda // 2: ln((lambda + 1) / 2) - ln i (Hansen 2016, Eq. (49)), as before
        for lam in (4, 6, 7, 10, 15):
            w = modcma.ModularCMAES(modcma.Settings(10, lambda0=lam)).p.weights
            raw = math.log((lam + 1) / 2) - np.log(np.arange(1, lam // 2 + 1))
            np.testing.assert_allclose(w.positive, raw / raw.sum(), atol=1e-15)


class TestSeparableRates(unittest.TestCase):
    def test_c1_plus_cmu_at_most_one(self):
        # cmu * (d + 2) / 3 is capped at 1 - c1 after the multiplication (#73, F4): every lambda
        # that IPOP reaches (up to 2^9 lambda0)
        for d in (2, 10, 100):
            lam0 = 4 + int(3 * math.log(d))
            for k in range(10):
                w = modcma.ModularCMAES(
                    modcma.settings_from_dict(d, matrix_adaptation="SEPARABLE", lambda0=lam0 * 2**k)
                ).p.weights
                self.assertLessEqual(w.c1 + w.cmu, 1.0 + 1e-15, (d, k))
                self.assertGreater(w.cmu, 0)

    def test_small_populations_unchanged(self):
        # where the cap does not bind, cmu is the full-covariance rate times (d + 2) / 3
        d = 10
        full = modcma.ModularCMAES(modcma.Settings(d)).p.weights
        sep = modcma.ModularCMAES(
            modcma.settings_from_dict(d, matrix_adaptation="SEPARABLE")
        ).p.weights
        self.assertAlmostEqual(sep.cmu, full.cmu * (d + 2) / 3, places=15)
        self.assertEqual(sep.c1, full.c1)


if __name__ == "__main__":
    unittest.main()
