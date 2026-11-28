"""Elitism with the active update (#161), with XNES (#172) and with TPA (#179).

#161: with elitism the population is the lambda best of the (mu + lambda) pool; the negative weights
of the matrix update go to the worst points of the pool, which the selection dropped.
#172: XNES's step-size update reads the offspring alone, ranked among themselves.
#179: a generation whose selection kept every parent is a failure of TPA and TPA_INPLACE: in the
next generation TPA evaluates no test points (r = -a_tpa), TPA_INPLACE samples no test pair (z = -1).
"""

import math
import unittest

import numpy as np

from modcma import c_maes
from modcma.c_maes import Population

ADAPTATIONS = {
    "COVARIANCE": ("CovarianceAdaptation", "C"),
    "COVARIANCE_NO_EIGV": ("CovarianceNoEigvAdaptation", "C"),
    "MATRIX": ("MatrixAdaptation", "M"),
    "CHOLESKY": ("CholeskyAdaptation", "A"),
    "SEPARABLE": ("SeparableAdaptation", "c"),
    "NATURAL_GRADIENT": ("NaturalGradientAdaptation", "A"),
}


def sphere(x):
    return float(np.sum((x - 0.5) ** 2))


def ellipsoid(x):
    d = len(x)
    return float(np.sum(10.0 ** (6.0 * np.arange(d) / (d - 1)) * x * x))


class TestActiveWithElitism(unittest.TestCase):
    """The matrix update with a pool equals the update of a population whose negatively weighted
    columns are the pool's last ones."""

    def pool(self, d, lam, mu, seed):
        rng = np.random.default_rng(seed)
        n = lam + mu
        Z = rng.normal(size=(d, n))
        f = np.sort(rng.normal(size=n))
        return Z, f

    @staticmethod
    def adaptation(cls, d):
        # m_old is set by the recombination, which these tests skip: the mean did not move
        a = getattr(c_maes.matrix_adaptation, cls)(d, np.zeros(d))
        a.m_old = np.zeros(d)
        return a

    def population(self, Z, f, columns):
        Z, f = Z[:, columns], f[columns]
        return Population(Z.copy(), Z.copy(), Z.copy(), f.copy(), np.ones_like(Z))

    def test_every_adaptation(self):
        d = 6
        for matrix, (cls, attribute) in ADAPTATIONS.items():
            with self.subTest(matrix=matrix):
                settings = c_maes.settings_from_dict(
                    d, matrix_adaptation=matrix, elitist=True, active=True,
                    always_compute_eigv=True,
                )
                es = c_maes.ModularCMAES(settings)
                p = es.p
                lam, mu = p.lamb, p.mu
                Z, f = self.pool(d, lam, mu, 1)
                # the sorted pool, of which the selection keeps the first lambda
                pooled = self.population(Z, f, list(range(lam + mu)))
                pooled.n_pool = lam + mu
                pooled.resize_cols(lam)
                self.assertEqual(pooled.n, lam)
                # the same steps: the mu best, then the lambda - mu worst of the pool
                n_positive = int(np.sum(p.weights.weights >= 0))
                tail = list(range(lam + mu - (lam - n_positive), lam + mu))
                gathered = self.population(Z, f, list(range(n_positive)) + tail)

                results = []
                for pop in (pooled, gathered):
                    a = self.adaptation(cls, d)
                    a.adapt_evolution_paths(pop, p.weights, p.stats, p.settings, mu)
                    self.assertTrue(a.adapt_matrix(p.weights, pop, mu, p.settings, p.stats))
                    results.append(np.array(getattr(a, attribute)))
                np.testing.assert_array_equal(results[0], results[1])

                # without a pool the negative weights go to ranks mu..lambda-1, as before
                plain = self.population(Z, f, list(range(lam)))
                a = self.adaptation(cls, d)
                a.adapt_evolution_paths(plain, p.weights, p.stats, p.settings, mu)
                a.adapt_matrix(p.weights, plain, mu, p.settings, p.stats)
                self.assertGreater(np.abs(np.array(getattr(a, attribute)) - results[0]).max(), 0)

    def test_covariance_update(self):
        # C = (1 - c1 - cmu sum(w)) C + cmu sum_i w°_i y_i y_i^T from C = I and pc = 0, the
        # negative weights (rescaled by d / ||y||^2) on the worst points of the pool
        d = 5
        es = c_maes.ModularCMAES(
            c_maes.settings_from_dict(d, elitist=True, active=True, always_compute_eigv=True)
        )
        p = es.p
        lam, mu = p.lamb, p.mu
        Z, f = self.pool(d, lam, mu, 2)
        pop = self.population(Z, f, list(range(lam + mu)))
        pop.n_pool = lam + mu
        pop.resize_cols(lam)
        a = c_maes.matrix_adaptation.CovarianceAdaptation(d, np.zeros(d))
        a.adapt_covariance_matrix(p.weights, p.settings.modules, pop, mu)
        w = np.array(p.weights.weights)
        Y = np.hstack([Z[:, :mu], Z[:, lam + mu - (lam - mu):]])
        scaled = w.copy()
        scaled[mu:] *= d / np.sum(Y[:, mu:] ** 2, axis=0)
        expected = (1 - p.weights.c1 - p.weights.cmu * w.sum()) * np.eye(d)
        expected += p.weights.cmu * (Y * scaled) @ Y.T
        np.testing.assert_allclose(a.C, expected, rtol=1e-12, atol=1e-14)

    def test_the_selection_keeps_the_pool(self):
        es = c_maes.ModularCMAES(c_maes.settings_from_dict(5, elitist=True, active=True, seed=1))
        for _ in range(3):
            es.step(sphere)
        es.mutate(sphere)
        es.select()
        self.assertEqual(es.p.pop.n, es.p.lamb)
        self.assertEqual(es.p.pop.n_pool, es.p.lamb + es.p.mu)
        es.mutate(sphere)  # a new population: no pool
        self.assertEqual(es.p.pop.n_pool, 0)

    def test_solves_the_rotated_ellipsoid(self):
        # 0 of 25 runs on BBOB f10 at d = 10 before (#161)
        d = 10
        rotation, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(d, d)))
        weights = 10.0 ** (6.0 * np.arange(d) / (d - 1))
        es = c_maes.ModularCMAES(c_maes.settings_from_dict(
            d, elitist=True, active=True, seed=2, budget=20000, target=1e-8, sigma0=2.0,
            x0=np.full(d, 3.0),
        ))
        es.run(lambda x: float(np.sum(weights * (rotation @ x) ** 2)))
        self.assertLessEqual(es.p.stats.global_best.y, 1e-8)


class TestXnesWithElitism(unittest.TestCase):
    def offspring_statistic(self, es, natural_gradient):
        p = es.p
        order = np.argsort(np.array(p.pop.f), kind="stable")
        Z = np.array(p.pop.Z)[:, order]
        d = p.settings.dim
        w = np.array(p.weights.positive)
        k = min(len(w), Z.shape[1])
        squared = np.sum(Z[:, :k] ** 2, axis=0)
        if natural_gradient:
            return (w[:k] @ squared - w[:k].sum() * d) / d
        return w[:k] @ (squared - d)

    def test_sigma_from_the_offspring(self):
        for natural_gradient in (False, True):
            with self.subTest(natural_gradient=natural_gradient):
                matrix = "NATURAL_GRADIENT" if natural_gradient else "COVARIANCE"
                es = c_maes.ModularCMAES(c_maes.settings_from_dict(
                    6, ssa="XNES", elitist=True, matrix_adaptation=matrix, seed=4,
                ))
                parents_selected = False
                for _ in range(12):
                    es.mutate(ellipsoid)
                    statistic = self.offspring_statistic(es, natural_gradient)
                    sigma, cs, d = es.p.mutation.sigma, es.p.weights.cs, es.p.settings.dim
                    t = es.p.stats.t
                    es.select()
                    parents_selected |= bool(np.any(np.array(es.p.pop.t) < t))
                    es.recombine()
                    es.adapt()
                    if natural_gradient:
                        expected = sigma * math.exp(cs / 2 * statistic)
                    else:
                        expected = sigma * math.exp(cs / math.sqrt(d) * statistic)
                    self.assertAlmostEqual(es.p.mutation.sigma / expected, 1.0, places=12)
                self.assertTrue(parents_selected)

    def test_without_elitism_unchanged(self):
        # the offspring are the selected points: sigma from the population after selection
        es = c_maes.ModularCMAES(c_maes.settings_from_dict(6, ssa="XNES", seed=4))
        for _ in range(3):
            es.step(ellipsoid)
        es.mutate(ellipsoid)
        es.select()
        es.recombine()
        p = es.p
        sigma, w, d = p.mutation.sigma, np.array(p.weights.positive), p.settings.dim
        Z = np.array(p.pop.Z)[:, : len(w)]
        expected = sigma * math.exp(p.weights.cs / math.sqrt(d) * (w @ (np.sum(Z**2, 0) - d)))
        es.adapt()
        self.assertAlmostEqual(p.mutation.sigma / expected, 1.0, places=12)

    def test_solves_the_ellipsoid(self):
        # 0 of 20 runs at d = 10 before (#172)
        for matrix in ("COVARIANCE", "NATURAL_GRADIENT"):
            with self.subTest(matrix=matrix):
                es = c_maes.ModularCMAES(c_maes.settings_from_dict(
                    10, ssa="XNES", elitist=True, matrix_adaptation=matrix, seed=1,
                    budget=100000, target=1e-8, x0=np.full(10, 2.0),
                ))
                es.run(ellipsoid)
                self.assertLessEqual(es.p.stats.global_best.y, 1e-8)


def kept_every_parent(es):
    """Whether the selection of the generation that just ended kept every parent: none of the mu
    selected points was sampled in that generation (stats.t - 1, after adapt counted it)."""
    t = np.array(es.p.pop.t)[: es.p.mu]
    return bool(np.all(t < es.p.stats.t - 1))


class TestTpaWithElitism(unittest.TestCase):
    def settings(self, d, ssa, seed, elitist=True):
        # The setup of #179: x0 uniform in [-3, 3]^d, sigma0 = 2, 10^4 d evaluations
        x0 = np.random.default_rng(seed + 1000).uniform(-3.0, 3.0, d)
        return c_maes.settings_from_dict(
            d, ssa=ssa, elitist=elitist, seed=seed, x0=x0, sigma0=2.0, budget=10000 * d,
            target=1e-8,
        )

    def test_solves_the_ellipsoid(self):
        # Before #179, with elitist: TPA solved 0 of 20 runs at d = 10 and 20 (sigma diverged),
        # TPA_INPLACE 11 and 0
        for ssa in ("TPA", "TPA_INPLACE"):
            for d in (10, 20):
                for seed in (0, 1):
                    with self.subTest(ssa=ssa, d=d, seed=seed):
                        es = c_maes.ModularCMAES(self.settings(d, ssa, seed))
                        es.run(ellipsoid)
                        self.assertLessEqual(es.p.stats.global_best.y, 1e-8)

    def test_no_test_points_after_a_stuck_generation(self):
        # ask() returns the population and TPA's two points, but only the population after a
        # generation whose selection kept every parent; r is then a failure, -a_tpa
        es = c_maes.ModularCMAES(self.settings(5, "TPA", 2))
        lam, a_tpa = es.p.lamb, es.p.mutation.a_tpa
        stuck = False
        counts = {True: 0, False: 0}
        for _ in range(80):
            X = es.ask()
            self.assertEqual(len(X), lam if stuck else lam + 2)
            evaluations = es.p.stats.evaluations
            self.assertTrue(es.tell(np.array([ellipsoid(x) for x in X])))
            self.assertEqual(es.p.stats.evaluations - evaluations, len(X))
            if stuck:
                self.assertEqual(es.p.mutation.rank_tpa, -a_tpa)
            counts[stuck] += 1
            stuck = kept_every_parent(es)
        # both kinds of generation occurred
        self.assertGreater(counts[True], 5)
        self.assertGreater(counts[False], 5)

    def test_no_test_pair_after_a_stuck_generation(self):
        # TPA_INPLACE's first two offspring are a mirrored pair along the mean shift, except after
        # a generation whose selection kept every parent: then ordinary samples, and z = -1
        es = c_maes.ModularCMAES(self.settings(5, "TPA_INPLACE", 2))
        stuck = False
        counts = {True: 0, False: 0}
        for generation in range(80):
            X = es.ask()
            self.assertEqual(len(X), es.p.lamb)
            Z = np.array(es.p.pop.Z)
            if generation > 0:
                self.assertEqual(np.array_equal(Z[:, 1], -Z[:, 0]), not stuck)
            self.assertTrue(es.tell(np.array([ellipsoid(x) for x in X])))
            if stuck:
                self.assertEqual(es.p.mutation.rank_difference, -1.0)
            counts[stuck] += 1
            stuck = kept_every_parent(es)
        self.assertGreater(counts[True], 5)
        self.assertGreater(counts[False], 5)

    def test_without_elitism_every_generation_has_test_points(self):
        for ssa in ("TPA", "TPA_INPLACE"):
            with self.subTest(ssa=ssa):
                es = c_maes.ModularCMAES(self.settings(5, ssa, 2, elitist=False))
                lam = es.p.lamb
                for generation in range(40):
                    X = es.ask()
                    if ssa == "TPA":
                        self.assertEqual(len(X), lam + 2)
                    elif generation > 0:
                        Z = np.array(es.p.pop.Z)
                        np.testing.assert_array_equal(Z[:, 1], -Z[:, 0])
                    es.tell(np.array([ellipsoid(x) for x in X]))


if __name__ == "__main__":
    unittest.main()
