import math
import unittest

import numpy as np

import modcma
from modcma.c_maes import parameters, Parameters, Population, ModularCMAES, options


class TestMutation(unittest.TestCase):

    def get_cma(self, modules, adapt_sigma=True):
        settings = parameters.Settings(2, modules)
        par = Parameters(settings)

        cma = ModularCMAES(par)
        if adapt_sigma:
            cma.mutate(sum)
            cma.select()
            cma.recombine()
        return cma

    def test_matrix_adaptation(self):
        modules = parameters.Modules()
        modules.matrix_adaptation = options.MatrixAdaptationType.MATRIX
        cma = self.get_cma(modules)

        M = cma.p.adaptation.M.copy()
        z = np.sum(
            cma.p.weights.positive * cma.p.pop.Z[:, : cma.p.mu], axis=1, keepdims=True
        )
        ps = (
            (1.0 - cma.p.weights.cs) * cma.p.adaptation.ps
            + (
                np.sqrt(
                    cma.p.weights.cs * (2.0 - cma.p.weights.cs) * cma.p.weights.mueff
                )
                * z.ravel()
            )
        ).reshape(-1, 1)
        old_M = (1 - 0.5 * cma.p.weights.c1 - 0.5 * cma.p.weights.cmu) * M
        scaled_ps = (0.5 * cma.p.weights.c1) * M.dot(ps).dot(ps.T)
        new_M = (
            (0.5 * cma.p.weights.cmu * cma.p.weights.positive)
            * cma.p.pop.Y[:, : cma.p.mu]
        ).dot(cma.p.pop.Z[:, : cma.p.mu].T)
        M = old_M + scaled_ps + new_M

        cma.adapt()
        self.assertTrue(np.all(np.isclose(cma.p.adaptation.M, M)))


def ellipsoid(x):
    x = np.asarray(x)
    return float(np.sum(10 ** (6 * np.arange(len(x)) / (len(x) - 1)) * x**2))


class TestActiveUpdate(unittest.TestCase):
    """The active update of MATRIX, CHOLESKY and SEPARABLE as COVARIANCE's (Hansen 2016, Eq. (46),
    (47)): the decay with the sum of all weights, the negative weights rescaled by
    d / ||A^-1 y||^2 (#73, F7)."""

    @staticmethod
    def until_the_matrix_update(matrix_adaptation, generations=8, **kwargs):
        settings = modcma.settings_from_dict(
            5, matrix_adaptation=matrix_adaptation, active=True, seed=2, budget=10**5, **kwargs
        )
        es = ModularCMAES(settings)
        for _ in range(generations):
            es.step(ellipsoid)
        es.mutate(ellipsoid)
        es.select()
        es.recombine()
        p = es.p
        p.adaptation.adapt_evolution_paths(p.pop, p.weights, p.stats, p.settings, p.mu)
        return es

    @staticmethod
    def rescaled(w, norms2, d):
        w = np.array(w, dtype=float)
        n_positive = np.argmax(w < 0) if np.any(w < 0) else len(w)
        w[n_positive:] *= d / norms2[n_positive:]
        return w

    def test_cholesky(self):
        es = self.until_the_matrix_update("CHOLESKY")
        p, a = es.p, es.p.adaptation
        A, Y = np.tril(a.A), np.asarray(p.pop.Y)[:, : p.pop.n]
        w = np.asarray(p.weights.for_population(p.pop.n))
        self.assertTrue(np.any(w < 0))
        Z = np.linalg.solve(A, Y)
        w_resc = self.rescaled(w, np.sum(Z**2, axis=0), 5)
        c1, cmu = p.weights.c1, p.weights.cmu
        expected = (1 - c1 - cmu * w.sum()) * A @ A.T + c1 * np.outer(a.pc, a.pc)
        expected += cmu * (Y * w_resc) @ Y.T
        self.assertTrue(a.adapt_matrix(p.weights, p.pop, p.mu, p.settings, p.stats))
        L = np.tril(a.A)
        np.testing.assert_allclose(L @ L.T, expected, rtol=1e-10, atol=1e-14)
        self.assertEqual(a.skipped_downdates, 0)

    def test_matrix(self):
        # M <- M ((1 - c1/2 - cmu/2 sum(w)) I + c1/2 ps ps^T + cmu/2 sum w°_i z_i z_i^T)
        es = self.until_the_matrix_update("MATRIX")
        p, a = es.p, es.p.adaptation
        M, Z = np.array(a.M), np.asarray(p.pop.Z)[:, : p.pop.n]
        w = np.asarray(p.weights.for_population(p.pop.n))
        w_resc = self.rescaled(w, np.sum(Z**2, axis=0), 5)
        c1, cmu = p.weights.c1, p.weights.cmu
        inner = (1 - c1 / 2 - cmu / 2 * w.sum()) * np.eye(5) + c1 / 2 * np.outer(a.ps, a.ps)
        inner += cmu / 2 * (Z * w_resc) @ Z.T
        a.adapt_matrix(p.weights, p.pop, p.mu, p.settings, p.stats)
        np.testing.assert_allclose(a.M, M @ inner, rtol=1e-10, atol=1e-14)

    def test_separable(self):
        es = self.until_the_matrix_update("SEPARABLE")
        p, a = es.p, es.p.adaptation
        c, Y = np.array(a.c), np.asarray(p.pop.Y)[:, : p.pop.n]
        w = np.asarray(p.weights.for_population(p.pop.n))
        w_resc = self.rescaled(w, np.sum(Y**2 / c[:, None], axis=0), 5)
        wt = p.weights
        ps_length = np.linalg.norm(a.ps) / math.sqrt(
            1 - (1 - wt.cs_path) ** (2 * (p.stats.t_run + 1))
        )
        dhs = 0.0 if ps_length < wt.expected_length_ps else wt.cc * (2 - wt.cc)
        decay = 1 + wt.c1 * dhs - wt.c1 - wt.cmu * w.sum()
        expected = decay * c + wt.c1 * a.pc**2 + wt.cmu * (Y**2) @ w_resc
        a.adapt_matrix(p.weights, p.pop, p.mu, p.settings, p.stats)
        np.testing.assert_allclose(a.c, np.maximum(expected, 1e-12), rtol=1e-12)

    def test_active_is_not_worse_on_the_ellipsoid(self):
        # f10 of BBOB is an ellipsoid; median evaluations to 1e-8, d = 10, 5 runs
        for ma in ("MATRIX", "CHOLESKY", "SEPARABLE"):
            evals = {}
            for active in (False, True):
                runs = []
                for seed in range(5):
                    settings = modcma.settings_from_dict(
                        10, matrix_adaptation=ma, active=active, seed=seed, budget=60_000,
                        target=1e-8,
                    )
                    es = ModularCMAES(settings)
                    es.run(ellipsoid)
                    self.assertLess(es.p.stats.global_best.y, 1e-8, (ma, active, seed))
                    runs.append(es.p.stats.evaluations)
                evals[active] = np.median(runs)
            self.assertLessEqual(evals[True], evals[False], ma)

    def test_cmsa_switches_active_off(self):
        # CMSA has no active variant: its update made C indefinite (#73, F6)
        for weights in ("DEFAULT", "UTILITY"):
            settings = modcma.settings_from_dict(
                5, matrix_adaptation="CMSA", active=True, weights=weights
            )
            self.assertFalse(settings.modules.active, weights)


class TestFactorUpdates(unittest.TestCase):
    """#74: a failing Cholesky downdate (F5) and the inverse of M with elitism (F18)."""

    def test_a_failing_downdate_is_skipped(self):
        es = TestActiveUpdate.until_the_matrix_update("CHOLESKY")
        p, a = es.p, es.p.adaptation
        A, Y = np.tril(a.A), np.asarray(p.pop.Y)[:, : p.pop.n]
        # a negative weight so large that its downdate cannot keep C positive definite
        w = np.array(p.weights.weights)
        w[-1] = -1e3
        p.weights.weights = w
        Z = np.linalg.solve(A, Y)
        w_resc = TestActiveUpdate.rescaled(w, np.sum(Z**2, axis=0), 5)
        c1, cmu = p.weights.c1, p.weights.cmu
        expected = (1 - c1 - cmu * w.sum()) * A @ A.T + c1 * np.outer(a.pc, a.pc)
        expected += cmu * (Y[:, :-1] * w_resc[:-1]) @ Y[:, :-1].T  # without the last term
        before = a.skipped_downdates
        self.assertTrue(a.adapt_matrix(p.weights, p.pop, p.mu, p.settings, p.stats))
        self.assertEqual(a.skipped_downdates, before + 1)
        L = np.tril(a.A)
        self.assertTrue(np.all(np.isfinite(L)) and np.all(np.diag(L) > 0))
        np.testing.assert_allclose(L @ L.T, expected, rtol=1e-10, atol=1e-14)

    def test_the_inverse_of_m_with_elitism(self):
        # M_inv was updated incrementally and drifted (0.98 relative error after 60 generations)
        solved = 0
        for seed in range(5):
            settings = modcma.settings_from_dict(
                10, matrix_adaptation="MATRIX", elitist=True, seed=seed, budget=50_000,
                target=1e-8,
            )
            es = ModularCMAES(settings)
            for _ in range(200):
                es.step(ellipsoid)
            a = es.p.adaptation
            a.invert_y(np.ones(10))  # brings M_inv up to date
            self.assertLess(np.linalg.norm(a.M @ a.M_inv - np.eye(10)), 1e-8)
            es.run(ellipsoid)
            solved += es.p.stats.global_best.y < 1e-8
        self.assertEqual(solved, 5)


if __name__ == "__main__":
    unittest.main()
