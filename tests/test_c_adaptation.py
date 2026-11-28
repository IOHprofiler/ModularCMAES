import math
import unittest

import numpy as np

import modcma
from modcma.c_maes import parameters, Parameters, ModularCMAES, options


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


class TestRepr(unittest.TestCase):
    def test_every_adaptation_names_its_own_class(self):
        # the classes without a repr of their own inherit Adaptation's (CovarianceAdaptation's for
        # COVARIANCE_NO_EIGV), which named the base class
        for option in options.MatrixAdaptationType.__members__.values():
            with self.subTest(option=option.name):
                modules = parameters.Modules()
                modules.matrix_adaptation = option
                adaptation = Parameters(parameters.Settings(3, modules)).adaptation
                text = repr(adaptation)
                self.assertTrue(text.startswith(f"<{type(adaptation).__name__} m: "), text)

    def test_the_repr_of_another_object_is_a_type_error(self):
        from modcma.c_maes import matrix_adaptation

        matrix = matrix_adaptation.MatrixAdaptation(2, np.zeros(2))
        with self.assertRaises(TypeError):
            matrix_adaptation.Adaptation.__repr__(5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            matrix_adaptation.CovarianceAdaptation.__repr__(matrix)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):  # not initialized
            repr(matrix_adaptation.CholeskyAdaptation.__new__(matrix_adaptation.CholeskyAdaptation))


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
            # kept with M by the Woodbury update (no invert_y needed)
            self.assertLess(np.linalg.norm(a.M @ a.M_inv - np.eye(10)), 1e-8)
            self.assertEqual(a.inverse_fallbacks, 0)
            es.run(ellipsoid)
            solved += es.p.stats.global_best.y < 1e-8
        self.assertEqual(solved, 5)

    def test_the_woodbury_update_with_active_and_its_recomputation(self):
        # With active at d = 10 the update's rounding errors grow (above 1e-6 after 1500
        # generations without a recomputation); M_inv is computed from M every 400 generations
        A = modcma.matrix_adaptation.MatrixAdaptation
        self.assertEqual((A.inverse_recompute_interval, A.inverse_recompute_max_dim), (400, 30))
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))

        def rotated(X):
            return np.sum(10 ** (6 * np.arange(10) / 9) * (np.asarray(X) @ Q.T) ** 2, axis=1)

        settings = modcma.settings_from_dict(
            10, matrix_adaptation="MATRIX", elitist=True, active=True, seed=1, budget=10**8
        )
        es = ModularCMAES(settings)
        a = es.p.adaptation
        worst = 0.0
        for g in range(1, 2001):
            es.step(rotated, vectorized=True)
            if g % 50 == 0 or g == 400:
                worst = max(worst, float(np.max(np.abs(a.M @ a.M_inv - np.eye(10)))))
            if g == 400:
                # just recomputed
                self.assertLess(np.max(np.abs(a.M @ a.M_inv - np.eye(10))), 1e-12)
        self.assertLess(worst, 1e-8)


class TestHsigma(unittest.TestCase):
    """#160: MATRIX and CHOLESKY stall the rank-one update by h_sigma (Hansen 2016, after
    Eq. (45)), as COVARIANCE does, with every step-size rule but CSA."""

    @staticmethod
    def before_the_update(matrix, ssa, long_path):
        settings = modcma.settings_from_dict(5, matrix_adaptation=matrix, ssa=ssa, seed=2)
        es = ModularCMAES(settings)
        for _ in range(3):
            es.step(ellipsoid)
        es.mutate(ellipsoid)
        es.select()
        es.recombine()
        if long_path:
            es.p.adaptation.ps = np.full(5, 10.0)
        return es

    def test_matrix(self):
        for ssa, long_path, hs in (("XNES", True, False), ("XNES", False, True),
                                   ("SA", True, False), ("CSA", True, True)):
            with self.subTest(ssa=ssa, long_path=long_path):
                es = self.before_the_update("MATRIX", ssa, long_path)
                p, a = es.p, es.p.adaptation
                M = np.array(a.M)
                a.adapt_evolution_paths(p.pop, p.weights, p.stats, p.settings, p.mu)
                self.assertEqual(a.hs, hs)
                ps, w = np.array(a.ps), np.array(p.weights.positive)
                Z, Y = np.array(p.pop.Z)[:, : p.mu], np.array(p.pop.Y)[:, : p.mu]
                tau_1 = p.weights.c1 / 2 if hs else 0.0  # no rank-one term when h_sigma = 0
                tau_m = p.weights.cmu / 2
                expected = (1 - tau_1 - tau_m) * M + tau_1 * np.outer(M @ ps, ps)
                expected += (Y * (tau_m * w)) @ Z.T
                a.adapt_matrix(p.weights, p.pop, p.mu, p.settings, p.stats)
                np.testing.assert_allclose(a.M, expected, rtol=1e-12, atol=1e-14)

    def test_cholesky(self):
        for ssa, long_path, hs in (("XNES", True, False), ("SA", True, False),
                                   ("SA", False, True), ("CSA", True, True)):
            with self.subTest(ssa=ssa, long_path=long_path):
                es = self.before_the_update("CHOLESKY", ssa, long_path)
                p, a = es.p, es.p.adaptation
                pc, A = np.array(a.pc), np.tril(a.A)
                a.adapt_evolution_paths(p.pop, p.weights, p.stats, p.settings, p.mu)
                self.assertEqual(a.hs, hs)
                cc = p.weights.cc
                # pc stalled when h_sigma = 0; the decay compensates the variance it loses
                step = p.weights.sqrt_cc_mueff * np.array(a.dm) if hs else 0.0
                expected_pc = (1 - cc) * pc + step
                np.testing.assert_allclose(a.pc, expected_pc, rtol=1e-12, atol=1e-15)
                c1, cmu = p.weights.c1, p.weights.cmu
                dhs = 0.0 if hs else cc * (2 - cc)
                w, Y = np.array(p.weights.positive), np.array(p.pop.Y)[:, : p.mu]
                expected = (1 + c1 * dhs - c1 - cmu) * A @ A.T + c1 * np.outer(a.pc, a.pc)
                expected += cmu * (Y * w) @ Y.T
                a.adapt_matrix(p.weights, p.pop, p.mu, p.settings, p.stats)
                L = np.tril(a.A)
                np.testing.assert_allclose(L @ L.T, expected, rtol=1e-10, atol=1e-14)

    def test_matrix_with_xnes_solves_the_ellipsoid(self):
        # 0 of 20 runs at d = 10 before #160
        settings = modcma.settings_from_dict(
            10, matrix_adaptation="MATRIX", ssa="XNES", seed=1, budget=100_000, target=1e-8,
            x0=np.full(10, 2.0),
        )
        es = ModularCMAES(settings)
        es.run(ellipsoid)
        self.assertLessEqual(es.p.stats.global_best.y, 1e-8)


if __name__ == "__main__":
    unittest.main()
