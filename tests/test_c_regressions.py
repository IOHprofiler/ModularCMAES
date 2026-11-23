"""Regression tests for the bugs fixed in WP0 (TASKS.md, section 1 and T0.3)."""

import unittest

import numpy as np

from modcma import c_maes


def sphere(x):
    return float(np.dot(x, x))


class TestIntegerMapping(unittest.TestCase):
    """B1: integer variables were mapped by loop index instead of by the given index."""

    def test_noncontiguous_integer_indices(self):
        c_maes.utils.set_seed(1)
        settings = c_maes.Settings(5, integer_variables=[1, 3], verbose=False)
        cma = c_maes.ModularCMAES(settings)
        cma.step(sphere)
        X, Xt = cma.p.pop.X, cma.p.pop.X_transformed
        for i in (1, 3):
            np.testing.assert_array_equal(Xt[i], np.round(X[i]))
        for i in (0, 2, 4):
            np.testing.assert_array_equal(Xt[i], X[i])
        # the continuous coordinates must actually be continuous in this sample
        self.assertFalse(np.all(Xt[[0, 2, 4]] == np.round(Xt[[0, 2, 4]])))

    def test_out_of_range_index_raises(self):
        settings = c_maes.Settings(3, integer_variables=[3], verbose=False)
        with self.assertRaises(ValueError):
            c_maes.ModularCMAES(settings)


class TestActiveCovarianceUpdate(unittest.TestCase):
    """B2: the negative weights of active CMA were never rescaled by d / ||C^-1/2 y||^2 when
    lambda = 2 mu (the default), because the loop ran over [mu, lambda - mu)."""

    def test_rank_mu_update_matches_reference(self):
        c_maes.utils.set_seed(3)
        modules = c_maes.Modules()
        modules.active = True
        d = 5
        settings = c_maes.Settings(d, modules, lambda0=10, mu0=5, verbose=False)
        cma = c_maes.ModularCMAES(settings)
        cma.step(sphere)  # the first generation does not update C (lazy update)

        cma.mutate(sphere)
        cma.select()
        cma.recombine()
        p, a = cma.p, cma.p.adaptation
        a.adapt_evolution_paths(p.pop, p.weights, p.stats, p.settings, p.lamb, p.mu)
        C0, pc, inv_root_C = a.C.copy(), a.pc.copy(), a.inv_root_C.copy()
        Y, hs = p.pop.Y.copy(), a.hs
        w = np.array(p.weights.weights[: Y.shape[1]])
        c1, cmu, cc = p.weights.c1, p.weights.cmu, p.weights.cc
        self.assertTrue(a.adapt_matrix(p.weights, p.settings.modules, p.pop, p.mu, p.settings, p.stats))

        rank_mu_w = w.copy()
        for i in range(p.mu, len(w)):
            rank_mu_w[i] *= d / np.sum((inv_root_C @ Y[:, i]) ** 2)
        dhs = (1 - hs) * cc * (2 - cc)
        decay = 1 - c1 * dhs - c1 - cmu * w.sum()
        expected = decay * C0 + c1 * np.outer(pc, pc) + cmu * (Y * rank_mu_w) @ Y.T
        expected = 0.5 * (expected + expected.T)
        np.testing.assert_allclose(a.C, expected, rtol=1e-12, atol=1e-14)
        # and the rescaling made a difference for this sample
        unscaled = decay * C0 + c1 * np.outer(pc, pc) + cmu * (Y * w) @ Y.T
        self.assertGreater(np.abs(a.C - 0.5 * (unscaled + unscaled.T)).max(), 1e-6)


class TestNaturalGradientInverse(unittest.TestCase):
    """B3: NaturalGradientAdaptation::invert_y recomputed the pseudo-inverse on every call."""

    def test_inverse_is_cached_and_correct(self):
        c_maes.utils.set_seed(1)
        modules = c_maes.Modules()
        modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.NATURAL_GRADIENT
        cma = c_maes.ModularCMAES(c_maes.Settings(4, modules, verbose=False))
        for _ in range(5):
            cma.step(sphere)
        a = cma.p.adaptation
        y = np.arange(1.0, 5.0)
        z = a.invert_y(y)
        np.testing.assert_allclose(np.array(a.A) @ z, y, rtol=1e-10)
        # after the first call, the cached inverse is used: changing A without a matrix update
        # must not change the result
        a.A = np.array(a.A) * 2.0
        np.testing.assert_array_equal(a.invert_y(y), z)


if __name__ == "__main__":
    unittest.main()
