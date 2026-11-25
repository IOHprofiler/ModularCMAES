"""The deprecated names of 1.x (#82): each still works and warns once, at the caller's line."""

import unittest
import warnings

import numpy as np

from modcma import c_maes
from modcma.c_maes import mutation, options, repelling, selection

ma = c_maes.cmaescpp.matrix_adaptation


class TestDeprecatedNames(unittest.TestCase):
    def assert_warns_here(self, call):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = call()
        self.assertEqual(len(caught), 1, [str(w.message) for w in caught])
        self.assertIs(caught[0].category, DeprecationWarning)
        self.assertEqual(caught[0].filename, __file__)  # points at the caller
        return result

    def test_classes(self):
        self.assertIs(self.assert_warns_here(lambda: selection.Elitsm), selection.Elitism)
        self.assertIs(self.assert_warns_here(lambda: selection.NoElitsm), selection.NoElitism)
        with self.assertRaises(AttributeError):
            selection.DoesNotExist  # noqa: B018
        self.assertFalse(hasattr(selection, "DoesNotExist"))

    def test_functions(self):
        u, v = np.array([0.0, 3.0]), np.array([4.0, 0.0])
        self.assertEqual(self.assert_warns_here(lambda: repelling.euclidian(u, v)), 5.0)
        self.assertEqual(
            self.assert_warns_here(lambda: repelling.mahanolobis(u, v, np.eye(2))), 5.0)

    def test_method(self):
        es = c_maes.ModularCMAES(c_maes.Settings(3, verbose=False, seed=1))
        es.step(lambda x: float(np.sum(x**2)))
        p = es.p
        self.assert_warns_here(lambda: p.adaptation.adapt_evolution_paths_innner(
            p.pop, p.weights, p.stats, p.settings, p.mu, p.lamb))

    def test_keyword(self):
        s = self.assert_warns_here(
            lambda: mutation.SequentialSelection(options.Mirror.NONE, seq_cuttoff_factor=2.0))
        self.assertIsInstance(s, mutation.SequentialSelection)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            mutation.SequentialSelection(options.Mirror.NONE, seq_cutoff_factor=2.0)
            mutation.SequentialSelection(options.Mirror.NONE, 2.0)
            mutation.NoSequentialSelection(options.Mirror.NONE, seq_cutoff_factor=2.0)
        self.assert_warns_here(
            lambda: mutation.NoSequentialSelection(options.Mirror.NONE, seq_cuttoff_factor=2.0))

    def test_properties(self):
        r = repelling.AdaptiveRepelling()
        self.assert_warns_here(lambda: setattr(r, "target_rejection_rate", 0.3))
        self.assertEqual(self.assert_warns_here(lambda: r.local_target_rejection_rate), 0.2)
        s = selection.Strategy(c_maes.Modules())
        self.assertIsInstance(self.assert_warns_here(lambda: s.elitsm), selection.NoElitism)

    def test_warnings_as_errors(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with self.assertRaises(DeprecationWarning):
                repelling.euclidian(np.zeros(2), np.ones(2))


class TestConstructors(unittest.TestCase):
    def test_step_size_adaptations(self):
        args = (mutation.NoThresholdConvergence(),
                mutation.NoSequentialSelection(options.Mirror.NONE),
                mutation.NoSigmaSampler(3), 0.5)
        for name in ("CSA", "TPA", "MSR", "PSR", "XNES", "MXNES", "LPXNES", "SR", "SA"):
            strategy = getattr(mutation, name)(*args)
            self.assertEqual(strategy.sigma, 0.5)

    def test_covariance_no_eigv(self):
        a = ma.CovarianceNoEigvAdaptation(3, np.ones(3), 1.6)
        np.testing.assert_array_equal(a.m, np.ones(3))


if __name__ == "__main__":
    unittest.main()
