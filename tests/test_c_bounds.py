"""Module containing tests for ModularCMA-ES C++ Bounds."""

import unittest
import numpy as np
from modcma.c_maes import bounds, Population, Parameters, parameters, options



class TestBounds(unittest.TestCase):
    """Test case for Bounds of Modular C++ CMA-ES."""

    __bound_fixers = (
        bounds.COTN,
        bounds.Mirror,
        bounds.Saturate,
        bounds.Toroidal,
        bounds.UniformResample,
    )

    __bound_fixers_options = (
        options.CorrectionMethod.COTN,
        options.CorrectionMethod.MIRROR,
        options.CorrectionMethod.SATURATE,
        options.CorrectionMethod.TOROIDAL,
        options.CorrectionMethod.UNIFORM_RESAMPLE,
    )
    def setUp(self):
        self.lb, self.ub = np.zeros(2), np.ones(2) * 2
        self.par = Parameters(parameters.Settings(2, lambda0=2, lb=self.lb, ub=self.ub))
        self.par.pop.s = np.ones(2) * 2
        self.par.adaptation.m = np.ones(2) * 0.1
        Z = np.ones((2, 2)) * 0.9
        Z[0, 0] *= 2
        self.par.pop.Z = Z
        self.par.pop.Y = self.par.pop.Z.copy()
        self.par.pop.X = self.par.adaptation.m + (self.par.pop.s * self.par.pop.Y)

    def test_bound_fixers(self):
        for boundcntrl, option in zip(self.__bound_fixers, self.__bound_fixers_options):
            self.par.settings.modules.bound_correction = option
            method = boundcntrl(self.lb, self.ub)
            method.correct(1, self.par)
            self.assertEqual(method.n_out_of_bounds, 0)
            method.correct(0, self.par)
            self.assertEqual(method.n_out_of_bounds, 1)

            self.assertTrue(np.all(self.par.pop.X <= 2))
            self.assertTrue(np.all(np.isclose(self.par.pop.Y.ravel()[1:], 0.9)))
            self.assertTrue(np.all(np.isclose(self.par.pop.X.ravel()[1:], 1.9)))
            self.setUp()

    def test_do_nothing(self):
        method = bounds.NoCorrection(self.lb, self.ub)
        method.correct(0, self.par)
        self.assertEqual(method.n_out_of_bounds, 1)
        self.assertFalse(np.all(self.par.pop.X <= 2))
        self.assertTrue(np.all(np.isclose(self.par.pop.Y.ravel()[1:], 0.9)))
        self.assertTrue(np.all(np.isclose(self.par.pop.X.ravel()[1:], 1.9)))


if __name__ == "__main__":
    unittest.main()
