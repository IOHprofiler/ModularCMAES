"""Module containing tests for ModularCMA-ES C++ Bounds."""

import unittest
import numpy as np
from modcma.c_maes import bounds, Population, Parameters, parameters, options



class TestBounds(unittest.TestCase):
    """Test case for Bounds of Modular C++ CMA-ES."""

    __bound_fixers = (
        bounds.COTN,
        bounds.UniformResample, 
        bounds.Mirror,
        bounds.Saturate,
        bounds.Toroidal,
    )

    __bound_fixers_options = (
        options.CorrectionMethod.COTN,
        options.CorrectionMethod.UNIFORM_RESAMPLE,
        options.CorrectionMethod.MIRROR,
        options.CorrectionMethod.SATURATE,
        options.CorrectionMethod.TOROIDAL,
    )
    def setUp(self):
        self.lb, self.ub = np.zeros(2), np.ones(2) * 2
        self.par = Parameters(parameters.Settings(2, lambda0=2, lb=self.lb, ub=self.ub))
        self.par.pop.S = np.ones((2, 2)) * 2
        self.par.adaptation.m = np.ones(2) * 0.1
        Z = np.ones((2, 2)) * 0.9
        Z[0, 0] *= 2
        self.par.pop.Z = Z
        self.par.pop.Y = self.par.pop.Z.copy()
        self.par.pop.X_transformed = self.par.pop.X = self.par.adaptation.m + (self.par.pop.S * self.par.pop.Y)
        

    def test_bound_fixers(self):
        for i, (boundcntrl, option) in enumerate(zip(self.__bound_fixers, self.__bound_fixers_options)):
            if i < 2:
                method = boundcntrl(self.lb.size)
            else:
                method = boundcntrl()
            
            self.par.settings.modules.bound_correction = option
            method.correct(1, self.par)
            self.assertEqual(method.n_out_of_bounds, 0)
            method.correct(0, self.par)
            self.assertEqual(method.n_out_of_bounds, 1)
            self.assertTrue(np.all(self.par.pop.X <= 2))
            self.assertTrue(np.all(np.isclose(self.par.pop.Y.ravel()[1:], 0.9)))
            self.assertTrue(np.all(np.isclose(self.par.pop.X.ravel()[1:], 1.9)))
            self.setUp()

    def test_do_nothing(self):
        method = bounds.NoCorrection()
        method.correct(0, self.par)
        self.assertEqual(method.n_out_of_bounds, 1)
        self.assertFalse(np.all(self.par.pop.X <= 2))
        self.assertTrue(np.all(np.isclose(self.par.pop.Y.ravel()[1:], 0.9)))
        self.assertTrue(np.all(np.isclose(self.par.pop.X.ravel()[1:], 1.9)))
        
    def test_center(self):
        self.assertTrue(np.all(self.par.settings.center == 1))
        self.assertTrue(np.all(self.par.settings.lb == 0))
        self.assertTrue(np.all(self.par.settings.ub == 2))
        self.assertTrue(np.all(self.par.settings.db == 2))


if __name__ == "__main__":
    unittest.main()
