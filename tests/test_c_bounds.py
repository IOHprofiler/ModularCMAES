"""Module containing tests for ModularCMA-ES C++ Bounds."""

import unittest
import numpy as np
from modcma.c_maes import bounds, Population


class TestBounds(unittest.TestCase):
    """Test case for Bounds of Modular C++ CMA-ES."""

    __bound_fixers = (
        bounds.COTN,
        bounds.Mirror,
        bounds.Saturate,
        bounds.Toroidal,
        bounds.UniformResample,
    )
    __do_nothing = (bounds.CountOutOfBounds, bounds.NoCorrection, )


    def setUp(self):
        self.lb =  np.zeros(2)
        self.ub =  np.ones(2) * 2
        
        self.pop = Population(2, 2)
        self.pop.s = np.ones(2) * 1
        self.m = np.ones(2) * 0.1
        Z = np.ones((2, 2)) * 1.5
        Z[0, 0] *= 2
        self.pop.Z = Z
        self.pop.Y = self.pop.Z.copy()
        self.pop.X = self.m + (self.pop.s * self.pop.Y)

    def test_bound_fixers(self):
        for boundcntrl in self.__bound_fixers:
            method = boundcntrl(self.lb, self.ub)
            method.correct(self.pop, self.m)
            self.assertEqual(method.n_out_of_bounds, 1)
            self.assertTrue(np.all(self.pop.X <= 2))
            self.assertTrue(np.all(self.pop.Y.ravel()[1:] == 1.5))
            self.assertTrue(np.all(self.pop.X.ravel()[1:] == 1.6))
            self.setUp()

    def test_do_nothing(self):
        method = bounds.NoCorrection(self.lb, self.ub)
        method.correct(self.pop, self.m)

        self.assertEqual(method.n_out_of_bounds, 0)
        self.assertFalse(np.all(self.pop.X <= 2))
        self.assertTrue(np.all(self.pop.Y.ravel()[1:] == 1.5))
        self.assertTrue(np.all(self.pop.X.ravel()[1:] == 1.6))

    def test_do_count(self):
        method = bounds.CountOutOfBounds(self.lb, self.ub)
        method.correct(self.pop, self.m)

        self.assertEqual(method.n_out_of_bounds, 1)
        self.assertFalse(np.all(self.pop.X <= 2))
        self.assertTrue(np.all(self.pop.Y.ravel()[1:] == 1.5))
        self.assertTrue(np.all(self.pop.X.ravel()[1:] == 1.6))


if __name__ == "__main__":
    unittest.main()
