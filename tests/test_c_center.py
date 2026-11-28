"""Tests of Settings.center with infinite bounds (#134)."""

import unittest
import numpy as np
from modcma import c_maes
from modcma.c_maes import parameters

inf = np.inf


def center(lb, ub, sigma0=2.0):
    s = parameters.Settings(len(lb), lb=lb, ub=ub, sigma0=sigma0)
    return np.array(s.center)


class TestCenter(unittest.TestCase):
    def test_rule(self):
        # finite, infinite both, [lb, inf) with 0 inside and outside, (-inf, ub] likewise
        lb = [2, -inf, 0, -3, 5, -inf, -inf]
        ub = [4, inf, inf, inf, inf, 7, -7]
        want = [3, 0, 0, 0, 5 + 2, 0, -7 - 2]
        np.testing.assert_array_equal(center(lb, ub), want)

    def test_sigma0_larger_than_distance(self):
        # lb + sigma0 is still in the half-line
        np.testing.assert_array_equal(center([1, -inf], [inf, -1], 10.0), [11, -11])

    def test_finite_box_unchanged(self):
        np.testing.assert_array_equal(center([-5, 0], [5, 4]), [0, 2])

    def test_follows_setters(self):
        s = parameters.Settings(2, lb=[2, 0], ub=[4, inf])
        np.testing.assert_array_equal(s.center, [3, 0])
        s.lb = [-inf, 1]
        np.testing.assert_array_equal(s.center, [0, 3])
        s.sigma0 = 0.5
        np.testing.assert_array_equal(s.center, [0, 1.5])
        s.ub = [-1, inf]
        np.testing.assert_array_equal(s.center, [-1.5, 1.5])

    def test_first_mean_in_box_run_and_ask_tell(self):
        lb, ub = np.array([2.0, 0.0]), np.array([4.0, inf])
        seen = []

        def f(x):
            seen.append(np.array(x))
            return float(np.sum((x - 3) ** 2))

        def make():
            c_maes.set_seed(1)
            return c_maes.ModularCMAES(
                parameters.Settings(2, lb=lb, ub=ub, budget=200, sigma0=0.5)
            )

        es = make()
        m = np.array(es.p.adaptation.m)
        self.assertTrue(np.all(m >= lb) and np.all(m <= ub))
        es.run(f)
        es2 = make()
        np.testing.assert_array_equal(es2.p.adaptation.m, m)
        np.testing.assert_array_equal(np.array(es2.ask())[0], seen[0])


if __name__ == "__main__":
    unittest.main()
