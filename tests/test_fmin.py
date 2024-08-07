"""Module containing tests for fmin function."""

import unittest
from modcma import modularcmaes


class TestFmin(unittest.TestCase):
    """Test case for fmin function of Modular CMA-ES."""

    def test_best_so_far_storage(self):
        """Test storage of best so far individual."""
        c = modularcmaes.ModularCMAES(sum, 5)
        c.step()
        self.assertEqual(len(c.parameters.xopt), 5)
        self.assertAlmostEqual(sum(c.parameters.xopt), c.parameters.fopt)

    def test_fmin(self):
        """Test a single run of the mechanism."""
        xopt, fopt, evaluations = modularcmaes.fmin(sum, [1, 1,1,1,1], target=0.0)
        self.assertAlmostEqual(sum(xopt), fopt)
        self.assertGreater(evaluations, 0)
        self.assertAlmostEqual(len(xopt), 5)


    
