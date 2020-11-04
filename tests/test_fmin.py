import unittest
from modcma import modularcmaes

class TestFmin(unittest.TestCase):
    def test_best_so_far_storage(self):
        c = modularcmaes.ModularCMAES(sum, 5)
        c.step()
        self.assertEqual(len(c.parameters.xopt), 5)
        self.assertEqual(sum(c.parameters.xopt), c.parameters.fopt)
        
    def test_fmin(self):
        xopt, fopt, evaluations = modularcmaes.fmin(sum, 5, target = 0.)
        self.assertEqual(sum(xopt), fopt)
        self.assertGreater(evaluations, 0)
        self.assertEqual(len(xopt), 5)

