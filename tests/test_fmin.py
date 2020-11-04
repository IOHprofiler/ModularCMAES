import unittest
from ccmaes import configurablecmaes

class TestFmin(unittest.TestCase):
    def test_best_so_far_storage(self):
        c = configurablecmaes.ConfigurableCMAES(sum, 5)
        c.step()
        self.assertEqual(len(c.parameters.xopt), 5)
        self.assertEqual(sum(c.parameters.xopt), c.parameters.fopt)
        
    def test_fmin(self):
        xopt, fopt, evaluations = configurablecmaes.fmin(sum, 5)
        self.assertEqual(sum(xopt), fopt)
        self.assertGreater(evaluations, 0)
        self.assertEqual(len(xopt), 5)

