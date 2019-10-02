import unittest
from ccmaes.parameters import SimpleParameters


class TestSimpleParameters(unittest.TestCase):
    def test_types(self):
        parameters = SimpleParameters(.1, 100, .1, 0)
        with self.assertRaises(TypeError):
            parameters.target = 1

        with self.assertRaises(TypeError):
            parameters.budget = 1.

        with self.assertRaises(TypeError):
            parameters.fopt = 1

        with self.assertRaises(TypeError):
            parameters.used_budget = 1.


if __name__ == '__main__':
    unittest.main()
