import unittest
import numpy as np
from src import optimizer, parameters, utils


class TestOptimizer(unittest.TestCase):
    _dim = 5
    _budget = int(1e4 * _dim)
    _target = 79.48 + 1e-08

    def setUp(self):
        params = parameters.SimpleParameters(
            self._target, self._budget
        )
        self.optimizer = optimizer.Optimizer()
        self.optimizer.parameters = params
        self.optimizer._fitness_func = utils.sphere_function

    def test_fitness_func(self):
        f = self.optimizer.fitness_func(
            np.random.randn(self._dim)
        )
        self.assertIsInstance(f, np.float)
        self.assertEqual(self.optimizer.parameters.used_budget, 1)

    def test_step(self):
        with self.assertRaises(NotImplementedError):
            self.optimizer.step()

    def test_run(self):
        with self.assertRaises(NotImplementedError):
            self.optimizer.run()

    def test_break_conditions(self):
        self.assertEqual(
            self.optimizer.break_conditions, [False, False])

        self.optimizer.parameters = parameters.SimpleParameters(
            self._target, self._budget, self._target
        )
        self.assertEqual(
            self.optimizer.break_conditions, [True, False])

        self.optimizer.parameters = parameters.SimpleParameters(
            self._target, self._budget,
            float("inf"), self._budget
        )
        self.assertEqual(
            self.optimizer.break_conditions, [False, True])

        self.optimizer.parameters = parameters.SimpleParameters(
            self._target, self._budget,
            self._target, self._budget
        )
        self.assertEqual(
            self.optimizer.break_conditions, [True, True])


if __name__ == '__main__':
    unittest.main()
