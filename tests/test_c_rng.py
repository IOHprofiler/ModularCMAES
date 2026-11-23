"""Random generators of the C++ ModularCMAES (TASKS.md, T4.1): one generator per optimizer.

The golden trajectories (tests/golden.py) check that set_seed(s) followed by one optimizer draws the
same numbers as before T4.1.
"""

import math
import threading
import unittest

import numpy as np

from modcma import c_maes


def sphere(x):
    return math.fsum(xi * xi for xi in x)


def settings(**kwargs):
    modules = c_maes.Modules()
    modules.restart_strategy = c_maes.options.RestartStrategy.BIPOP
    modules.bound_correction = c_maes.options.CorrectionMethod.COTN
    modules.center_placement = c_maes.options.CenterPlacement.UNIFORM
    modules.sample_sigma = True
    kwargs.setdefault("budget", 3000)
    return c_maes.Settings(4, modules, lb=np.full(4, -2.0), ub=np.full(4, 2.0), **kwargs)


def trajectory(cma, generations=None):
    states = []
    while (generations is None or len(states) < generations) and cma.step(sphere):
        states.append((cma.p.adaptation.m.copy(), cma.p.mutation.sigma, cma.p.stats.evaluations))
    return states


def assert_same(test, a, b):
    test.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x[0], y[0])
        test.assertEqual(x[1:], y[1:])


class TestGeneratorPerOptimizer(unittest.TestCase):
    def test_interleaved_optimizers_do_not_share_random_numbers(self):
        alone = trajectory(c_maes.ModularCMAES(settings(seed=3)))
        a = c_maes.ModularCMAES(settings(seed=3))
        b = c_maes.ModularCMAES(settings(seed=3))
        states_a, states_b = [], []
        while True:
            running_a, running_b = a.step(sphere), b.step(sphere)
            states_a.append((a.p.adaptation.m.copy(), a.p.mutation.sigma, a.p.stats.evaluations))
            states_b.append((b.p.adaptation.m.copy(), b.p.mutation.sigma, b.p.stats.evaluations))
            self.assertEqual(running_a, running_b)
            if not running_a:
                break
        assert_same(self, states_a, states_b)
        assert_same(self, states_a[:-1], alone)
        self.assertGreater(len(a.p.stats.centers), 0)  # restarts: BIPOP and center placement

    def test_global_seed_before_creation(self):
        c_maes.utils.set_seed(11)
        first = trajectory(c_maes.ModularCMAES(settings()), 30)
        c_maes.utils.set_seed(11)
        again = trajectory(c_maes.ModularCMAES(settings()), 30)
        assert_same(self, first, again)

    def test_optimizers_created_after_one_seed_differ(self):
        c_maes.utils.set_seed(11)
        a = c_maes.ModularCMAES(settings())
        b = c_maes.ModularCMAES(settings())
        self.assertFalse(
            np.array_equal(trajectory(a, 5)[-1][0], trajectory(b, 5)[-1][0])
        )

    def test_seed_is_independent_of_the_global_generator(self):
        c_maes.utils.set_seed(1)
        a = trajectory(c_maes.ModularCMAES(settings(seed=5)), 20)
        c_maes.utils.set_seed(2)
        c_maes.sampling.Gaussian(4)()  # draws from the global generator
        b = trajectory(c_maes.ModularCMAES(settings(seed=5)), 20)
        assert_same(self, a, b)

    def test_threads(self):
        reference = trajectory(c_maes.ModularCMAES(settings(seed=9)), 40)
        results = [None] * 4

        def work(k):
            results[k] = trajectory(c_maes.ModularCMAES(settings(seed=9)), 40)

        threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for result in results:
            assert_same(self, result, reference)


if __name__ == "__main__":
    unittest.main()
