"""Every run ends with an evaluated center of mass and a record of its distribution (#111)."""

import unittest

import numpy as np

from modcma import c_maes
from modcma.c_maes import ModularCMAES, Settings, restart, settings_from_dict


def sphere(x):
    return float(np.sum(x * x))


class RestartEvery(restart.Criterion):
    """Met every `period` generations"""

    def __init__(self, period):
        super().__init__("RestartEvery")
        self.period = period

    def update(self, parameters):
        self.met = parameters.stats.t % self.period == self.period - 1


def records(es):
    return [(c.x.tolist(), c.y, c.t, c.e, c.sigma, c.metric.d.tolist(), c.reasons)
            for c in es.p.stats.centers]


class TestTheEndOfARun(unittest.TestCase):
    def check_end(self, es, reason):
        stats = es.p.stats
        self.assertTrue(es.break_conditions())
        self.assertLessEqual(stats.evaluations, es.p.settings.budget)
        self.assertEqual(len(stats.centers), len(stats.solutions))
        last = stats.centers[-1]
        self.assertIn(reason, last.reasons)
        # The record is the center of mass and the distribution at the end of the run
        np.testing.assert_array_equal(last.x, es.center_of_mass)
        self.assertEqual(last.e, stats.evaluations)
        self.assertEqual(last.sigma, es.p.mutation.sigma)
        np.testing.assert_array_equal(last.metric.d, es.p.adaptation.metric().d)
        self.assertEqual(last.std, last.sigma * last.metric.d.max())
        self.assertLessEqual(stats.global_best.y, last.y)
        self.assertIsInstance(last, c_maes.parameters.Solution)

    def test_every_way_a_run_ends(self):
        cases = [
            ("budget", dict(budget=1000)),
            ("target", dict(target=1e-8)),
            ("max_generations", dict(max_generations=20)),
            ("TolX", dict(restart_strategy="STOP")),
        ]
        for reason, kwargs in cases:
            with self.subTest(reason):
                es = ModularCMAES(settings_from_dict(5, seed=1, **kwargs))
                es.run(sphere)
                self.check_end(es, reason)
                self.assertEqual(len(es.p.stats.centers), 1)

    def test_restarts(self):
        es = ModularCMAES(settings_from_dict(5, seed=1, restart_strategy="IPOP", budget=20_000))
        es.run(sphere)
        self.check_end(es, "budget")
        centers = es.p.stats.centers
        self.assertGreater(len(centers), 1)
        for center in centers[:-1]:
            self.assertTrue(center.reasons)  # the criteria that started the restart
            self.assertNotIn("budget", center.reasons)

    def test_every_matrix_adaptation(self):
        for name in ["COVARIANCE", "COVARIANCE_NO_EIGV", "MATRIX", "SEPARABLE", "CHOLESKY",
                     "CMSA", "NONE", "NATURAL_GRADIENT"]:
            with self.subTest(name):
                es = ModularCMAES(settings_from_dict(5, matrix_adaptation=name, seed=1,
                                                     budget=1000))
                es.run(sphere)
                self.check_end(es, "budget")
                self.assertEqual(es.p.stats.centers[-1].metric.B is None,
                                 name in ("SEPARABLE", "NONE"))

    def test_the_center_at_a_restart_is_the_center_of_mass(self):
        # Without repelling, every one-point batch is a center of mass: at each restart, and the
        # last one when the optimizer stops
        es = ModularCMAES(Settings(3, lambda0=6, budget=400, seed=1))
        es.p.criteria.items.append(RestartEvery(5))
        asked = []
        while not es.break_conditions():
            X = es.ask()
            if len(X) == 1:
                asked.append(X[0].copy())
                np.testing.assert_array_equal(X[0], es.center_of_mass)
            es.tell([sphere(x) for x in X])
        self.assertGreater(len(asked), 2)
        np.testing.assert_array_equal([c.x for c in es.p.stats.centers], asked)
        self.assertEqual(es.p.stats.centers[0].reasons, ["RestartEvery"])

    def test_a_mean_outside_the_box(self):
        evaluated = []

        def objective(x):
            evaluated.append(x.copy())
            return sphere(x)

        for correction, expected in [("SATURATE", 5.0), ("NONE", 10.0)]:
            with self.subTest(correction):
                es = ModularCMAES(settings_from_dict(
                    3, lb=-5 * np.ones(3), ub=5 * np.ones(3), bound_correction=correction,
                    max_generations=1, seed=1))
                es.step(objective)
                es.p.adaptation.m = np.full(3, 10.0)  # e.g. after injected points
                self.assertFalse(es.step(objective))  # the final center of mass
                np.testing.assert_array_equal(evaluated[-1], np.full(3, expected))
                np.testing.assert_array_equal(es.p.stats.centers[-1].x, np.full(3, expected))

    def test_the_drivers_give_the_same_records(self):
        def make():
            es = ModularCMAES(settings_from_dict(4, seed=2, restart_strategy="IPOP",
                                                 budget=3000))
            return es

        reference = make()
        reference.run(sphere)

        stepped = make()
        while stepped.step(sphere):
            pass

        asktell = make()
        while not asktell.break_conditions():
            X = asktell.ask()
            asktell.tell([sphere(x) for x in X])

        steps = make()
        while not steps.break_conditions():
            steps.mutate(sphere)
            steps.select()
            steps.recombine()
            steps.adapt()

        vectorized = make()
        vectorized.run(lambda X: np.sum(X * X, axis=1), vectorized=True)

        for es in (stepped, asktell, steps, vectorized):
            self.assertEqual(records(es), records(reference))
            self.assertEqual(es.p.stats.evaluations, reference.p.stats.evaluations)


class TestTheDrivers(unittest.TestCase):
    def test_stop_and_break_conditions(self):
        es = ModularCMAES(Settings(4, lambda0=8, budget=25, seed=1))
        while not es.stop_conditions():
            X = es.ask()
            self.assertEqual(len(X), 8)
            es.tell([sphere(x) for x in X])
        # The optimizer should stop, but has not: the final center of mass is still to come
        self.assertFalse(es.break_conditions())
        X = es.ask()
        self.assertEqual(X.shape, (1, 4))
        self.assertTrue(es.tell([sphere(X[0])]))
        self.assertTrue(es.break_conditions())
        self.assertEqual(es.p.stats.evaluations, 25)

    def test_the_last_step_evaluates_the_center(self):
        es = ModularCMAES(Settings(4, lambda0=8, budget=25, seed=1))
        results = []
        while True:
            before = es.p.stats.evaluations
            running = es.step(sphere)
            results.append((es.p.stats.evaluations - before, running))
            if not running:
                break
        self.assertEqual(results, [(8, True)] * 3 + [(1, False)])

    def test_the_steps_after_the_end_do_nothing(self):
        es = ModularCMAES(Settings(4, lambda0=8, budget=25, seed=1))
        for _ in range(3):
            es.mutate(sphere)
            es.select()
            es.recombine()
            es.adapt()
        es.mutate(sphere)  # the final center of mass instead of a generation
        m, t = es.p.adaptation.m.copy(), es.p.stats.t
        es.select()
        es.recombine()
        es.adapt()
        np.testing.assert_array_equal(es.p.adaptation.m, m)
        self.assertEqual(es.p.stats.t, t)
        self.assertTrue(es.break_conditions())

    def test_a_raised_budget_continues_the_run(self):
        es = ModularCMAES(Settings(4, lambda0=8, budget=25, seed=1))
        es.run(sphere)
        self.assertEqual(len(es.p.stats.centers), 1)
        es.p.settings.budget = 50
        self.assertFalse(es.break_conditions())
        es.run(sphere)
        self.assertEqual(es.p.stats.evaluations, 50)
        self.assertEqual(len(es.p.stats.centers), 2)
        self.assertTrue(es.break_conditions())

    def test_asking_after_the_end_continues_with_generations(self):
        # ask-tell that ignores break_conditions: generations, not the center again
        es = ModularCMAES(Settings(4, lambda0=8, budget=25, seed=1))
        es.run(sphere)
        X = es.ask()
        self.assertEqual(len(X), 8)
        es.tell([sphere(x) for x in X])
        self.assertEqual(len(es.p.stats.centers), 1)

    def test_an_empty_record(self):
        center = c_maes.parameters.Center()
        self.assertTrue(np.isnan(center.std))
        self.assertEqual(center.reasons, [])
        self.assertEqual(center.y, np.inf)


if __name__ == "__main__":
    unittest.main()
