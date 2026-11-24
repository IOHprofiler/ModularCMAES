"""The ask-tell interface of the C++ ModularCMAES (TASKS.md, WP3).

tests/golden.py checks that ask/tell, ask/tell with the points passed back and a vectorized
objective reproduce the golden trajectories of step() bit for bit. These tests cover what that does
not: configurations the golden runs do not reach, the batches ask() returns, injection and errors.
"""

import math
import unittest

import numpy as np

from modcma import c_maes


def sphere(x):
    return math.fsum(xi * xi for xi in x)


def rastrigin(x):
    return 10.0 * len(x) + math.fsum(xi * xi - 10.0 * math.cos(2.0 * math.pi * xi) for xi in x)


class RestartAt(c_maes.restart.Criterion):
    """Triggers a restart after the given generations"""

    def __init__(self, *generations):
        super().__init__("RestartAt")
        self.generations = set(generations)

    def update(self, parameters):
        self.met = parameters.stats.t in self.generations


def make(d=4, seed=12, **kwargs):
    modules = c_maes.Modules()
    for name, value in kwargs.pop("modules", {}).items():
        if not c_maes.set_module(modules, name, value):
            raise ValueError(name)
    c_maes.set_seed(seed)
    return c_maes.ModularCMAES(c_maes.Settings(d, modules, verbose=False, **kwargs))


def ask_tell_generation(cma, objective, sizes=None):
    while True:
        X = cma.ask()
        if sizes is not None:
            sizes.append(len(X))
        if cma.tell([objective(x) for x in X]):
            return not cma.break_conditions()


def state(cma):
    p = cma.p
    return (
        p.pop.X.copy(),
        p.pop.f.copy(),
        p.adaptation.m.copy(),
        p.mutation.sigma,
        p.stats.evaluations,
        p.stats.global_best.y,
        len(p.stats.centers),
    )


def assert_same_state(test, a, b):
    for x, y in zip(state(a), state(b)):
        np.testing.assert_array_equal(x, y)


class TestEquivalentToStep(unittest.TestCase):
    """ask/tell gives the same trajectory as step(), bit for bit, in configurations that the
    golden runs do not reach"""

    CONFIGS = {
        # mu=2: sequential selection with pairwise mirroring can stop after a complete pair
        "pairwise_sequential": dict(
            modules={"mirrored": "PAIRWISE", "sequential_selection": True}, mu0=2, lambda0=10
        ),
        "sequential_tpa": dict(
            modules={"sequential_selection": True, "ssa": "TPA"}, mu0=2, lambda0=10
        ),
        "threshold_sequential": dict(
            modules={"sequential_selection": True, "threshold_convergence": True}, mu0=2, lambda0=8
        ),
        "integer_resample": dict(
            modules={"bound_correction": "RESAMPLE"},
            integer_variables=[0, 2],
            lb=np.full(4, -1.0),
            ub=np.full(4, 3.0),
        ),
        "elitist_sa": dict(modules={"elitist": True, "ssa": "SA"}),
        "bipop_coverage": dict(
            modules={"restart_strategy": "BIPOP", "repelling_type": "COVERAGE"}, budget=6000
        ),
        "ipop_adaptive_current": dict(
            modules={
                "restart_strategy": "IPOP",
                "repelling_type": "ADAPTIVE",
                "repelling_metric": "CURRENT",
            },
            budget=6000,
        ),
    }

    def run_both(self, objective, restarts=(), **config):
        a, b = make(**config), make(**config)
        for cma in (a, b):
            if restarts:
                cma.p.criteria.items.append(RestartAt(*restarts))
        # set_seed sets the one global generator: run the two optimizers one after the other
        c_maes.set_seed(12)
        states_a, running = [], True
        while running:
            running = a.step(objective)
            states_a.append(state(a))
        c_maes.set_seed(12)
        states_b, running = [], True
        while running:
            running = ask_tell_generation(b, objective)
            states_b.append(state(b))
        self.assertEqual(len(states_a), len(states_b))
        for sa, sb in zip(states_a, states_b):
            for x, y in zip(sa, sb):
                np.testing.assert_array_equal(x, y)
        return a

    def test_configs(self):
        for name, config in self.CONFIGS.items():
            for objective in (sphere, rastrigin):
                with self.subTest(config=name, objective=objective.__name__):
                    config = dict(config)
                    config.setdefault("budget", 3000)
                    self.run_both(objective, restarts=(15, 40, 70), **config)


class TestBatches(unittest.TestCase):
    def test_default_population(self):
        cma = make(lambda0=9, mu0=3)
        self.assertEqual(cma.ask().shape, (9, 4))

    def test_sequential_selection_pairwise(self):
        # cut-off 2 mu = 4, the first possible stop after a pair: individuals 0..5, then pairs
        cma = make(modules={"mirrored": "PAIRWISE", "sequential_selection": True}, mu0=2, lambda0=10)
        stopped_early = False
        for _ in range(60):
            sizes = []
            ask_tell_generation(cma, sphere, sizes)
            self.assertEqual(sizes[0], 6)
            self.assertTrue(all(k == 2 for k in sizes[1:]), sizes)
            self.assertEqual(cma.p.pop.n % 2, 0)
            stopped_early |= cma.p.pop.n < 10
        self.assertTrue(stopped_early)

    def test_tpa_points(self):
        cma = make(modules={"ssa": "TPA"}, lambda0=6, mu0=3)
        for _ in range(3):
            ask_tell_generation(cma, sphere)
        X = cma.ask()
        self.assertEqual(X.shape, (8, 4))
        a, sigma = cma.p.adaptation, cma.p.mutation.sigma
        # not bitwise: a compiler may fuse m + sigma * dm into one multiply-add
        np.testing.assert_allclose(X[-2], a.m + sigma * a.dm, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(X[-1], a.m + sigma * -a.dm, rtol=1e-14, atol=1e-15)

    def test_restart_asks_center(self):
        cma = make(modules={"restart_strategy": "RESTART"})
        cma.p.criteria.items.append(RestartAt(3))  # met in the 4th generation (t = 3)
        for _ in range(4):
            ask_tell_generation(cma, sphere)
        m = cma.p.adaptation.m.copy()
        X = cma.ask()
        np.testing.assert_array_equal(X, m[None, :])
        self.assertFalse(cma.tell([12.5]))
        self.assertEqual(len(cma.p.stats.centers), 1)
        self.assertEqual(cma.p.stats.centers[0].y, 12.5)
        np.testing.assert_array_equal(cma.p.stats.centers[0].x, m)
        self.assertEqual(len(cma.ask()), cma.p.lamb)

    def test_hill_valley_points_one_at_a_time(self):
        # Coverage repelling tests every restart's candidate against the archive with up to 10
        # points between them, each asked on its own
        cma = make(modules={"restart_strategy": "RESTART", "repelling_type": "COVERAGE"})
        cma.p.criteria.items.append(RestartAt(5, 10))
        sizes = []
        for _ in range(12):
            ask_tell_generation(cma, lambda x: 1.0, sizes)  # flat: every test passes all 10
        self.assertEqual(len(cma.p.repelling.archive), 1)
        self.assertEqual(cma.p.repelling.archive[0].n_rep, 2)
        # restart 1: center (empty archive); restart 2: center + 10 hill-valley points
        self.assertEqual(sizes.count(1), 12)


class TestInterface(unittest.TestCase):
    def test_ask_twice_returns_the_same_points(self):
        cma = make()
        X = cma.ask()
        X[0] = 1e9  # a copy: changing it does not change the pending points
        Y = cma.ask()
        self.assertFalse(np.array_equal(X, Y))
        np.testing.assert_array_equal(Y, cma.ask())
        self.assertTrue(cma.has_pending)

    def test_errors(self):
        cma = make()
        with self.assertRaisesRegex(RuntimeError, "ask"):
            cma.tell([1.0])
        X = cma.ask()
        with self.assertRaisesRegex(ValueError, "objective values"):
            cma.tell(np.zeros(len(X) + 1))
        with self.assertRaisesRegex(ValueError, "matrix"):
            cma.tell(np.zeros(len(X)), X[:, :2])
        # nothing was consumed: the right values still complete the generation
        self.assertTrue(cma.tell([sphere(x) for x in X]))
        self.assertFalse(cma.has_pending)

    def test_tell_accepts_sequences_and_dtypes(self):
        cma = make()
        X = cma.ask()
        self.assertTrue(cma.tell(np.array([sphere(x) for x in X], dtype=np.float32)))
        X = cma.ask()
        self.assertTrue(cma.tell(tuple(int(sphere(x)) for x in X)))

    def test_mutate_continues_pending_generation(self):
        a, b = make(), make()
        c_maes.set_seed(5)
        a.step(sphere)
        c_maes.set_seed(5)
        b.ask()
        b.mutate(sphere)
        b.select()
        b.recombine()
        b.adapt()
        assert_same_state(self, a, b)

    def test_vectorized_objective_once_per_batch(self):
        calls = []

        def objective(X):
            calls.append(X.shape)
            return (X**2).sum(axis=1)

        cma = make(modules={"ssa": "TPA"}, lambda0=6, mu0=3, budget=400)
        cma.run(objective, vectorized=True)
        self.assertTrue(all(shape == (8, 4) for shape in calls), set(calls))
        self.assertEqual(len(calls), cma.p.stats.t)
        self.assertEqual(cma.p.stats.evaluations, 8 * len(calls))

    def test_run_keyword_only_vectorized(self):
        cma = make(budget=200)
        with self.assertRaises(TypeError):
            cma.run(sphere, True)


class TestInjection(unittest.TestCase):
    def test_injected_point_replaces_the_asked_one(self):
        cma = make(lambda0=8, mu0=4)
        for _ in range(3):
            ask_tell_generation(cma, sphere)
        m, sigma = cma.p.adaptation.m.copy(), cma.p.mutation.sigma
        X = cma.ask()
        injected = X.copy()
        injected[3] = 0.0  # the optimum
        f = [sphere(x) for x in injected]
        self.assertTrue(cma.tell(f, injected))

        p = cma.p
        np.testing.assert_array_equal(p.stats.global_best.x, injected[3])
        self.assertEqual(p.stats.global_best.y, 0.0)
        # the best individual after selection is the injected one, with y and z derived from it
        np.testing.assert_array_equal(p.pop.X[:, 0], injected[3])
        np.testing.assert_allclose(p.pop.Y[:, 0] * p.pop.S[:, 0] + m, injected[3], atol=1e-12)
        self.assertEqual(p.pop.S[0, 0], sigma)

    def test_unchanged_points_are_not_injected(self):
        a, b = make(), make()
        c_maes.set_seed(8)
        for _ in range(5):
            ask_tell_generation(a, sphere)
        c_maes.set_seed(8)
        for _ in range(5):
            while True:
                X = b.ask()
                if b.tell([sphere(x) for x in X], X.copy()):
                    break
        assert_same_state(self, a, b)


if __name__ == "__main__":
    unittest.main()
