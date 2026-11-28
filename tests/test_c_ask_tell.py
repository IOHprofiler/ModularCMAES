"""The ask-tell interface of the C++ ModularCMAES (dev/overhaul-2.0.md, WP3).

tests/golden.py checks that ask/tell, ask/tell with the points passed back and a vectorized
objective reproduce the golden trajectories of step() bit for bit. These tests cover what that does
not: configurations the golden runs do not reach, the batches ask() returns, injection and errors.
"""

import math
import subprocess
import sys
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
        # make() gives both the same generator (each optimizer draws from its own)
        states_a, running = [], True
        while running:
            running = a.step(objective)
            states_a.append(state(a))
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
        a, sigma_old = cma.p.adaptation, cma.p.mutation.sigma_old
        # Half the last mean step, with the step size that sampled it (#167). Not bitwise: a
        # compiler may fuse m + s * dm into one multiply-add
        step = 0.5 * sigma_old * a.dm
        np.testing.assert_allclose(X[-2], a.m + step, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(X[-1], a.m - step, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(X[-2] - X[-1], a.m - a.m_old, rtol=1e-12, atol=1e-15)

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


class TestCounting(unittest.TestCase):
    def test_every_told_point_is_counted_once(self):
        # #84: the optimizer counts every evaluation where the values arrive (no module counts)
        configs = [
            {"ssa": "TPA", "sequential_selection": True, "mirrored": "PAIRWISE"},
            {"restart_strategy": "IPOP", "repelling_type": "COVERAGE", "ssa": "TPA"},
        ]
        for modules in configs:
            cma = make(modules=modules, budget=3000)
            told = 0
            while not cma.break_conditions():
                X = cma.ask()
                cma.tell([np.sum(x * x) - 0.5 * np.cos(9 * x).sum() for x in X])
                told += len(X)
            self.assertEqual(cma.p.stats.evaluations, told, modules)
            self.assertLessEqual(told, 3000)


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
        a.step(sphere)
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
        # One call per generation, then one for the final center of mass (#111)
        self.assertTrue(all(shape == (8, 4) for shape in calls[:-1]), set(calls))
        self.assertEqual(calls[-1], (1, 4))
        self.assertEqual(len(calls), cma.p.stats.t + 1)
        self.assertEqual(cma.p.stats.evaluations, 8 * cma.p.stats.t + 1)

    def test_run_keyword_only_vectorized(self):
        cma = make(budget=200)
        with self.assertRaises(TypeError):
            cma.run(sphere, True)


class TestSteps(unittest.TestCase):
    """mutate(), select(), recombine() and adapt() behave as in 1.x"""

    def test_mutate_samples_a_new_generation_on_every_call(self):
        cma = make(lambda0=6, mu0=3)
        cma.mutate(sphere)
        first = cma.p.pop.X.copy()
        cma.mutate(sphere)
        self.assertEqual(cma.p.stats.evaluations, 12)
        np.testing.assert_array_equal(cma.p.old_pop.X, first)
        cma.select()
        cma.mutate(sphere)
        self.assertEqual(cma.p.stats.evaluations, 18)
        cma.select()
        cma.recombine()
        cma.adapt()
        self.assertEqual(cma.p.stats.t, 1)

    def test_select_recombine_and_adapt_refuse_a_generation_that_waits_for_tell(self):
        cma = make()
        cma.ask()
        for step in (cma.select, cma.recombine, cma.adapt):
            with self.assertRaisesRegex(RuntimeError, "tell"):
                step()
        cma.mutate(sphere)  # evaluates the generation that ask() started
        cma.select()
        cma.recombine()
        cma.adapt()
        self.assertEqual(cma.p.stats.t, 1)


# Runs in a subprocess with a timeout: an interrupted mutate() once made the next ask() loop forever
INTERRUPTED_RUNS = """
import math

import numpy as np

from modcma import c_maes


def rastrigin(x):
    return 10.0 * len(x) + math.fsum(xi * xi - 10.0 * math.cos(2.0 * math.pi * xi) for xi in x)


class Interrupt(KeyboardInterrupt):
    pass


class RestartAt(c_maes.restart.Criterion):
    def __init__(self, *generations):
        super().__init__("RestartAt")
        self.generations = set(generations)

    def update(self, parameters):
        self.met = parameters.stats.t in self.generations


def make():
    settings = c_maes.settings_from_dict(
        4, seed=3, budget=3000, ssa="TPA", restart_strategy="IPOP", repelling_type="COVERAGE",
        sequential_selection=True,
    )
    cma = c_maes.ModularCMAES(settings)
    cma.p.criteria.items.append(RestartAt(10, 20, 30))
    return cma


def state(cma):
    p = cma.p
    return (p.adaptation.m.tolist(), p.mutation.sigma, p.stats.evaluations, p.stats.t,
            len(p.stats.centers))


def generation(cma, driver, objective):
    if driver == "step":
        cma.step(objective)
    elif driver == "vectorized":
        cma.step(lambda X: [objective(x) for x in X], vectorized=True)
    elif driver == "asktell":
        while not cma.tell([objective(x) for x in cma.ask()]):
            pass
    else:
        cma.mutate(objective)
        cma.select()
        cma.recombine()
        cma.adapt()


def run(driver, interrupts=()):
    # The objective raises at the given calls (as a KeyboardInterrupt would); the run goes on with
    # the next call of the same driver, which evaluates the pending points again
    calls = [0]

    def objective(x):
        calls[0] += 1
        if calls[0] in interrupts:
            raise Interrupt()
        return rastrigin(x)

    cma, n = make(), 0
    while n < 45:
        try:
            generation(cma, driver, objective)
            n += 1
        except Interrupt:
            pass
    return state(cma)


for driver in ("step", "vectorized", "asktell", "mutate"):
    interrupted = run(driver, interrupts={3, 50, 222, 401, 650, 777})
    assert interrupted == run(driver), driver
    assert interrupted[4] == 3, interrupted  # the restarts and their hill-valley tests were reached

# After an interrupted mutate(): step(), ask() and tell() go on
cma, calls = make(), [0]


def objective(x):
    calls[0] += 1
    if calls[0] == 3:
        raise Interrupt()
    return rastrigin(x)


try:
    cma.mutate(objective)
except Interrupt:
    pass
assert cma.step(rastrigin)
assert cma.step(rastrigin)
X = cma.ask()
while not cma.tell([rastrigin(x) for x in X]):
    X = cma.ask()
reference = make()
for _ in range(3):
    reference.step(rastrigin)
assert state(cma) == state(reference)
print("ok")
"""


class TestRobustness(unittest.TestCase):
    """An exception from the objective, or a module replaced between ask() and tell(), leaves the
    optimizer usable"""

    def test_interrupted_objective(self):
        result = subprocess.run(
            [sys.executable, "-c", INTERRUPTED_RUNS],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_step_size_adaptation_replaced_between_ask_and_tell(self):
        cma, other = make(modules={"ssa": "TPA"}), make(modules={"ssa": "TPA"})
        X = cma.ask()
        # The strategy that queued the points is freed: their values go to the new one
        cma.p.mutation = other.p.mutation
        del other
        self.assertTrue(cma.tell([sphere(x) for x in X]))
        for _ in range(3):
            self.assertTrue(cma.step(sphere))
        self.assertEqual(cma.p.stats.evaluations, 4 * 10)

    def test_repelling_replaced_during_a_hill_valley_test(self):
        cma = make(modules={"restart_strategy": "RESTART", "repelling_type": "COVERAGE"})
        cma.p.criteria.items.append(RestartAt(5, 10))
        previous = 0
        for _ in range(200):
            X = cma.ask()
            # a restart asks for the final mean (one point); a single point right after it is a
            # point of a hill-valley test
            if len(X) == 1 and previous == 1:
                break
            previous = len(X)
            cma.tell([1.0] * len(X))
        else:
            self.fail("no hill-valley test was reached")
        # The archive that queued the point is freed; the new one has no test in progress
        cma.p.repelling = c_maes.repelling.CoverageRepelling()
        with self.assertRaisesRegex(RuntimeError, "no hill-valley point is pending"):
            cma.tell([1.0])
        # The restart goes on: the next batch is the population of the new run
        self.assertFalse(cma.has_pending)
        self.assertEqual(len(cma.ask()), cma.p.lamb)

    def test_exception_in_the_update(self):
        # An exception in the update (here from a restart criterion written in Python) does not
        # make the next ask() run the update again: it starts a new generation
        class Raises(c_maes.restart.Criterion):
            def __init__(self):
                super().__init__("Raises")
                self.armed = False

            def update(self, parameters):
                if self.armed:
                    self.armed = False
                    raise ValueError("raised by the criterion")

        cma, criterion = make(), Raises()
        cma.p.criteria.items.append(criterion)
        for _ in range(3):
            ask_tell_generation(cma, sphere)
        X = cma.ask()
        criterion.armed = True
        with self.assertRaisesRegex(ValueError, "raised by the criterion"):
            cma.tell([sphere(x) for x in X])
        t, m = cma.p.stats.t, cma.p.adaptation.m.copy()
        self.assertFalse(cma.has_pending)
        self.assertEqual(len(cma.ask()), cma.p.lamb)
        self.assertEqual(cma.p.stats.t, t)
        np.testing.assert_array_equal(cma.p.adaptation.m, m)
        self.assertTrue(ask_tell_generation(cma, sphere))
        self.assertEqual(cma.p.stats.t, t + 1)

    def test_exception_in_the_reset_of_a_restart(self):
        # An exception while the modules are reset for the new run (from the on_reset of a restart
        # criterion written in Python) does not make the next ask() reset again: the new run keeps
        # its placed mean and the restart strategy's sigma
        class Restart(c_maes.restart.Criterion):
            def __init__(self, raises):
                super().__init__("Restart")
                self.raises = raises

            def update(self, parameters):
                self.met = parameters.stats.t == 5

            def on_reset(self, parameters):
                if self.raises:
                    self.raises = False
                    raise ValueError("raised by on_reset")

        runs = []
        for raises in (False, True):
            cma = make(modules={"restart_strategy": "BIPOP", "center_placement": "UNIFORM"})
            cma.p.criteria.items.append(Restart(raises))
            X = cma.ask()
            while len(X) != 1:  # the center of the restart is asked alone
                cma.tell([sphere(x) for x in X])
                X = cma.ask()
            if raises:
                with self.assertRaisesRegex(ValueError, "raised by on_reset"):
                    cma.tell([sphere(X[0])])
            else:
                cma.tell([sphere(X[0])])
            runs.append((cma.ask().copy(), cma.p.mutation.sigma, len(cma.p.stats.centers)))
        np.testing.assert_array_equal(runs[0][0], runs[1][0])
        self.assertEqual(runs[0][1:], runs[1][1:])


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
        for _ in range(5):
            ask_tell_generation(a, sphere)
        for _ in range(5):
            while True:
                X = b.ask()
                if b.tell([sphere(x) for x in X], X.copy()):
                    break
        assert_same_state(self, a, b)


if __name__ == "__main__":
    unittest.main()
