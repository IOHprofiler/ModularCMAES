"""Saving and resuming a run (#91): pickle and copy of the optimizer and of its objects.

A run that is pickled and loaded continues bit for bit. The full check is over the golden
configurations, against the fixture of this platform: `python -m tests.golden check --drivers
resume` (a pickle round trip before every generation, and between every ask() and tell()). Here a
subset of those configurations, against the same runs without interruption, on every platform; and
what the golden runs do not cover: the objects on their own, copies, criteria defined in Python,
and states that cannot be loaded.
"""

import copy
import pickle
import unittest
import zlib

import numpy as np

from modcma import c_maes
from tests import golden
from tests.golden import rastrigin, sphere


def make(d=4, seed=3, budget=4000, **modules):
    m = c_maes.Modules()
    for name, value in modules.items():
        if not c_maes.set_module(m, name, value):
            raise ValueError(name)
    return c_maes.ModularCMAES(c_maes.Settings(d, m, budget=budget, seed=seed))


def resumed(obj):
    return pickle.loads(pickle.dumps(obj))


def generation(es, objective=rastrigin):
    while True:
        X = es.ask()
        if es.tell([objective(x) for x in X]):
            return not es.break_conditions()


class CountGenerations(c_maes.restart.Criterion):
    """A restart criterion defined in Python, with attributes of its own: met every `every`
    generations of a run"""

    def __init__(self, every):
        super().__init__("CountGenerations")
        self.every = every
        self.updates = []

    def update(self, parameters):
        self.updates.append(int(parameters.stats.t))
        self.met = (parameters.stats.t - self.last_restart + 1) % self.every == 0


class Annotated(c_maes.ModularCMAES):
    """A Python subclass with an attribute of its own"""


# Every option of every module, in the configurations of the golden fixture's pairwise covering set
CONFIGS = [run["modules"] for run in golden.make_runs() if run["id"].startswith("pair/")]


class TestPickle(unittest.TestCase):
    def test_resumed_run_continues_bit_for_bit(self):
        for modules in ({}, {"ssa": "TPA", "restart_strategy": "BIPOP"}, {"sampler": "SOBOL"}):
            with self.subTest(**modules):
                a, b = make(**modules), make(**modules)
                for _ in range(25):
                    generation(a)
                    generation(b)
                b = resumed(b)
                for _ in range(200):
                    self.assertEqual(generation(a), generation(b))
                    np.testing.assert_array_equal(a.p.pop.X, b.p.pop.X)
                    self.assertEqual(a.p.mutation.sigma, b.p.mutation.sigma)
                # The whole state, through its serialization
                self.assertEqual(a.__getstate__(), b.__getstate__())

    def test_between_ask_and_tell(self):
        # The points of ask() wait for their values in the saved state: the loaded optimizer
        # returns them again and takes their values
        a = make(ssa="TPA", sequential_selection=True)
        for _ in range(10):
            generation(a)
        X = a.ask()
        b = resumed(a)
        self.assertTrue(b.has_pending)
        np.testing.assert_array_equal(b.ask(), X)
        f = [rastrigin(x) for x in X]
        self.assertEqual(a.tell(f), b.tell(f))
        self.assertEqual(a.__getstate__(), b.__getstate__())

    def test_gradient_injection_after_loading(self):
        # Between the ask() that starts a population and its tell(), the loaded optimizer accepts
        # the gradient point as the original does
        a = make(d=5)
        for _ in range(5):
            generation(a, sphere)
        a.ask()
        b = resumed(a)
        self.assertTrue(b.accepts_gradient)
        for es in (a, b):
            es.inject_gradient(2.0 * es.center_of_mass)
            self.assertFalse(es.accepts_gradient)
        np.testing.assert_array_equal(a.ask(), b.ask())
        self.assertEqual(a.__getstate__(), resumed(b).__getstate__())

    def test_stopped_run(self):
        # A run that has stopped stays stopped after loading, and continues as the original when
        # the budget is raised
        a = make(d=2, budget=300)
        a.run(sphere)
        b = resumed(a)
        self.assertTrue(b.break_conditions())
        for es in (a, b):
            es.p.settings.budget = 600
            es.run(sphere)
        self.assertEqual(a.__getstate__(), b.__getstate__())

    def test_given_mu0_after_restart(self):
        # A given mu0 sets mu after an IPOP restart by its ratio to lambda0 (Settings.mu_for):
        # whether mu0 was given is part of the saved settings
        m = c_maes.Modules()
        m.restart_strategy = c_maes.options.RestartStrategy.IPOP
        a = c_maes.ModularCMAES(c_maes.Settings(4, m, lambda0=10, mu0=2, seed=2, budget=3000))
        a.p.criteria.items.append(CountGenerations(6))
        for _ in range(3):
            generation(a)
        b = resumed(a)
        for _ in range(6):
            generation(a)
            generation(b)
        self.assertEqual((a.p.lamb, a.p.mu), (20, 4))
        self.assertEqual((b.p.lamb, b.p.mu), (20, 4))

    def test_box_muller_pair_across_generations(self):
        # With the Box-Muller transform (constants.use_box_muller) in an odd dimension, the last
        # coordinate of every second sample is the second number of a pair; with an odd lambda the
        # pair spans two generations
        default = c_maes.constants.use_box_muller
        c_maes.constants.use_box_muller = True
        try:
            a = c_maes.ModularCMAES(c_maes.Settings(3, lambda0=5, seed=4, budget=2000))
            for _ in range(5):
                generation(a)
            b = resumed(a)
            for _ in range(20):
                generation(a)
                generation(b)
            np.testing.assert_array_equal(a.p.pop.X, b.p.pop.X)
        finally:
            c_maes.constants.use_box_muller = default

    def test_every_pickle_protocol(self):
        # From protocol 2 on (the default is 5); 0 and 1 cannot create the C++ object
        es = make(d=2)
        es.step(sphere)
        for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                copied = pickle.loads(pickle.dumps(es, protocol))
                self.assertEqual(copied.__getstate__(), es.__getstate__())

    def test_every_module_option(self):
        # The pairwise covering set of the module options: after a restart and in the middle of a
        # generation, every module object on its own and the optimizer give back their state
        for modules in CONFIGS:
            with self.subTest(**modules):
                es = make(d=3, budget=600, **modules)
                es.p.criteria.items.append(CountGenerations(15))
                for _ in range(20):
                    generation(es)
                es.ask()
                self.assertEqual(resumed(es).__getstate__(), es.__getstate__())
                p = es.p
                for obj in (p, p.settings, p.settings.modules, p.stats, p.pop, p.old_pop,
                            p.weights, p.criteria, p.sampler, p.adaptation, p.mutation,
                            p.mutation.threshold_convergence, p.mutation.sequential_selection,
                            p.mutation.sigma_sampler, p.selection,
                            p.selection.pairwise, p.selection.elitism, p.restart_strategy,
                            p.bounds, p.repelling, p.center_placement, p.coordinate_mapping):
                    copied = resumed(obj)
                    self.assertIs(type(copied), type(obj))
                    self.assertEqual(copied.__getstate__(), obj.__getstate__())
                for criterion in p.criteria.items:
                    self.assertIs(type(resumed(criterion)), type(criterion))

    def test_deepcopy_is_independent(self):
        a = make()
        for _ in range(10):
            generation(a)
        before = a.__getstate__()
        for b in (copy.deepcopy(a), copy.copy(a)):
            self.assertEqual(b.__getstate__(), before)
            for _ in range(5):
                generation(b)
            self.assertEqual(a.__getstate__(), before)
        # A copy of the parameters is not the optimizer's
        p = copy.deepcopy(a.p)
        p.mutation.sigma = 123.0
        self.assertNotEqual(a.p.mutation.sigma, 123.0)

    def test_objective_is_not_pickled(self):
        # The run continues with the objective given to it again
        a = make()
        a.step(sphere)
        b = resumed(a)
        self.assertEqual(a.step(sphere), b.step(sphere))
        self.assertEqual(a.__getstate__(), b.__getstate__())

    def test_criterion_defined_in_python(self):
        a = make(restart_strategy="RESTART")
        a.p.criteria.items.append(CountGenerations(7))
        for _ in range(10):
            generation(a)
        b = resumed(a)
        mine = b.p.criteria.items[len(b.p.criteria.items) - 1]
        self.assertIsInstance(mine, CountGenerations)
        self.assertEqual(mine.every, 7)
        original = a.p.criteria.items[len(a.p.criteria.items) - 1]
        self.assertEqual(mine.updates, original.updates)
        self.assertEqual(mine.last_restart, original.last_restart)
        for _ in range(30):
            self.assertEqual(generation(a), generation(b))
        self.assertEqual(len(a.p.stats.centers), len(b.p.stats.centers))
        np.testing.assert_array_equal(a.p.pop.X, b.p.pop.X)
        # On its own, and copied
        alone = resumed(CountGenerations(4))
        self.assertEqual((alone.name, alone.every, alone.met), ("CountGenerations", 4, False))
        self.assertEqual(copy.deepcopy(mine).updates, mine.updates)

    def test_python_subclass_keeps_its_attributes(self):
        es = Annotated(c_maes.Settings(3, seed=1))
        es.note = "a run"
        copied = resumed(es)
        self.assertIs(type(copied), Annotated)
        self.assertEqual(copied.note, "a run")

    def test_python_sampler_cannot_be_pickled(self):
        es = make(d=2)
        es.p.sampler = c_maes.sampling.PySampler(2, lambda: 0.5)
        with self.assertRaises(TypeError) as context:
            pickle.dumps(es)
        self.assertIn("cannot save a sampler", str(context.exception))

    def test_settings_and_values(self):
        settings = c_maes.Settings(
            3,
            x0=np.array([1.0, 2.0, 3.0]),
            lb=np.array([-4.0, -5.0, -6.0]),
            integer_variables=[1],
            sigma0=0.3,
            seed=11,
            cs=0.4,
        )
        settings.C0 = np.diag([1.0, 2.0, 3.0])
        copied = resumed(settings)
        for name in ("dim", "x0", "lb", "ub", "integer_variables", "sigma0", "seed", "cs",
                     "C0", "lambda0", "mu0", "budget", "target", "one_plus_one"):
            np.testing.assert_array_equal(getattr(copied, name), getattr(settings, name))
        self.assertEqual(copied.__getstate__(), settings.__getstate__())
        self.assertEqual(resumed(c_maes.Modules()).__getstate__(), c_maes.Modules().__getstate__())

    def test_samplers_on_their_own(self):
        # A sampler continues its sequence (the global generator is not part of its state)
        for sampler in (c_maes.sampling.Sobol(3), c_maes.sampling.Halton(3),
                        c_maes.sampling.FastGaussian(3),
                        c_maes.sampling.Orthogonal(c_maes.sampling.GaussianTransformer(
                            c_maes.sampling.Halton(3)), 5),
                        c_maes.sampling.Mirrored(c_maes.sampling.UniformScaler(
                            c_maes.sampling.Sobol(3)))):
            with self.subTest(sampler=type(sampler).__name__):
                for _ in range(7):
                    sampler()
                copied = resumed(sampler)
                for _ in range(20):
                    np.testing.assert_array_equal(sampler(), copied())
        # A sampler created on its own draws from the global generator, also when it is loaded
        c_maes.set_seed(5)
        x = c_maes.sampling.Uniform(3)()
        sampler = resumed(c_maes.sampling.Uniform(3))
        c_maes.set_seed(5)
        np.testing.assert_array_equal(sampler(), x)


class TestStateErrors(unittest.TestCase):
    def setUp(self):
        self.state = make(d=2).__getstate__()

    def load(self, state):
        es = c_maes.ModularCMAES.__new__(c_maes.ModularCMAES)
        es.__setstate__(state)
        return es

    def test_format(self):
        # The magic number, the format version, the kind of object, ..., the CRC-32 of zlib
        self.assertEqual(self.state[:8], b"MODCMAST")
        self.assertEqual(int.from_bytes(self.state[8:12], "little"), 1)
        self.assertEqual(self.state[12:20], len("ModularCMAES").to_bytes(8, "little"))
        self.assertEqual(self.state[20:32], b"ModularCMAES")
        self.assertEqual(zlib.crc32(self.state[:-4]), int.from_bytes(self.state[-4:], "little"))
        self.load(self.state)

    def test_newer_format(self):
        newer = self.state[:8] + (2).to_bytes(4, "little") + self.state[12:]
        with self.assertRaisesRegex(ValueError, "newer modcma, in format version 2"):
            self.load(newer)

    def test_corrupted(self):
        for i in (40, len(self.state) // 2, len(self.state) - 1):
            corrupted = bytearray(self.state)
            corrupted[i] ^= 0x10
            with self.subTest(byte=i), self.assertRaisesRegex(ValueError, "checksum"):
                self.load(bytes(corrupted))

    def test_truncated(self):
        for n in (0, 5, 12, 30, len(self.state) // 2, len(self.state) - 1):
            with self.subTest(length=n), self.assertRaisesRegex(ValueError, "modcma state"):
                self.load(self.state[:n])

    def test_not_a_state(self):
        with self.assertRaisesRegex(ValueError, "not a modcma state"):
            self.load(b"an objective value of 3.0")
        with self.assertRaises(TypeError):
            self.load("a string")

    def test_another_kind_of_object(self):
        settings = c_maes.Settings.__new__(c_maes.Settings)
        with self.assertRaisesRegex(ValueError, "holds a ModularCMAES, not a Settings"):
            settings.__setstate__(self.state)
        sobol = c_maes.sampling.Sobol.__new__(c_maes.sampling.Sobol)
        with self.assertRaisesRegex(ValueError, "module of another type"):
            sobol.__setstate__(c_maes.sampling.Halton(2).__getstate__())

    def test_newer_module_type(self):
        # A state with a type the reader does not know (as a newer modcma could write), with a
        # valid checksum
        state = c_maes.mutation.CSA.__new__(c_maes.mutation.CSA)
        csa = make(d=2).p.mutation.__getstate__()
        tag = (3).to_bytes(8, "little")
        body = csa[:-4].replace(tag + b"CSA", tag + b"XYZ", 1)
        with self.assertRaisesRegex(ValueError, "unknown mutation.Strategy type 'XYZ'"):
            state.__setstate__(body + zlib.crc32(body).to_bytes(4, "little"))


class TestResumeGoldenSubset(unittest.TestCase):
    """A run that is pickled and loaded at every step (before every generation, between every ask()
    and tell(), or between the steps of the 1.x API, the drivers taking turns) has the trajectory
    of the run without interruption, bit for bit: the pairwise configurations of the golden fixture,
    which cover every pair of module options, and some of its special cases, with at most 2000
    evaluations (a few seconds). The full set, every driver and the full budgets, against the
    fixture: python -m tests.golden check --drivers resume."""

    SPECIAL = (
        "special/integer_noncontiguous/ellipsoid",
        "special/large_lambda/ellipsoid",
        "special/tight_bounds_cotn/ellipsoid",
        "special/tight_bounds_resample/ellipsoid",
        "special/restart_bipop_elitist/rastrigin",
        "special/repelling_adaptive_archived_cholesky/rastrigin",
        "special/tpa_threshold_sequential/ellipsoid",
        "special/pairwise_sequential/ellipsoid",
        "special/orthogonal_mirrored_halton/ellipsoid",
        "special/fast_sampling_mirrored_orthogonal/ellipsoid",
        "constants/box_muller/ellipsoid",
        "constants/cache_samples/ellipsoid",
        "constants/box_muller_cotn/ellipsoid",
    )

    def test_resumed_runs_match_the_uninterrupted_runs(self):
        runs = [
            dict(run, budget=min(run["budget"], 2000)) for run in golden.make_runs()
            if run["id"].startswith("pair/") or run["id"].rsplit("/", 1)[0] in self.SPECIAL
        ]
        self.assertGreater(len(runs), 40)
        try:
            for i, run in enumerate(runs):
                resume = golden.RESUME_DRIVERS[i % len(golden.RESUME_DRIVERS)]
                with self.subTest(run=run["id"], driver=resume):
                    expected = golden.execute(run, "step")
                    self.assertEqual(golden.compare(expected, golden.execute(run, resume)), "")
        finally:
            for name, value in golden.DEFAULT_CONSTANTS.items():
                setattr(c_maes.constants, name, value)


if __name__ == "__main__":
    unittest.main()
