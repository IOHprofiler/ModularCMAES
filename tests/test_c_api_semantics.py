"""Semantics of the Python API of the C++ extension that a change of binding library can silently
change (dev/overhaul-2.0.md, T1.1). Pinned on the pybind11 bindings; the nanobind port must keep them, or
change a test here deliberately (with a note in the changelog)."""

import enum
import gc
import unittest

import numpy as np

from modcma import c_maes
from modcma.c_maes import cmaescpp  # noqa: F401  (c_maes.cmaescpp)


def sphere(x):
    return float(x @ x)


class TestArraySemantics(unittest.TestCase):
    def setUp(self):
        self.cma = c_maes.ModularCMAES(c_maes.Settings(4, verbose=False))
        self.cma.step(sphere)
        self.p = self.cma.p

    def test_population_arrays_are_writable_views(self):
        for name in ("X", "X_transformed", "Y", "Z", "S"):
            a = getattr(self.p.pop, name)
            self.assertIsInstance(a, np.ndarray)
            self.assertEqual(a.shape, (4, self.p.pop.n), name)
            a[0, 0] = 123.0
            self.assertEqual(getattr(self.p.pop, name)[0, 0], 123.0, name)
        f = self.p.pop.f
        f[0] = -1.0
        self.assertEqual(self.p.pop.f[0], -1.0)

    def test_bounds_are_copies_and_setting_them_updates_the_derived_values(self):
        # lb and ub are returned as copies: writing into a view would leave db, center, ... stale
        settings = c_maes.Settings(3)
        lb = settings.lb
        lb[0] = -1.0
        self.assertEqual(settings.lb[0], -5.0)
        settings.lb = np.array([0.0, 0.0, 0.0])
        settings.ub = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(settings.db, [2.0, 4.0, 6.0])
        np.testing.assert_allclose(settings.center, [1.0, 2.0, 3.0])
        self.assertTrue(settings.has_bounds)
        settings.ub = np.full(3, np.inf)
        self.assertFalse(settings.has_bounds)

    def test_solution_x_is_a_copy(self):
        # #115: a view of current_best.x read freed memory after the next improvement, and
        # global_best.x changed under a stored reference
        x = self.p.stats.global_best.x
        self.assertFalse(np.shares_memory(x, self.p.stats.global_best.x))
        x[...] = 1e9
        self.assertNotEqual(self.p.stats.global_best.x.flat[0], 1e9)

    def test_other_eigen_members_are_writable_views(self):
        # pybind11 returned read-only views; nanobind's are writable and writes reach the C++
        # object (a deliberate change, in the changelog: nothing that worked before breaks)
        for get in (
            lambda: self.p.adaptation.C,
            lambda: self.p.adaptation.m,
            lambda: self.p.weights.positive,
        ):
            a = get()
            self.assertIsInstance(a, np.ndarray)
            self.assertTrue(np.shares_memory(a, get()))
            old = a.flat[0]
            a.flat[0] = old + 1.0
            self.assertEqual(get().flat[0], old + 1.0)
            a.flat[0] = old

    def test_eigen_members_are_assignable(self):
        a = self.p.adaptation
        a.m = np.arange(4.0)
        np.testing.assert_array_equal(a.m, np.arange(4.0))
        a.C = np.eye(4) * 2.0
        np.testing.assert_array_equal(a.C, np.eye(4) * 2.0)

    def test_sampler_returns_new_arrays(self):
        a, b = self.p.sampler(), self.p.sampler()
        self.assertEqual(a.shape, (4,))
        self.assertTrue(a.flags.writeable)
        self.assertFalse(np.shares_memory(a, b))

    def test_objective_receives_a_vector(self):
        seen = []

        def objective(x):
            seen.append((type(x), x.shape, x.dtype))
            return float(x @ x)

        self.cma.step(objective)
        self.assertEqual(seen[0], (np.ndarray, (4,), np.float64))

    def test_objective_may_keep_its_argument(self):
        # The objective gets an array of its own: one stored across evaluations is not
        # overwritten by later samples (no view of the population's buffers)
        kept, values = [], []

        def objective(x):
            kept.append(x)
            values.append(x.copy())
            return float(x @ x)

        for _ in range(3):
            self.cma.step(objective)
        for x, v in zip(kept, values):
            np.testing.assert_array_equal(x, v)

    def test_objective_accepts_numpy_scalars_and_ints(self):
        for value in (np.float64(1.5), np.float32(1.5), 3):
            c_maes.ModularCMAES(c_maes.Settings(2, verbose=False)).step(lambda x, v=value: v)


class TestObjectSemantics(unittest.TestCase):
    def test_shared_objects_keep_identity(self):
        cma = c_maes.ModularCMAES(c_maes.Settings(3, verbose=False))
        self.assertIs(cma.p, cma.p)
        self.assertIs(cma.p.adaptation, cma.p.adaptation)

    def test_polymorphic_objects_have_their_most_derived_type(self):
        modules = c_maes.Modules()
        modules.ssa = c_maes.options.StepSizeAdaptation.TPA
        modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.MATRIX
        p = c_maes.Parameters(c_maes.Settings(3, modules, verbose=False))
        self.assertEqual(type(p.mutation).__name__, "TPA")
        self.assertEqual(type(p.adaptation).__name__, "MatrixAdaptation")
        self.assertIsInstance(p.adaptation, c_maes.cmaescpp.matrix_adaptation.Adaptation)

    def test_enums_are_python_enums(self):
        mirror = c_maes.options.Mirror
        self.assertTrue(issubclass(mirror, enum.Enum))
        self.assertEqual(mirror.PAIRWISE.name, "PAIRWISE")
        self.assertEqual(mirror["MIRRORED"], mirror.MIRRORED)
        self.assertEqual(c_maes.Modules().mirrored, mirror.NONE)

    def test_settings_keywords_and_optionals(self):
        s = c_maes.Settings(3, lambda0=8, mu0=2, sigma0=0.5, x0=np.ones(3), verbose=False)
        self.assertEqual((s.lambda0, s.mu0, s.sigma0), (8, 2, 0.5))
        np.testing.assert_array_equal(s.x0, np.ones(3))
        self.assertIsNone(c_maes.Settings(3).x0)
        self.assertIsNone(c_maes.Settings(3).target)

    def test_integer_variables_accept_lists(self):
        # integer_variables is given as a list of indices
        cma = c_maes.ModularCMAES(c_maes.Settings(4, integer_variables=[1, 3], verbose=False))
        cma.step(sphere)
        np.testing.assert_array_equal(cma.p.pop.X_transformed[1], np.round(cma.p.pop.X[1]))

    def test_python_restart_criterion(self):
        """A Python subclass of restart.Criterion, appended to p.criteria.items (as in
        scripts/competetions/cec26/main.py), is called from C++ and can trigger a restart."""

        class AfterGenerations(c_maes.restart.Criterion):
            def __init__(self, generations):
                super().__init__("AfterGenerations")
                self.generations = generations
                self.calls = 0

            def update(self, par):
                self.calls += 1
                self.met = par.stats.t == self.generations

        modules = c_maes.Modules()
        modules.restart_strategy = c_maes.options.RestartStrategy.RESTART
        cma = c_maes.ModularCMAES(c_maes.Settings(3, modules, verbose=False))
        criterion = AfterGenerations(5)
        cma.p.criteria.items.append(criterion)
        # The vector keeps the Python object alive (with pybind11 the caller had to keep a
        # reference, or the next update raised 'Tried to call pure virtual function')
        del criterion
        gc.collect()
        for _ in range(8):
            cma.step(sphere)
        item = cma.p.criteria.items[-1]
        self.assertIsInstance(item, AfterGenerations)
        self.assertGreaterEqual(item.calls, 5)
        self.assertEqual(len(cma.p.stats.centers), 1)  # one restart, by this criterion
        self.assertIn("AfterGenerations", [c.name for c in cma.p.criteria.items])

    def test_lists_of_structs_are_copies(self):
        """#176: repelling.archive, stats.solutions and stats.centers returned references into the
        C++ vectors. A new entry reallocates the vector (and the archive is sorted before each
        population), so a list held across step() read freed memory, and could segfault; a write
        to an element changed the optimizer. They are copies now, as their docstrings say."""

        def rastrigin(x):
            return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))

        modules = c_maes.Modules()
        modules.restart_strategy = c_maes.options.RestartStrategy.IPOP
        modules.repelling_type = c_maes.options.RepellingType.COVERAGE
        settings = c_maes.Settings(
            3, modules, budget=20_000, lb=np.full(3, -5.0), ub=np.full(3, 5.0), seed=1,
            verbose=False,
        )
        cma = c_maes.ModularCMAES(settings)
        while len(cma.p.repelling.archive) == 0:
            cma.step(rastrigin)
        p = cma.p
        held = (p.repelling.archive, p.stats.solutions, p.stats.centers)
        y = [p.repelling.archive[0].solution.y, p.stats.solutions[0].y, p.stats.centers[0].y]
        # A write to the copy does not reach the optimizer
        held[0][0].n_rep = 99
        held[1][0].y = -1.0
        held[2][0].y = -2.0
        self.assertEqual(p.repelling.archive[0].n_rep, 1)
        self.assertEqual([p.stats.solutions[0].y, p.stats.centers[0].y], y[1:])
        # The held lists stay valid while the vectors grow (restarts, new taboo points)
        n = (len(p.repelling.archive), len(p.stats.solutions))
        while not cma.break_conditions():
            cma.step(rastrigin)
            self.assertEqual(
                (held[0][0].n_rep, held[0][0].solution.y, held[1][0].y, held[2][0].y),
                (99, y[0], -1.0, -2.0),
            )
        self.assertGreater(len(p.repelling.archive), n[0])
        self.assertGreater(len(p.stats.solutions), n[1] + 2)
        self.assertEqual(p.stats.solutions[0].y, y[1])
        # Assigning a list still replaces the vector
        p.repelling.archive = held[0][:1]
        self.assertEqual(len(p.repelling.archive), 1)
        self.assertEqual(p.repelling.archive[0].n_rep, 99)

    def test_constants_are_module_level_properties(self):
        default = c_maes.constants.use_box_muller
        try:
            c_maes.constants.use_box_muller = not default
            self.assertEqual(c_maes.constants.use_box_muller, not default)
        finally:
            c_maes.constants.use_box_muller = default


if __name__ == "__main__":
    unittest.main()
