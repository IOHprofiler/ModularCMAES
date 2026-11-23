"""Semantics of the Python API of the C++ extension that a change of binding library can silently
change (TASKS.md, T1.1). Pinned on the pybind11 bindings; the nanobind port must keep them, or
change a test here deliberately (with a note in the changelog)."""

import enum
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

    def test_other_eigen_members_are_read_only_views(self):
        for get in (
            lambda: self.p.adaptation.C,
            lambda: self.p.adaptation.m,
            lambda: self.p.weights.positive,
            lambda: self.p.settings.lb,
            lambda: self.p.stats.global_best.x,
        ):
            a = get()
            self.assertIsInstance(a, np.ndarray)
            self.assertFalse(a.flags.writeable)
            self.assertTrue(np.shares_memory(a, get()))

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
        # Settings has no integer_variables attribute (only the constructor argument)
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
        # pybind11: the caller must keep a reference (the shared_ptr in C++ does not keep the
        # Python part alive: 'Tried to call pure virtual function'). nanobind should lift this.
        for _ in range(8):
            cma.step(sphere)
        item = cma.p.criteria.items[-1]
        self.assertIsInstance(item, AfterGenerations)
        self.assertGreaterEqual(item.calls, 5)
        self.assertEqual(len(cma.p.stats.centers), 1)  # one restart, by this criterion
        self.assertIn("AfterGenerations", [c.name for c in cma.p.criteria.items])

    def test_constants_are_module_level_properties(self):
        default = c_maes.constants.use_box_muller
        try:
            c_maes.constants.use_box_muller = not default
            self.assertEqual(c_maes.constants.use_box_muller, not default)
        finally:
            c_maes.constants.use_box_muller = default


if __name__ == "__main__":
    unittest.main()
