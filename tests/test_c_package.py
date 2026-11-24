"""The top-level package: the C++ API under `modcma`, and the names of modcma 1.x."""

import unittest

import numpy as np

import modcma
from modcma import c_maes


class TestPackage(unittest.TestCase):
    def test_top_level_is_c_maes(self):
        for name in modcma.__all__:
            if name != "c_maes":
                self.assertIs(getattr(modcma, name), getattr(c_maes, name), name)
        self.assertIs(modcma.c_maes, c_maes)

    def test_fmin(self):
        xopt, fopt, evaluations, es = modcma.fmin(
            lambda x: float(np.sum(x * x)), np.ones(3), 0.5, 2000, target=1e-8, seed=1
        )
        self.assertLess(fopt, 1e-8)
        self.assertLessEqual(evaluations, 2000)
        self.assertIsInstance(es, modcma.ModularCMAES)

    def test_fmin_starts_at_x0(self):
        start = []

        def record(x):
            if not start:
                start.append(np.array(x))
            return float(np.sum((x - 3) ** 2))

        modcma.fmin(record, [3.0, 3.0, 3.0], 1e-6, 50, seed=1)
        np.testing.assert_allclose(start[0], 3.0, atol=1e-4)

    def test_version(self):
        self.assertRegex(modcma.__version__, r"^\d+\.\d+")

    def test_signature_of_1x_raises(self):
        with self.assertRaises(TypeError) as context:
            modcma.ModularCMAES(sum, 2)
        self.assertIn("modcma<2", str(context.exception))
        with self.assertRaises(TypeError):
            modcma.ModularCMAES(sum, d=2, budget=100)

    def test_removed_names(self):
        for name in ("AskTellCMAES", "evaluate_bbob", "BIPOPParameters"):
            with self.assertRaises(AttributeError) as context:
                getattr(modcma, name)
            self.assertIn("modcma<2", str(context.exception))
        with self.assertRaises(AttributeError):
            modcma.does_not_exist  # noqa: B018

    def test_modules_str_names_every_option(self):
        for name, choices in modcma.get_all_module_options().items():
            default = getattr(modcma.Modules(), name)
            if isinstance(default, bool):
                continue
            for choice in choices:
                modules = modcma.Modules()
                setattr(modules, name, getattr(type(default), choice))
                self.assertIn(f"{name}: {choice}", str(modules))

    def test_constructors_still_work(self):
        self.assertEqual(modcma.ModularCMAES(3).p.settings.dim, 3)
        self.assertEqual(modcma.ModularCMAES(modcma.Settings(4)).p.settings.dim, 4)


if __name__ == "__main__":
    unittest.main()
