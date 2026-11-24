"""set_seed and build_info, and the deprecated modcma.c_maes.utils of 1.x."""

import unittest
import warnings

import numpy as np

import modcma
from modcma import c_maes


class TestSeedAndBuildInfo(unittest.TestCase):
    def test_the_names_are_in_both_namespaces(self):
        self.assertIs(modcma.set_seed, c_maes.set_seed)
        self.assertIs(modcma.build_info, c_maes.build_info)
        self.assertIn("set_seed", modcma.__all__)
        self.assertNotIn("utils", modcma.__all__)
        self.assertNotIn("utils", c_maes.__all__)

    def test_set_seed_makes_new_optimizers_reproducible(self):
        def first_points():
            return c_maes.ModularCMAES(c_maes.Settings(4, verbose=False)).ask()

        modcma.set_seed(10)
        a = first_points()
        modcma.set_seed(10)
        np.testing.assert_array_equal(a, first_points())

    def test_build_info(self):
        info = modcma.build_info()
        self.assertEqual(set(info), {"compiler", "fma", "lapack"})
        self.assertIn(info["lapack"], ("linked", "scipy", "none"))


class TestDeprecatedUtils(unittest.TestCase):
    def test_set_seed_and_build_info_warn(self):
        for call in (lambda: c_maes.utils.set_seed(1), c_maes.utils.build_info):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                call()
            self.assertEqual(len(caught), 1)
            self.assertIs(caught[0].category, DeprecationWarning)
            self.assertEqual(caught[0].filename, __file__)  # points at the caller

    def test_the_alias_forwards(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual(c_maes.utils.build_info(), modcma.build_info())
            c_maes.utils.set_seed(5)
        a = c_maes.ModularCMAES(c_maes.Settings(3, verbose=False)).ask()
        modcma.set_seed(5)
        np.testing.assert_array_equal(a, c_maes.ModularCMAES(c_maes.Settings(3, verbose=False)).ask())

    def test_removed_names_say_what_to_use(self):
        for name in ("cdf", "ppf", "random_uniform", "random_normal", "compute_ert", "i8_sobol",
                     "Shuffler", "CachedShuffleSequence", "lapack_eigh", "blas_gemm"):
            with self.assertRaisesRegex(AttributeError, "removed in modcma 2.0"):
                getattr(c_maes.utils, name)
        with self.assertRaisesRegex(AttributeError, "scipy.stats.norm.cdf"):
            c_maes.utils.cdf  # noqa: B018
        with self.assertRaises(AttributeError):
            c_maes.utils.does_not_exist  # noqa: B018


if __name__ == "__main__":
    unittest.main()
