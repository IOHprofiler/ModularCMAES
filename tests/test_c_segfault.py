"""Module containing tests for ModularCMA-ES C++ Bounds."""

import unittest
import numpy as np

from modcma.c_maes import settings_from_dict, ModularCMAES


def sphere(x):
    return np.linalg.norm(x)


def always_inf(x):
    return np.inf


class TestSeqfault(unittest.TestCase):
    """Test case for Bounds of Modular C++ CMA-ES."""

    def test_sa_large_mu_remains_finite(self):
        settings = settings_from_dict(
            5,
            lambda0=500,
            mu0=125,
            sigma0=2.0,
            ssa="SA",
            target=0.0,
        )

        cma = ModularCMAES(settings)
        cma.step(sphere)

        self.assertTrue(np.isfinite(cma.p.mutation.sigma))
        self.assertGreater(cma.p.mutation.sigma, 1.0)

    def test_nonfinite_restart_never_archives_invalid_solution(self):
        centers = (
            "UNIFORM",
            "NOVELTY_WEIGHTED",
            "MAXIMIN_TABOO",
        )
        repelling_types = (
            "COVERAGE",
            "ADAPTIVE",
        )

        for center in centers:
            for repelling in repelling_types:
                with self.subTest(center=center, repelling=repelling):
                    settings = settings_from_dict(
                        5,
                        lambda0=20,
                        mu0=10,
                        repelling_type=repelling,
                        center_placement=center,
                        restart_strategy="RESTART",
                    )

                    cma = ModularCMAES(settings)
                    cma.p.perform_restart(always_inf)

                    self.assertEqual(len(cma.p.repelling.archive), 0)


if __name__ == "__main__":
    unittest.main()
