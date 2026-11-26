"""Adaptation.metric(): the shape of the sampled distribution, for every matrix adaptation (#123)."""

import unittest

import numpy as np

from modcma import c_maes
from modcma.c_maes import settings_from_dict

TYPES = [
    "COVARIANCE",
    "COVARIANCE_NO_EIGV",
    "MATRIX",
    "SEPARABLE",
    "CHOLESKY",
    "CMSA",
    "NONE",
    "NATURAL_GRADIENT",
]
DIAGONAL = {"SEPARABLE", "NONE"}


def rotated_ellipsoid(d, seed=0):
    """An ellipsoid with condition 1e3 in a random rotation: C learns a full, non-trivial shape."""
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    weights = 10.0 ** (3 * np.arange(d) / (d - 1))
    return lambda x: float(np.sum(weights * (rotation @ x) ** 2))


def run(matrix_adaptation, d, generations, seed=1):
    es = c_maes.ModularCMAES(
        settings_from_dict(d, matrix_adaptation=matrix_adaptation, seed=seed)
    )
    f = rotated_ellipsoid(d)
    for _ in range(generations):
        es.step(f)
    return es


def sampled_covariance(adaptation, d):
    """C of the distribution that is sampled: F F^T, with the sampling factor F from compute_y."""
    factor = np.column_stack([adaptation.compute_y(e) for e in np.eye(d)])
    return factor @ factor.T


class TestMetric(unittest.TestCase):
    def test_the_metric_is_the_sampled_distribution(self):
        d = 10
        for name in TYPES:
            with self.subTest(name):
                adaptation = run(name, d, 200).p.adaptation
                metric = adaptation.metric()
                s = sampled_covariance(adaptation, d)
                scale = np.linalg.eigvalsh(s).max()

                self.assertEqual(metric.is_diagonal, name in DIAGONAL)
                self.assertEqual(metric.B is None, name in DIAGONAL)
                np.testing.assert_allclose(metric.eigenvalues(), metric.d**2)
                np.testing.assert_allclose(
                    np.sort(metric.eigenvalues()), np.linalg.eigvalsh(s), rtol=0, atol=1e-10 * scale
                )
                vectors = np.eye(d) if metric.B is None else metric.B
                np.testing.assert_allclose(
                    s @ vectors, vectors * metric.eigenvalues(), rtol=0, atol=1e-10 * scale
                )
                np.testing.assert_allclose(metric.inv_C() @ s, np.eye(d), rtol=0, atol=1e-8)
                root = metric.inv_root_C()
                np.testing.assert_allclose(root, root.T)
                np.testing.assert_allclose(root @ s @ root, np.eye(d), rtol=0, atol=1e-8)

    def test_a_full_shape_was_learned(self):
        # The test above is only meaningful when C is far from the identity
        for name in sorted(set(TYPES) - {"NONE"}):
            with self.subTest(name):
                eigenvalues = run(name, 10, 200).p.adaptation.metric().eigenvalues()
                self.assertGreater(eigenvalues.max() / eigenvalues.min(), 10)

    def test_the_lapack_eigensolvers(self):
        # The tests above run below constants.lapack_min_dim, with Eigen's solver
        if c_maes.build_info()["lapack"] == "none":
            self.skipTest("LAPACK is not available")
        d = 30
        try:
            for solver in (1, 2):  # dsyevd, dsyevr
                c_maes.constants.eigen_solver = solver
                for name in ["MATRIX", "COVARIANCE_NO_EIGV"]:
                    with self.subTest(solver=solver, adaptation=name):
                        adaptation = run(name, d, 50).p.adaptation
                        metric = adaptation.metric()
                        s = sampled_covariance(adaptation, d)
                        scale = np.linalg.eigvalsh(s).max()
                        np.testing.assert_allclose(
                            s @ metric.B, metric.B * metric.eigenvalues(), rtol=0, atol=1e-10 * scale
                        )
        finally:
            c_maes.constants.eigen_solver = -1

    def test_the_metric_follows_the_factor_not_c(self):
        # With the default settings, d = 100 decomposes C every second generation (generations
        # 2, 4, ...). After 6 generations the last update of C was not decomposed: the samples,
        # and the metric, still come from the decomposition of generation 4.
        d = 100
        for name in ["COVARIANCE", "COVARIANCE_NO_EIGV"]:
            with self.subTest(name):
                es = run(name, d, 6)
                self.assertGreater(es.p.weights.lazy_update_interval, 1)
                adaptation = es.p.adaptation
                sampled = np.linalg.eigvalsh(sampled_covariance(adaptation, d))
                latest = np.linalg.eigvalsh(adaptation.C)
                metric = np.sort(adaptation.metric().eigenvalues())
                np.testing.assert_allclose(metric, sampled, rtol=0, atol=1e-10 * sampled.max())
                self.assertGreater(np.max(np.abs(latest - sampled)), 1e-6 * sampled.max())

    def test_a_computed_metric_is_reused_until_the_factor_changes(self):
        es = run("MATRIX", 4, 5)
        adaptation = es.p.adaptation
        first = adaptation.metric()
        version = adaptation.factor_version

        # A write from Python does not change the version: the stored metric is reused
        adaptation.M = 2 * adaptation.M
        self.assertEqual(adaptation.factor_version, version)
        np.testing.assert_array_equal(adaptation.metric().d, first.d)

        # An update changes it: the metric is computed again, from the new M
        es.step(rotated_ellipsoid(4))
        self.assertEqual(adaptation.factor_version, version + 1)
        np.testing.assert_allclose(
            np.sort(adaptation.metric().d), np.sort(np.linalg.svd(adaptation.M, compute_uv=False))
        )

        # So does a restart
        adaptation.restart(es.p.settings)
        self.assertEqual(adaptation.factor_version, version + 2)
        np.testing.assert_allclose(adaptation.metric().d, np.ones(4))

    def test_the_views_follow_writes(self):
        # COVARIANCE and SEPARABLE return their own d: a write is seen at once
        for name, attribute in [("COVARIANCE", "d"), ("SEPARABLE", "d")]:
            with self.subTest(name):
                adaptation = run(name, 4, 5).p.adaptation
                setattr(adaptation, attribute, 3 * getattr(adaptation, attribute))
                np.testing.assert_array_equal(adaptation.metric().d, getattr(adaptation, attribute))

    def test_the_metric_is_a_copy(self):
        es = run("COVARIANCE", 4, 5)
        metric = es.p.adaptation.metric()
        d, b = metric.d.copy(), metric.B.copy()
        for _ in range(5):
            es.step(rotated_ellipsoid(4))
        self.assertFalse(np.array_equal(es.p.adaptation.d, d))
        np.testing.assert_array_equal(metric.d, d)
        np.testing.assert_array_equal(metric.B, b)

    def test_diagonal_inverses(self):
        adaptation = run("SEPARABLE", 4, 20).p.adaptation
        metric = adaptation.metric()
        np.testing.assert_allclose(metric.inv_C(), np.diag(1 / adaptation.c), rtol=1e-12)
        np.testing.assert_allclose(metric.inv_root_C(), np.diag(1 / adaptation.d), rtol=1e-12)

    def test_no_adaptation(self):
        metric = c_maes.matrix_adaptation.NoAdaptation(3, np.zeros(3)).metric()
        np.testing.assert_array_equal(metric.d, np.ones(3))
        self.assertIsNone(metric.B)

    def test_a_factor_that_is_not_finite_gives_nan(self):
        adaptation = c_maes.matrix_adaptation.MatrixAdaptation(3, np.zeros(3))
        adaptation.M = np.full((3, 3), np.nan)
        metric = adaptation.metric()
        self.assertTrue(np.all(np.isnan(metric.d)))
        self.assertTrue(np.all(np.isnan(metric.B)))


if __name__ == "__main__":
    unittest.main()
