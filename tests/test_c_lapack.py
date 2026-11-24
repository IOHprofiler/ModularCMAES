"""LAPACK for the eigendecomposition (PERFORMANCE.md): linked (macOS: Accelerate) or scipy's."""

import unittest

import numpy as np

from modcma import c_maes


class TestLapack(unittest.TestCase):
    def test_source(self):
        # macOS builds link Accelerate; the others get scipy's LAPACK at import
        source = c_maes.utils.build_info()["lapack"]
        self.assertIn(source, ("linked", "scipy"))
        self.assertEqual(c_maes.constants.eigen_solver, -1)  # automatic
        self.assertEqual(c_maes.constants.cholesky_solver, -1)
        self.assertEqual(c_maes.constants.blas, -1)

    def test_eigendecomposition(self):
        rng = np.random.default_rng(1)
        for n in (1, 5, 40, 130):
            A = rng.normal(size=(n, n))
            C = A @ A.T + np.eye(n)
            expected = np.linalg.eigvalsh(C)
            for method in (1, 2):
                with self.subTest(n=n, method=method):
                    w, V = c_maes.utils.lapack_eigh(C, method)
                    np.testing.assert_allclose(w, expected, rtol=1e-10, atol=1e-10)
                    np.testing.assert_allclose(V @ np.diag(w) @ V.T, C, rtol=1e-10, atol=1e-9)
                    np.testing.assert_allclose(V.T @ V, np.eye(n), atol=1e-10)

    def test_cholesky(self):
        rng = np.random.default_rng(2)
        for n in (1, 6, 130):
            A = rng.normal(size=(n, n))
            C = A @ A.T + np.eye(n)
            L = c_maes.utils.lapack_cholesky(C)
            np.testing.assert_allclose(L, np.linalg.cholesky(C), rtol=1e-10, atol=1e-10)
            self.assertTrue(np.all(np.triu(L, 1) == 0))
        with self.assertRaises(RuntimeError):
            c_maes.utils.lapack_cholesky(-np.eye(3))

    def test_cholesky_adaptations_with_lapack(self):
        # The solvers give the same factor up to rounding, but the trajectories diverge once the
        # last bits differ (e.g. OpenBLAS): compare what they reach, not the values themselves
        d = 30
        w = 10.0 ** (6.0 * np.arange(d) / (d - 1))
        try:
            for name, reaches in (("COVARIANCE_NO_EIGV", 1e-8), ("CMSA", 1.0)):
                for solver in (0, 1):
                    with self.subTest(matrix_adaptation=name, cholesky_solver=solver):
                        c_maes.constants.cholesky_solver = solver
                        modules = c_maes.Modules()
                        modules.matrix_adaptation = getattr(
                            c_maes.options.MatrixAdaptationType, name
                        )
                        settings = c_maes.Settings(d, modules, budget=80_000, seed=4)
                        settings.target = 1e-8
                        cma = c_maes.ModularCMAES(settings)
                        cma.run(lambda X: (X * X) @ w, vectorized=True)
                        # from f ~ 1e6 at the start; CMSA learns slowly on this problem
                        self.assertLess(cma.p.stats.global_best.y, reaches)
        finally:
            c_maes.constants.cholesky_solver = -1

    def test_inverse_and_products(self):
        rng = np.random.default_rng(3)
        for n in (1, 7, 64):
            M = rng.normal(size=(n, n)) + n * np.eye(n)
            np.testing.assert_allclose(c_maes.utils.lapack_inverse(M) @ M, np.eye(n), atol=1e-10)
            B = rng.normal(size=(n, 5))
            np.testing.assert_allclose(c_maes.utils.blas_gemm(M, B), M @ B, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                c_maes.utils.blas_gemm(B, M, transpose_a=True), B.T @ M, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                c_maes.utils.blas_gemm(B, B, transpose_b=True), B @ B.T, rtol=1e-12, atol=1e-12
            )
        with self.assertRaises(RuntimeError):
            c_maes.utils.lapack_inverse(np.zeros((3, 3)))

    def test_blas_in_the_adaptations(self):
        # BLAS products and the LU inverse (forced on, d = 40) reach what Eigen reaches; MATRIX
        # with tight bounds and saturation inverts M after every update
        d = 40
        w = 10.0 ** (6.0 * np.arange(d) / (d - 1))
        cases = [
            ("COVARIANCE", {}, 1e-8),
            ("MATRIX", {"bound_correction": "SATURATE"}, 1e-8),
            ("CHOLESKY", {}, 1e-8),
            ("NATURAL_GRADIENT", {}, 1e-8),
            ("COVARIANCE_NO_EIGV", {"active": True}, 1e-8),
            ("CMSA", {}, 1e2),
        ]
        try:
            for name, extra, reaches in cases:
                for blas in (0, 1):
                    with self.subTest(matrix_adaptation=name, blas=blas):
                        c_maes.constants.blas = blas
                        modules = c_maes.Modules()
                        modules.matrix_adaptation = getattr(
                            c_maes.options.MatrixAdaptationType, name
                        )
                        for option, value in extra.items():
                            self.assertTrue(c_maes.set_module(modules, option, value))
                        settings = c_maes.Settings(
                            d, modules, budget=200_000, seed=7,
                            lb=np.full(d, -3.0), ub=np.full(d, 3.0),
                        )
                        settings.target = 1e-8
                        cma = c_maes.ModularCMAES(settings)
                        cma.run(lambda X: (X * X) @ w, vectorized=True)
                        self.assertLess(cma.p.stats.global_best.y, reaches)
        finally:
            c_maes.constants.blas = -1

    def test_optimizer_with_lapack(self):
        d = 30
        w = 10.0 ** (6.0 * np.arange(d) / (d - 1))
        try:
            results = {}
            for solver in (0, 1, 2):
                c_maes.constants.eigen_solver = solver
                cma = c_maes.ModularCMAES(c_maes.Settings(d, budget=60_000, seed=4))
                cma.run(lambda X: (X * X) @ w, vectorized=True)
                results[solver] = cma.p.stats.global_best.y
            for solver, y in results.items():
                self.assertLess(y, 1e-6, f"eigen_solver {solver}")
        finally:
            c_maes.constants.eigen_solver = -1

    def test_invalid_solver(self):
        with self.assertRaises(ValueError):
            c_maes.constants.eigen_solver = 3
        self.assertEqual(c_maes.constants.eigen_solver, -1)


if __name__ == "__main__":
    unittest.main()
