"""Warm starting (#89): modcma.warm_start and Settings.C0.

warm_start is compared with cmaes' get_warm_start_mgd in two ways: with reference values that
cmaes 0.13.1 computed (tests/data/warm_start/cmaes-0.13.1.json; CI does not install cmaes), and,
when cmaes is installed, on random cases. Regenerate the reference values with cmaes installed:

    python -m tests.test_c_warm_start write
"""

import hashlib
import json
import math
import os
import sys
import unittest

import numpy as np

import modcma
from modcma import c_maes
from modcma.c_maes import ModularCMAES, Settings, restart, settings_from_dict

try:
    import cmaes
except ImportError:  # pragma: no cover - optional
    cmaes = None

REFERENCE = os.path.join(os.path.dirname(__file__), "data", "warm_start", "cmaes-0.13.1.json")

TYPES = [
    "COVARIANCE",
    "COVARIANCE_NO_EIGV",
    "MATRIX",
    "SEPARABLE",
    "CHOLESKY",
    "CMSA",
    "NATURAL_GRADIENT",
]


def random_spd(d, seed=0, condition=100.0):
    """A symmetric positive definite matrix with eigenvalues from 1 to condition, rotated."""
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    C = rotation @ np.diag(np.geomspace(1.0, condition, d)) @ rotation.T
    return (C + C.T) / 2


def ellipsoid(d, seed=1):
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    weights = 10.0 ** (3 * np.arange(d) / (d - 1))
    return lambda x: float(np.sum(weights * (rotation @ x) ** 2))


def reference_cases():
    """The inputs of the reference values: points in [0, 1] (the paper's setting), with ties, and
    a larger scale with every solution selected."""
    rng = np.random.default_rng(2021)
    cases = []
    X = rng.uniform(0, 1, size=(20, 2))
    cases.append((X, np.sum((X - 0.6) ** 2, axis=1), 0.25, 0.1))
    X = rng.uniform(0, 1, size=(50, 5))
    cases.append((X, np.round(np.sum((X - 0.3) ** 2, axis=1), 1), 0.1, 0.1))
    X = rng.uniform(-5, 5, size=(12, 4))
    cases.append((X, np.sum(X**2, axis=1), 1.0, 0.5))
    return [
        {"X": X.tolist(), "f": f.tolist(), "gamma": gamma, "alpha": alpha}
        for X, f, gamma, alpha in cases
    ]


def write_reference():
    cases = reference_cases()
    for case in cases:
        solutions = [(np.array(x), y) for x, y in zip(case["X"], case["f"])]
        mean, sigma, cov = cmaes.get_warm_start_mgd(solutions, case["gamma"], case["alpha"])
        case.update(mean=mean.tolist(), sigma=sigma, cov=cov.tolist())
    os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
    with open(REFERENCE, "w") as f:
        json.dump({"cmaes": cmaes.__version__,
                   "cases": cases}, f, indent=1)
        f.write("\n")


class TestWarmStart(unittest.TestCase):
    def assert_same_as_cmaes(self, result, mean, sigma, cov):
        # 1e-12, relative to the size of the values
        scale = max(1, np.abs(mean).max())
        np.testing.assert_allclose(result["x0"], mean, rtol=0, atol=1e-12 * scale)
        np.testing.assert_allclose(result["sigma0"], sigma, rtol=1e-12, atol=0)
        np.testing.assert_allclose(result["C0"], cov, rtol=0, atol=1e-12 * np.abs(cov).max())

    def test_the_reference_values_of_cmaes(self):
        with open(REFERENCE) as f:
            cases = json.load(f)["cases"]
        self.assertEqual(len(cases), 3)
        for i, case in enumerate(cases):
            with self.subTest(case=i):
                solutions = list(zip(case["X"], case["f"]))
                result = modcma.warm_start(solutions, gamma=case["gamma"], alpha=case["alpha"])
                self.assert_same_as_cmaes(
                    result, np.array(case["mean"]), case["sigma"], np.array(case["cov"])
                )

    @unittest.skipIf(cmaes is None, "cmaes is not installed")
    def test_the_same_as_cmaes(self):
        rng = np.random.default_rng(5)
        for d, n, gamma, alpha, scale in [(2, 30, 0.1, 0.1, 1), (10, 300, 0.2, 0.05, 1),
                                          (20, 100, 0.5, 0.2, 10), (40, 1000, 0.1, 0.1, 1)]:
            with self.subTest(d=d, n=n):
                X = scale * rng.uniform(size=(n, d))
                f = np.sum((X - 0.4 * scale) ** 2, axis=1)
                solutions = [(x, float(y)) for x, y in zip(X, f)]
                with np.errstate(all="ignore"):  # numpy.linalg.det of cmaes, with Accelerate
                    expected = cmaes.get_warm_start_mgd(solutions, gamma=gamma, alpha=alpha)
                self.assert_same_as_cmaes(
                    modcma.warm_start(solutions, gamma=gamma, alpha=alpha), *expected
                )

    def test_the_estimate(self):
        # Nomura et al. (2021), Eqs. 4 and 5, with Sigma = sigma0^2 C0 and det(C0) = 1
        rng = np.random.default_rng(3)
        X = rng.normal(size=(40, 3))
        f = X[:, 0]
        result = modcma.warm_start([(x, y) for x, y in zip(X, f)], gamma=0.25, alpha=0.2)
        top = X[np.argsort(f)[:10]]
        sigma = 0.04 * np.eye(3) + np.cov(top.T, bias=True)
        np.testing.assert_allclose(result["x0"], top.mean(axis=0))
        np.testing.assert_allclose(result["sigma0"] ** 2 * result["C0"], sigma)
        self.assertAlmostEqual(np.linalg.det(result["C0"]), 1.0)
        np.testing.assert_array_equal(result["C0"], result["C0"].T)
        self.assertIsInstance(result["sigma0"], float)

    def test_settings_from_the_result(self):
        X = np.random.default_rng(4).uniform(size=(100, 3))
        result = modcma.warm_start([(x, float(np.sum((x - 0.6) ** 2))) for x in X])
        settings = Settings(3, **result, budget=500)
        np.testing.assert_array_equal(settings.x0, result["x0"])
        self.assertEqual(settings.sigma0, result["sigma0"])
        np.testing.assert_array_equal(settings.C0, result["C0"])
        # fmin and settings_from_dict pass it on
        xopt, fopt, _, es = modcma.fmin(lambda x: float(np.sum((x - 0.6) ** 2)), budget=500,
                                        seed=1, **result)
        np.testing.assert_array_equal(es.p.settings.C0, result["C0"])
        self.assertLess(fopt, 1e-6)

    def test_ties_keep_their_order_and_nan_is_the_worst(self):
        solutions = [([0.0], 1.0), ([1.0], float("nan")), ([2.0], 1.0), ([3.0], 0.0)]
        self.assertEqual(modcma.warm_start(solutions, gamma=0.5, alpha=1)["x0"][0], 1.5)
        self.assertEqual(modcma.warm_start(solutions, gamma=0.75, alpha=1)["x0"][0], 5 / 3)

    def test_invalid_input(self):
        good = [([0.0, 1.0], 1.0)] * 10
        for kwargs, message in [
            (dict(solutions=[]), "at least one"),
            (dict(solutions=good, gamma=0.0), "gamma"),
            (dict(solutions=good, gamma=1.5), "gamma"),
            (dict(solutions=good, gamma=0.05), "selects no solution"),
            (dict(solutions=good, alpha=-1.0), "alpha"),
            (dict(solutions=good + [([0.0], 1.0)]), "same length"),
            (dict(solutions=good + [([0.0, math.inf], 1.0)]), "finite"),
            # alpha = 0 and fewer solutions than d + 1: Sigma is singular
            (dict(solutions=good, alpha=0.0, gamma=1.0), "positive definite"),
        ]:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    modcma.warm_start(**kwargs)


class TestC0(unittest.TestCase):
    def test_every_adaptation_starts_from_C0(self):
        # The sampling factor F (from compute_y) has F F^T = C0, and metric() reports C0 at
        # generation 0, before and after the first generation is sampled
        d = 6
        C0 = random_spd(d)
        for name in TYPES:
            with self.subTest(name):
                es = ModularCMAES(settings_from_dict(d, matrix_adaptation=name, C0=C0, seed=1))
                expected = np.diag(np.diag(C0)) if name == "SEPARABLE" else C0
                adaptation = es.p.adaptation
                for _ in range(2):
                    factor = np.column_stack([adaptation.compute_y(e) for e in np.eye(d)])
                    np.testing.assert_allclose(factor @ factor.T, expected, rtol=0, atol=1e-10)
                    metric = adaptation.metric()
                    vectors = np.eye(d) if metric.B is None else metric.B
                    np.testing.assert_allclose(vectors @ np.diag(metric.eigenvalues()) @ vectors.T,
                                               expected, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(adaptation.coordinate_wise_variances, np.diag(C0))
                    es.ask()  # the first generation is sampled from it

    def test_the_lapack_solvers(self):
        # The test above runs below constants.lapack_min_dim, with Eigen; here the eigensolvers and
        # the Cholesky factorization of LAPACK start the matrix
        if c_maes.build_info()["lapack"] == "none":
            self.skipTest("LAPACK is not available")
        d = 12
        C0 = random_spd(d)
        try:
            c_maes.constants.cholesky_solver = 1
            for solver in (1, 2):  # dsyevd, dsyevr
                c_maes.constants.eigen_solver = solver
                for name in TYPES:
                    with self.subTest(solver=solver, adaptation=name):
                        es = ModularCMAES(settings_from_dict(d, matrix_adaptation=name, C0=C0))
                        adaptation = es.p.adaptation
                        factor = np.column_stack([adaptation.compute_y(e) for e in np.eye(d)])
                        expected = np.diag(np.diag(C0)) if name == "SEPARABLE" else C0
                        np.testing.assert_allclose(factor @ factor.T, expected, rtol=0, atol=1e-10)
        finally:
            c_maes.constants.eigen_solver = -1
            c_maes.constants.cholesky_solver = -1

    def test_the_first_generation_is_sampled_from_C0(self):
        # Many samples of generation 0: their covariance is sigma0^2 C0
        d, n = 3, 20_000
        C0 = random_spd(d, condition=10)
        for name in TYPES:
            with self.subTest(name):
                es = ModularCMAES(settings_from_dict(d, matrix_adaptation=name, C0=C0, seed=1,
                                                     lambda0=n, mu0=n // 2, sigma0=0.5,
                                                     x0=np.ones(d), budget=n + 1))
                X = es.ask()
                expected = np.diag(np.diag(C0)) if name == "SEPARABLE" else C0
                np.testing.assert_allclose(np.cov(X.T), 0.25 * expected, rtol=0,
                                           atol=0.03 * 0.25 * np.abs(expected).max())

    def test_none_rejects_C0(self):
        with self.assertRaisesRegex(ValueError, "NONE"):
            settings_from_dict(3, matrix_adaptation="NONE", C0=np.eye(3))
        settings = settings_from_dict(3, matrix_adaptation="NONE")
        with self.assertRaisesRegex(ValueError, "NONE"):
            settings.C0 = np.eye(3)
        # The modules changed after C0 was set: the optimizer rejects it
        settings = Settings(3, C0=np.eye(3))
        settings.modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.NONE
        with self.assertRaisesRegex(ValueError, "NONE"):
            ModularCMAES(settings)

    def test_validation(self):
        d = 3
        C0 = random_spd(d)
        for value, message in [
            (np.eye(2), r"\(dim, dim\) = \(3, 3\) matrix, got \(2, 2\)"),
            (np.ones(3), "2-d"),
            (np.full((d, d), np.nan), "finite"),
            (np.triu(C0), "symmetric"),
            (-np.eye(d), "positive definite, its smallest eigenvalue is -1"),
            (np.ones((d, d)), "positive definite"),
            (np.diag([1.0, 0.0, 1.0]), "positive definite"),
        ]:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    Settings(d, C0=value)
                settings = Settings(d)
                with self.assertRaisesRegex(ValueError, message):
                    settings.C0 = value
                self.assertIsNone(settings.C0)

    def test_symmetric_up_to_rounding(self):
        # A product such as R D R^T is symmetric only up to rounding: accepted, and its symmetric
        # part is stored
        d = 5
        rotation, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(d, d)))
        C0 = rotation @ np.diag(np.arange(1.0, d + 1)) @ rotation.T
        settings = Settings(d, C0=C0.tolist())
        np.testing.assert_array_equal(settings.C0, (C0 + C0.T) / 2)
        np.testing.assert_array_equal(settings.C0, settings.C0.T)
        settings.C0 = None
        self.assertIsNone(settings.C0)
        self.assertIn("C0: None", repr(settings))

    def test_ask_tell_is_run(self):
        # Every driver of a generation gives the same trajectory, bit for bit, with C0 and
        # restarts (which start from C0 again)
        d = 4
        C0 = random_spd(d)
        f = ellipsoid(d)

        def trajectory(name, driver):
            es = ModularCMAES(settings_from_dict(d, matrix_adaptation=name, C0=C0, seed=3,
                                                 restart_strategy="IPOP", budget=3000))
            es.p.criteria.items.append(RestartEvery(15))
            digest = hashlib.sha256()
            while True:
                if driver == "step":
                    running = es.step(f)
                elif driver == "vectorized":
                    running = es.step(lambda X: [f(x) for x in X], vectorized=True)
                else:
                    while True:
                        X = es.ask()
                        values = [f(x) for x in X]
                        told = es.tell(values, X) if driver == "asktell_points" else es.tell(values)
                        if told:
                            break
                    running = not es.break_conditions()
                for array in (es.p.pop.X, es.p.pop.f, es.p.adaptation.m):
                    digest.update(np.ascontiguousarray(array).tobytes())
                digest.update(np.float64(es.p.mutation.sigma).tobytes())
                if not running:
                    break
            self.assertGreater(len(es.p.stats.centers), 2)  # restarts happened
            return digest.hexdigest(), es.p.stats.evaluations

        for name in TYPES:
            with self.subTest(name):
                expected = trajectory(name, "step")
                for driver in ("asktell", "asktell_points", "vectorized"):
                    self.assertEqual(trajectory(name, driver), expected, driver)

    def test_a_restart_starts_from_C0(self):
        d = 4
        C0 = random_spd(d)
        for name in TYPES:
            with self.subTest(name):
                es = ModularCMAES(settings_from_dict(d, matrix_adaptation=name, C0=C0, seed=1,
                                                     restart_strategy="RESTART", budget=2000))
                es.p.criteria.items.append(RestartEvery(10))
                expected = np.diag(np.diag(C0)) if name == "SEPARABLE" else C0
                checked = 0
                f = ellipsoid(d)
                while not es.break_conditions():
                    X = es.ask()
                    if len(X) > 1 and es.p.stats.t_run == 0 and len(es.p.stats.centers) > 0:
                        # The first generation of a new run: sampled from C0 again
                        factor = np.column_stack([es.p.adaptation.compute_y(e)
                                                  for e in np.eye(d)])
                        np.testing.assert_allclose(factor @ factor.T, expected, rtol=0,
                                                   atol=1e-10)
                        checked += 1
                    es.tell([f(x) for x in X])
                self.assertGreater(checked, 1)

    def test_identity_is_the_default(self):
        # C0 = I gives the trajectory of no C0, bit for bit, where the factor of I is exactly I
        f = ellipsoid(4)
        for name in ["COVARIANCE_NO_EIGV", "SEPARABLE", "CHOLESKY", "CMSA"]:
            with self.subTest(name):
                runs = []
                for C0 in (None, np.eye(4)):
                    es = ModularCMAES(settings_from_dict(4, matrix_adaptation=name, C0=C0,
                                                         seed=2, budget=1000))
                    es.run(f)
                    runs.append((es.p.stats.evaluations, es.p.stats.global_best.y,
                                 es.p.adaptation.m.tolist()))
                self.assertEqual(runs[0], runs[1])


class RestartEvery(restart.Criterion):
    """Met every `period` generations"""

    def __init__(self, period):
        super().__init__("RestartEvery")
        self.period = period

    def update(self, parameters):
        self.met = parameters.stats.t % self.period == self.period - 1


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        if cmaes is None:
            sys.exit("python -m tests.test_c_warm_start write: needs cmaes (pip install cmaes)")
        write_reference()
        print(f"wrote {REFERENCE}")
    else:
        unittest.main()
