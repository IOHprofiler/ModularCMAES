"""Gradient injection (#90): ModularCMAES.inject_gradient, run(f, gradient=...), fmin(gradient=...).

Every generation, a point along the negative gradient from the mean, at the length of a typical
sample, replaces individual 0 of the population (Hansen 2011, "Injecting external solutions into
CMA-ES", Eq. 14; libcmaes' gradient injection). Without a gradient the results are unchanged (the
golden fixtures, tests/test_c_golden.py); here: the point, its interplay with the other modules,
every driver gives the same run with a gradient, the errors, and a small version of the BBOB
experiment of the pull request.
"""

import math
import unittest

import numpy as np

import modcma
from modcma import c_maes
from tests import golden


# ---------------------------------------------------------------------------------------------------
# The gradients of the objectives of the golden runs
# ---------------------------------------------------------------------------------------------------


def ellipsoid_weights(n):
    return np.array([10.0 ** (6.0 * i / (n - 1)) if n > 1 else 1.0 for i in range(n)])


def sphere_gradient(x):
    return 2.0 * np.asarray(x)


def ellipsoid_gradient(x):
    x = np.asarray(x)
    return 2.0 * ellipsoid_weights(len(x)) * x


def rosenbrock_gradient(x):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


def rastrigin_gradient(x):
    x = np.asarray(x)
    return 2.0 * x + 20.0 * math.pi * np.sin(2.0 * math.pi * x)


GRADIENTS = {
    "sphere": sphere_gradient,
    "ellipsoid": ellipsoid_gradient,
    "rosenbrock": rosenbrock_gradient,
    "rastrigin": rastrigin_gradient,
}

# ---------------------------------------------------------------------------------------------------
# BBOB f2, f10 and f12 with analytic gradients, after ioh's definitions (IOHexperimenter,
# include/ioh/problem/bbob/ellipsoid.hpp, ellipsoid_rotated.hpp, bent_cigar.hpp, and
# transformation.hpp: oscillate is T_osz, asymmetric T_asy). ioh does not expose its rotations, so
# x_opt and R are drawn here: the same functions, other instances.
# ---------------------------------------------------------------------------------------------------


def t_osz(u):
    """T_osz per coordinate, and its derivative (1 at 0, where T_osz(0) = 0)."""
    u = np.asarray(u, dtype=float)
    t, dt = np.zeros_like(u), np.ones_like(u)
    nz = u != 0
    uh = np.log(np.abs(u[nz]))
    a = np.where(u[nz] > 0, 10.0, 5.5)
    b = np.where(u[nz] > 0, 7.9, 3.1)
    e = np.exp(uh + 0.049 * (np.sin(a * uh) + np.sin(b * uh)))
    t[nz] = np.sign(u[nz]) * e
    dt[nz] = e / np.abs(u[nz]) * (1.0 + 0.049 * (a * np.cos(a * uh) + b * np.cos(b * uh)))
    return t, dt


def t_asy(u, beta=0.5):
    """T_asy^beta per coordinate (exponent 1 + beta i / (d - 1) sqrt(u_i)), and its derivative."""
    u = np.asarray(u, dtype=float)
    c = np.arange(u.size) / (u.size - 1.0)
    t, dt = u.copy(), np.ones_like(u)
    p = u > 0
    s = np.sqrt(u[p])
    e = 1.0 + beta * c[p] * s
    t[p] = np.exp(e * np.log(u[p]))
    dt[p] = t[p] * (beta * c[p] / (2.0 * s) * np.log(u[p]) + e / u[p])
    return t, dt


class BBOB:
    """f2 (separable ellipsoid), f10 (rotated ellipsoid) or f12 (bent cigar), f_opt = 0."""

    def __init__(self, fid, dim, seed):
        rng = np.random.default_rng(seed)
        self.fid = fid
        self.xopt = rng.uniform(-4, 4, dim)
        q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
        self.R = q * np.sign(np.diag(r))
        self.w = 1e6 ** (np.arange(dim) / (dim - 1.0))
        self.cigar = np.full(dim, 1e6)
        self.cigar[0] = 1.0

    def value_and_gradient(self, x):
        x = np.asarray(x, dtype=float)
        if self.fid == 12:
            u = self.R @ (x - self.xopt)
            v, dv = t_asy(u)
            z = self.R @ v
            return float(self.cigar @ (z * z)), self.R.T @ ((self.R.T @ (2.0 * self.cigar * z)) * dv)
        rotated = self.fid == 10
        u = self.R @ (x - self.xopt) if rotated else x - self.xopt
        z, dz = t_osz(u)
        g = 2.0 * self.w * z * dz
        return float(self.w @ (z * z)), self.R.T @ g if rotated else g

    def __call__(self, x):
        return self.value_and_gradient(x)[0]

    def gradient(self, x):
        return self.value_and_gradient(x)[1]


# ---------------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------------


def generation(cma, objective, gradient, driver):
    """One generation with a gradient, by one of the drivers of tests/golden.py."""
    if driver == "step":
        return cma.step(objective, gradient=gradient)
    if driver == "vectorized":
        return cma.step(lambda X: [objective(x) for x in X], vectorized=True, gradient=gradient)
    if driver == "steps":
        cma.mutate(objective, gradient=gradient)
        cma.select()
        cma.recombine()
        cma.adapt()
        return not cma.break_conditions()
    while True:
        X = cma.ask()
        if cma.accepts_gradient:
            cma.inject_gradient(gradient(cma.center_of_mass))
            X = cma.ask()
        f = [objective(x) for x in X]
        if cma.tell(f, X) if driver == "asktell_points" else cma.tell(f):
            return not cma.break_conditions()


def trajectory(run, driver, gradient=None):
    """The states of a golden run, generation by generation, with a gradient (None: the
    gradient of its function)."""
    for name, value in golden.DEFAULT_CONSTANTS.items():
        setattr(c_maes.constants, name, value)
    for name, value in run["constants"].items():
        setattr(c_maes.constants, name, value)
    c_maes.set_seed(golden.SEED)
    cma = c_maes.ModularCMAES(golden.make_settings(run))
    objective = golden.FUNCTIONS[run["function"]]
    gradient = gradient or GRADIENTS[run["function"]]
    states = []
    try:
        running = True
        while running:
            running = generation(cma, objective, gradient, driver)
            p = cma.p
            states.append(
                (
                    p.pop.X.tobytes(),
                    p.pop.f.tobytes(),
                    p.adaptation.m.tobytes(),
                    p.mutation.sigma,
                    p.stats.evaluations,
                )
            )
    except Exception as error:  # the same error with every driver
        states.append(f"{type(error).__name__}: {error}")
    return states


def make(d=5, seed=3, modules=None, **kwargs):
    m = c_maes.Modules()
    for name, value in (modules or {}).items():
        if not c_maes.set_module(m, name, value):
            raise ValueError(name)
    kwargs.setdefault("verbose", False)
    return c_maes.ModularCMAES(c_maes.Settings(d, m, seed=seed, **kwargs))


def covariance(cma):
    """sigma**2 C of the sampled distribution, from its metric"""
    metric = cma.p.adaptation.metric()
    B = np.eye(len(metric.d)) if metric.is_diagonal else metric.B
    return B @ np.diag(metric.d**2) @ B.T


class TestDriversWithGradient(unittest.TestCase):
    """step(), ask-tell (with and without the points), a vectorized objective and the steps of the
    1.x API give the same trajectory with a gradient, bit for bit, in every golden configuration"""

    def tearDown(self):
        for name, value in golden.DEFAULT_CONSTANTS.items():
            setattr(c_maes.constants, name, value)

    def test_every_golden_configuration(self):
        for run in golden.make_runs():
            reference = trajectory(run, "step")
            for driver in golden.DRIVERS[1:]:
                with self.subTest(run=run["id"], driver=driver):
                    self.assertEqual(trajectory(run, driver), reference)


class TestOffWithoutGradient(unittest.TestCase):
    def test_none_and_zero_gradient_change_nothing(self):
        for modules in ({}, {"mirrored": "PAIRWISE", "ssa": "TPA"}, {"elitist": True}):
            with self.subTest(modules=modules):
                runs = []
                for gradient in ("absent", None, lambda x: np.zeros(len(x))):
                    cma = make(modules=modules, budget=1500)
                    if gradient == "absent":
                        cma.run(golden.rastrigin)
                    else:
                        cma.run(golden.rastrigin, gradient=gradient)
                    runs.append(
                        (cma.p.stats.evaluations, cma.p.adaptation.m.tobytes(), cma.p.mutation.sigma)
                    )
                self.assertEqual(runs[0], runs[1])
                self.assertEqual(runs[0], runs[2])


class TestGradientPoint(unittest.TestCase):
    def after(self, generations, d=6, **kwargs):
        """An optimizer whose matrix has adapted for some generations, with a population asked"""
        cma = make(d=d, x0=np.full(d, 2.0), sigma0=0.5, **kwargs)
        for _ in range(generations):
            cma.step(golden.ellipsoid)
        X = cma.ask()
        self.assertTrue(cma.accepts_gradient)
        return cma, X

    def test_point_along_the_metric_gradient(self):
        # x = m - sigma c C g / sqrt(g^T C g), Hansen (2011), Eq. 14 with c = E||z||, for every
        # matrix adaptation
        for adaptation in (
            "COVARIANCE", "SEPARABLE", "MATRIX", "CHOLESKY", "CMSA", "COVARIANCE_NO_EIGV",
            "NATURAL_GRADIENT", "NONE",
        ):
            with self.subTest(adaptation=adaptation):
                cma, X = self.after(25, modules={"matrix_adaptation": adaptation})
                C = covariance(cma)
                g = ellipsoid_gradient(cma.center_of_mass)
                cma.inject_gradient(g)
                self.assertFalse(cma.accepts_gradient)
                Y = cma.ask()
                m, sigma = cma.p.adaptation.m, cma.p.mutation.sigma
                c = cma.p.weights.expected_length_z
                expected = m - sigma * c * C @ g / math.sqrt(g @ C @ g)
                np.testing.assert_allclose(Y[0], expected, rtol=1e-9, atol=1e-12 * sigma)
                np.testing.assert_array_equal(Y[1:], X[1:])
                # its z has the length of a typical sample
                self.assertAlmostEqual(np.linalg.norm(cma.p.pop.Z[:, 0]), c, places=10)
                # f decreases along it at a small enough step: the direction is a descent one
                self.assertLess(g @ (expected - m), 0)

    def test_mirrored_pair(self):
        for mirrored in ("MIRRORED", "PAIRWISE"):
            with self.subTest(mirrored=mirrored):
                cma, X = self.after(5, modules={"mirrored": mirrored})
                cma.inject_gradient(ellipsoid_gradient(cma.center_of_mass))
                Y = cma.ask()
                m = cma.p.adaptation.m
                np.testing.assert_allclose(Y[1] - m, -(Y[0] - m), atol=1e-12)
                np.testing.assert_array_equal(cma.p.pop.Z[:, 1], -cma.p.pop.Z[:, 0])
                np.testing.assert_array_equal(Y[2:], X[2:])

    def test_unpaired_without_mirroring_and_with_lambda_1(self):
        cma, X = self.after(5, lambda0=1)
        cma.inject_gradient(ellipsoid_gradient(cma.center_of_mass))
        self.assertEqual(len(cma.ask()), 1)
        cma, X = self.after(5, modules={"mirrored": "MIRRORED"}, lambda0=1)
        cma.inject_gradient(ellipsoid_gradient(cma.center_of_mass))
        self.assertEqual(len(cma.ask()), 1)

    def test_sampled_step_size(self):
        # The point gets the step size of the individual it replaces
        cma, _ = self.after(5, modules={"ssa": "SA"})
        s = cma.p.pop.S[0, 0]
        self.assertNotEqual(s, cma.p.mutation.sigma)
        g = ellipsoid_gradient(cma.center_of_mass)
        cma.inject_gradient(g)
        C = covariance(cma)
        expected = cma.p.adaptation.m - s * cma.p.weights.expected_length_z * C @ g / math.sqrt(
            g @ C @ g
        )
        np.testing.assert_allclose(cma.ask()[0], expected, rtol=1e-9)

    def test_tpa_points_stay(self):
        cma, X = self.after(5, modules={"ssa": "TPA"})
        self.assertEqual(len(X), cma.p.lamb + 2)
        cma.inject_gradient(ellipsoid_gradient(cma.center_of_mass))
        Y = cma.ask()
        np.testing.assert_array_equal(Y[1:], X[1:])
        self.assertFalse(np.array_equal(Y[0], X[0]))

    def test_sequential_selection_first_batch(self):
        cma, X = self.after(5, modules={"sequential_selection": True})
        self.assertLess(len(X), cma.p.lamb)
        cma.inject_gradient(ellipsoid_gradient(cma.center_of_mass))
        self.assertFalse(np.array_equal(cma.ask()[0], X[0]))
        cma.tell([golden.ellipsoid(x) for x in cma.ask()])
        if cma.has_pending:
            self.assertFalse(cma.accepts_gradient)

    def test_bound_correction(self):
        # A large sigma in a small box: the point along -g leaves it, and is moved back in, as
        # TPA's points; then it enters the update as an injected point (its z from x)
        lb, ub = np.full(5, -1.0), np.full(5, 0.5)
        for method in ("SATURATE", "MIRROR", "COTN", "TOROIDAL", "UNIFORM_RESAMPLE", "RESAMPLE"):
            with self.subTest(method=method):
                cma = make(modules={"bound_correction": method}, lb=lb, ub=ub, sigma0=3.0,
                           x0=np.full(5, 0.4))
                cma.ask()
                g = np.ones(5)
                cma.inject_gradient(g)
                x = cma.ask()[0]
                self.assertTrue(np.all(x >= lb) and np.all(x <= ub))
                if method in ("SATURATE", "RESAMPLE"):
                    np.testing.assert_array_equal(x, lb)
                # y and z are those of the corrected point (clipped if far outside)
                p = cma.p
                y = (p.pop.X[:, 0] - p.adaptation.m) / p.pop.S[:, 0]
                np.testing.assert_allclose(p.pop.Y[:, 0], y, atol=1e-12)
                self.assertLessEqual(
                    np.linalg.norm(p.pop.Z[:, 0]), math.sqrt(5) + 2 * 5 / 7 + 1e-12
                )

    def test_integer_variables_rounded(self):
        cma = make(integer_variables=[0, 2], x0=np.full(5, 2.0), sigma0=1.0)
        cma.ask()
        cma.inject_gradient(np.array([1.0, 2.0, -1.0, 0.5, 3.0]))
        x = cma.ask()[0]
        self.assertEqual(x[0], round(x[0]))
        self.assertEqual(x[2], round(x[2]))

    def test_tell_with_points_replaces_the_gradient_point(self):
        cma, X = self.after(3)
        cma.inject_gradient(ellipsoid_gradient(cma.center_of_mass))
        Y = cma.ask()
        Y[0] = np.zeros(len(Y[0]))
        cma.tell([golden.ellipsoid(x) for x in Y], Y)
        self.assertEqual(cma.p.stats.global_best.y, 0.0)


class TestWhenAGradientIsAccepted(unittest.TestCase):
    def test_states(self):
        cma = make()
        self.assertFalse(cma.accepts_gradient)  # nothing asked
        with self.assertRaises(RuntimeError):
            cma.inject_gradient(np.ones(5))
        cma.ask()
        self.assertTrue(cma.accepts_gradient)
        with self.assertRaises(ValueError):
            cma.inject_gradient(np.ones(4))
        with self.assertRaises(ValueError):
            cma.inject_gradient([1.0, np.nan, 0.0, 0.0, 0.0])
        self.assertTrue(cma.accepts_gradient)  # a refused gradient changes nothing
        cma.inject_gradient(np.zeros(5))  # a zero gradient replaces nothing, but counts
        self.assertFalse(cma.accepts_gradient)
        with self.assertRaises(RuntimeError):
            cma.inject_gradient(np.ones(5))
        cma.tell([golden.sphere(x) for x in cma.ask()])
        self.assertFalse(cma.accepts_gradient)
        cma.ask()
        self.assertTrue(cma.accepts_gradient)
        cma.tell([golden.sphere(x) for x in cma.ask()])
        self.assertFalse(cma.accepts_gradient)

    def test_not_for_the_center_of_mass(self):
        cma = make(budget=200)
        while not cma.break_conditions():
            X = cma.ask()
            if len(X) == 1:  # the final center of mass
                self.assertFalse(cma.accepts_gradient)
            cma.tell([golden.sphere(x) for x in X])

    def test_gradient_called_once_per_population_at_the_center_of_mass(self):
        for modules in ({}, {"sequential_selection": True}, {"restart_strategy": "IPOP"}):
            with self.subTest(modules=modules):
                cma = make(modules=modules, budget=3000)
                points = []

                def gradient(x):
                    points.append(np.array(x))
                    np.testing.assert_array_equal(x, cma.center_of_mass)
                    return rastrigin_gradient(x)

                cma.run(golden.rastrigin, gradient=gradient)
                self.assertEqual(len(points), cma.p.stats.t)

    def test_not_callable(self):
        cma = make()
        with self.assertRaises(TypeError):
            cma.run(golden.sphere, gradient=1.0)

    def test_gradient_may_return_a_list(self):
        cma = make(budget=500)
        cma.run(golden.sphere, gradient=lambda x: [2.0 * v for v in x])
        self.assertGreater(cma.p.stats.evaluations, 0)


class TestFmin(unittest.TestCase):
    def test_fmin_with_gradient(self):
        calls = []

        def gradient(x):
            calls.append(1)
            return 2.0 * x

        xopt, fopt, evals, es = modcma.fmin(
            lambda x: float(np.sum(x**2)), np.ones(5), 0.5, 2000, gradient=gradient, seed=1
        )
        self.assertEqual(len(calls), es.p.stats.t)
        self.assertLess(fopt, 1e-8)


class TestBBOB(unittest.TestCase):
    """A small version of the experiment of #90 (BBOB f2, f10, f12 at d = 10, 40 and 100)"""

    def test_gradients_against_finite_differences(self):
        rng = np.random.default_rng(1)
        for fid in (2, 10, 12):
            for d in (5, 20):
                problem = BBOB(fid, d, 7)
                for scale in (3.0, 0.1):
                    x = problem.xopt + scale * rng.uniform(-1, 1, d)
                    g = problem.gradient(x)
                    h = 1e-7
                    fd = np.array(
                        [
                            (problem(x + h * e) - problem(x - h * e)) / (2 * h)
                            for e in np.eye(d)
                        ]
                    )
                    with self.subTest(fid=fid, d=d, scale=scale):
                        self.assertLess(np.max(np.abs(g - fd)), 1e-5 * np.max(np.abs(fd)))

    def evaluations(self, fid, d, variant, seed):
        problem = BBOB(fid, d, 100 + seed)
        rng = np.random.default_rng(1000 + seed)
        gradient = {
            "none": None,
            "exact": problem.gradient,
            "random": lambda x: rng.standard_normal(d),
        }[variant]
        es = make(d=d, seed=seed, x0=np.random.default_rng(seed).uniform(-4, 4, d), sigma0=2.0,
                  budget=20_000 * d, target=1e-8)
        es.run(problem, gradient=gradient)
        self.assertLessEqual(es.p.stats.global_best.y, 1e-8)
        return es.p.stats.global_best.e

    def test_exact_gradient_helps_random_costs_little(self):
        # The medians of 7 runs, f2 and f10 at d = 10, f12 at d = 20 (where its runs vary less
        # than at d = 10); the pull request has d = 10, 40 and 100 with 10 to 15 runs. Measured on
        # macOS: exact / none 0.87, 0.87, 0.64; random / none 1.02, 1.04, 1.02. A bad gradient
        # costs about one sample per generation (1 / lambda, 10 % at d = 10).
        for fid, d in ((2, 10), (10, 10), (12, 20)):
            with self.subTest(fid=fid, d=d):
                median = {
                    variant: np.median([self.evaluations(fid, d, variant, s) for s in range(7)])
                    for variant in ("none", "exact", "random")
                }
                self.assertLess(median["exact"], 0.95 * median["none"])
                self.assertLess(median["random"], 1.15 * median["none"])


if __name__ == "__main__":
    unittest.main()
