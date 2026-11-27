"""TPA within the population, StepSizeAdaptation.TPA_INPLACE (#173).

The step-size trajectory is compared with pycma 4.6.0's CMAAdaptSigmaTPA on scripted objective
values, in two ways: with the trajectories that pycma computed (tests/data/tpa_inplace/
pycma-4.6.0.json; CI does not install pycma), and, when pycma is installed, with pycma itself.
Regenerate the reference with pycma installed:

    python -m tests.test_c_tpa_inplace write
"""

import json
import math
import os
import sys
import types
import unittest

import numpy as np

from modcma import c_maes

try:
    import cma
    from cma.sigma_adaptation import CMAAdaptSigmaTPA
except ImportError:  # pragma: no cover - optional
    cma = None

REFERENCE = os.path.join(os.path.dirname(__file__), "data", "tpa_inplace", "pycma-4.6.0.json")

#: (dim, lambda, generations, seed) of the scripted sequences
CASES = [
    (5, 8, 60, 1),
    (3, 12, 60, 2),  # lambda > d: the second term of the damping
    (20, 12, 60, 3),
    (2, 2, 40, 4),  # the smallest population
]


def scripted_values(lam, generations, seed):
    """Integers, so that ties are frequent; every 7th generation the two test points tie."""
    rng = np.random.default_rng(seed)
    values = []
    for g in range(generations):
        f = rng.integers(0, 2 * lam, size=lam).astype(float)
        if g % 7 == 3:
            f[1] = f[0]
        values.append(f)
    return values


def pycma_sigmas(dim, lam, values):
    """pycma's sigma after each generation: its update2 reads the values, the dimension and the
    population size (the damping) only, so a stand-in for the ES suffices."""
    es = types.SimpleNamespace(
        N=dim, sp=types.SimpleNamespace(popsize=lam), opts={"TPA_dampfac": 1, "verbose": -9}
    )
    tpa = CMAAdaptSigmaTPA(dimension=dim, popsize=lam)
    tpa.initialize(es)
    sigma, sigmas = 1.0, []
    for f in values:
        sigma *= tpa.update2(es, function_values=np.asarray(f))
        sigmas.append(sigma)
    return sigmas, tpa.sp.damp


def make(dim, lam=None, seed=3, **modules):
    config = dict(ssa="TPA_INPLACE", seed=seed, sigma0=1.0, budget=10**6)
    config.update(modules)
    if lam is not None:
        config["lambda0"] = lam
    return c_maes.ModularCMAES(c_maes.settings_from_dict(dim, **config))


def modcma_sigmas(dim, lam, values):
    es = make(dim, lam)
    sigmas = []
    for f in values:
        X = es.ask()  # one point per row
        assert X.shape[0] == lam, X.shape  # no extra points
        es.tell(np.asarray(f))
        sigmas.append(es.p.mutation.sigma)
    return sigmas, es.p.weights.damps


def write():
    data = {"pycma": cma.__version__, "cases": []}
    for dim, lam, generations, seed in CASES:
        values = scripted_values(lam, generations, seed)
        sigmas, damp = pycma_sigmas(dim, lam, values)
        data["cases"].append(
            {"dim": dim, "lambda": lam, "seed": seed, "damp": damp,
             "values": [v.tolist() for v in values], "sigmas": sigmas}
        )
    with open(REFERENCE, "w") as f:
        json.dump(data, f, indent=1)


def sphere(x):
    return float(np.sum((x - 0.3) ** 2))


class TestPycma(unittest.TestCase):
    def test_matches_the_reference(self):
        with open(REFERENCE) as f:
            data = json.load(f)
        self.assertEqual(data["pycma"], "4.6.0")
        for case in data["cases"]:
            with self.subTest(dim=case["dim"], lam=case["lambda"]):
                sigmas, damp = modcma_sigmas(case["dim"], case["lambda"], case["values"])
                self.assertAlmostEqual(damp, case["damp"], places=13)
                np.testing.assert_allclose(sigmas, case["sigmas"], rtol=1e-12)

    @unittest.skipIf(cma is None, "pycma is not installed")
    def test_matches_pycma(self):  # pragma: no cover - optional
        for dim, lam, generations, seed in CASES:
            values = scripted_values(lam, generations, seed + 100)
            expected, damp = pycma_sigmas(dim, lam, values)
            sigmas, modcma_damp = modcma_sigmas(dim, lam, values)
            self.assertAlmostEqual(modcma_damp, damp, places=13)
            np.testing.assert_allclose(sigmas, expected, rtol=1e-12)


class TestTestPoints(unittest.TestCase):
    def test_mirrored_along_the_mean_shift(self):
        for matrix in ("COVARIANCE", "MATRIX", "CHOLESKY", "SEPARABLE", "NATURAL_GRADIENT",
                       "CMSA", "NONE"):
            with self.subTest(matrix=matrix):
                es = make(6, matrix_adaptation=matrix)
                for _ in range(5):
                    es.step(sphere)
                a = es.p.adaptation
                shift = a.m - a.m_old
                X = es.ask()
                pop = es.p.pop
                # z_1 = ||n|| A^-1 dm / ||A^-1 dm||, z_2 = -z_1
                np.testing.assert_array_equal(pop.Z[:, 1], -pop.Z[:, 0])
                direction = a.invert_y(a.dm)
                np.testing.assert_allclose(
                    pop.Z[:, 0] / np.linalg.norm(pop.Z[:, 0]),
                    direction / np.linalg.norm(direction), rtol=1e-9, atol=1e-12,
                )
                # in x: symmetric about m, along the mean shift, the first one ahead
                ahead, behind = X[0] - a.m, X[1] - a.m
                np.testing.assert_allclose(ahead, -behind, rtol=1e-12, atol=1e-14)
                cos = ahead @ shift / (np.linalg.norm(ahead) * np.linalg.norm(shift))
                self.assertAlmostEqual(cos, 1.0, places=9)

    def test_the_first_generation_has_ordinary_samples(self):
        es = make(6)
        es.ask()
        Z = es.p.pop.Z
        self.assertGreater(np.linalg.norm(Z[:, 0] + Z[:, 1]), 1e-3)

    def test_rank_difference(self):
        es = make(4, lam=6)
        es.step(sphere)
        es.ask()
        # the point ahead best, the point behind worst: z = (5 - 0) / 5
        es.tell(np.array([0.0, 9.0, 3.0, 4.0, 5.0, 6.0]))
        self.assertEqual(es.p.mutation.rank_difference, 1.0)
        es.ask()
        es.tell(np.array([2.0, 2.0, 1.0, 3.0, 0.0, 6.0]))  # a tie: 0
        self.assertEqual(es.p.mutation.rank_difference, 0.0)

    def test_defaults(self):
        for dim, lam in ((5, None), (3, 12), (10, 30)):
            es = make(dim, lam)
            lam = es.p.lamb
            self.assertEqual(es.p.weights.cs, 0.3)
            expected = 0.7 + 2 * math.log(dim) + 2 * math.log(max(1, lam - dim))
            self.assertAlmostEqual(es.p.weights.damps, expected, places=13)
            self.assertEqual(type(es.p.mutation).__name__, "TPAInPlace")

    def test_no_extra_points(self):
        es = make(5)
        self.assertEqual(es.ask().shape[0], es.p.lamb)
        # the budget of the first generation: lambda and the center of mass, no TPA points
        settings = c_maes.settings_from_dict(5, ssa="TPA_INPLACE", budget=9)
        self.assertEqual(settings.lambda0, 8)

    def test_lambda_one_runs_sr(self):
        settings = c_maes.settings_from_dict(5, ssa="TPA_INPLACE", lambda0=1)
        self.assertEqual(settings.modules.ssa, c_maes.options.StepSizeAdaptation.SR)


class TestCombinations(unittest.TestCase):
    """Each runs, keeps the test points a mirrored pair, and gives ask-tell the results of run()."""

    CONFIGS = [
        dict(mirrored="MIRRORED"),
        dict(mirrored="PAIRWISE"),
        dict(sequential_selection=True),
        dict(sequential_selection=True, mirrored="PAIRWISE"),
        dict(elitist=True),
        dict(orthogonal=True),
        dict(sample_sigma=True),
        dict(threshold_convergence=True),
        dict(bound_correction="RESAMPLE", lb=np.full(5, -1.0), ub=np.full(5, 0.5)),
        dict(restart_strategy="IPOP"),
        dict(sampler="SOBOL"),
    ]

    def test_runs(self):
        for config in self.CONFIGS:
            with self.subTest(config=sorted(config)):
                es = make(5, budget=4000, **config)
                for _ in range(6):
                    es.step(sphere)
                es.ask()
                pop = es.p.pop
                if es.p.stats.t_run > 0:
                    np.testing.assert_array_equal(pop.Z[:, 1], -pop.Z[:, 0])
                    np.testing.assert_array_equal(pop.S[:, 1], pop.S[:, 0])

    def test_ask_tell_is_run(self):
        for config in self.CONFIGS:
            with self.subTest(config=sorted(config)):
                a = make(5, budget=3000, **config)
                a.run(sphere)
                b = make(5, budget=3000, **config)
                while not b.break_conditions():
                    X = b.ask()
                    b.tell(np.array([sphere(x) for x in X]))
                self.assertEqual(a.p.stats.evaluations, b.p.stats.evaluations)
                self.assertEqual(a.p.mutation.sigma, b.p.mutation.sigma)
                self.assertEqual(a.p.stats.global_best.y, b.p.stats.global_best.y)

    def test_solves_the_sphere(self):
        es = make(10, target=1e-8, budget=20000)
        es.run(sphere)
        self.assertLessEqual(es.p.stats.global_best.y, 1e-8)


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        if cma is None:
            sys.exit("pycma is needed to write the reference")
        write()
    else:
        unittest.main()
