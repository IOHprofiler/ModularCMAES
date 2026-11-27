import math
import unittest

import numpy as np

from modcma.c_maes import (
    sampling, set_seed, ModularCMAES, Settings, settings_from_dict, cmaescpp
)


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def rastrigin(x):
    x = np.asarray(x)
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def sampled_generation(es, f):
    """Ask and tell one generation (without restarts: every asked point is an individual); the
    z of its evaluated individuals, in the order they were sampled (the selection sorts them)"""
    n = 0
    while True:
        X = es.ask()
        n += len(X)
        Z = es.p.pop.Z[:, :n].copy()
        if es.tell(np.array([f(x) for x in X])):
            return Z


class TestSampling(unittest.TestCase):
    """Test case for sampling methods of C++ version Modular CMA-ES."""

    def setUp(self):
        set_seed(10)

    def sampler_test(self, sampler, expected):
        sample = sampler()
        self.assertIsInstance(sample, np.ndarray)
        self.assertEqual(len(sample), sampler.d)
        self.assertAlmostEqual(sample.sum(), expected)

    def test_base_sampler_gauss(self):
        sampler = sampling.Gaussian(5)
        self.sampler_test(sampler, -1.88320896)

    def quasi_random_test(self, make_sampler, expected):
        # Halton and Sobol draw their scrambling from the raw mt19937 output (#116; before, from
        # std::shuffle and std::uniform_int_distribution, which differ between standard
        # libraries): the same values on every platform
        set_seed(10)
        sampler = make_sampler()
        first = np.array([sampler() for _ in range(3)])
        np.testing.assert_allclose(first, expected, rtol=0, atol=1e-11)
        set_seed(10)
        sampler = make_sampler()
        np.testing.assert_array_equal(first, np.array([sampler() for _ in range(3)]))

    def test_base_sampler_halton(self):
        self.quasi_random_test(lambda: sampling.Halton(5), [
            [0.319407335315, 0.414872155312, 0.557853405374, 0.100441146484, 0.416910768239],
            [0.819407335315, 0.081538821979, 0.757853405374, 0.488196248525, 0.780547131876],
            [0.131907335315, 0.748205488645, 0.357853405374, 0.20248196281, 0.235092586421],
        ])

    def test_base_sampler_sobol(self):
        self.quasi_random_test(lambda: sampling.Sobol(5), [
            [0.444938659668, 0.587791442871, 0.115440368652, 0.318916320801, 0.031044006348],
            [0.964805603027, 0.365898132324, 0.248832702637, 0.543251037598, 0.093269348145],
            [0.921142578125, 0.553466796875, 0.134521484375, 0.447021484375, 0.971435546875],
        ])

    def test_samplers_are_random(self):
        for sampler in (sampling.Halton, sampling.Sobol):
            samples = np.array([sampler(5)() for _ in range(5)])
            self.assertGreater(abs(samples[0] - samples).sum(), 1e-10)

    def test_mirrored(self):
        sampler = sampling.Mirrored(sampling.Gaussian(5))
        sample = sampler()
        sample2 = sampler()
        self.assertTrue(np.all(sample == -sample2))

    def test_cached_sampler(self):
        points = [[1, 1], [2, 2]]
        sampler = sampling.CachedSampler(points)
        self.assertEqual(sum(sampler()), 2)
        self.assertEqual(sum(sampler()), 4)
        self.assertEqual(sum(sampler()), 2)

        sampler = sampling.CachedSampler(points, True)
        self.assertEqual(sum(sampler()), float("inf"))

        points = [[0.1, 0.1], [0.2, 0.2]]
        sampler = sampling.CachedSampler(points, True)
        self.assertAlmostEqual(sum(sampler()), -2.5631031)
        self.assertAlmostEqual(sum(sampler()), -1.6832425)

        cma = ModularCMAES(Settings(2, lambda0=2))
        cma.p.sampler = sampler
        cma.step(sum)
        z_sum = cma.p.pop.Z.sum(axis=0)
        self.assertAlmostEqual(z_sum[0], -2.5631031)
        self.assertAlmostEqual(z_sum[1], -1.6832425)



class TestMirroredPairs(unittest.TestCase):
    """#75, F10: individuals 2i and 2i + 1 of every generation are z and -z"""

    def assert_pairs(self, Z, lamb):
        # Not bit for bit: RESAMPLE evaluates a point that is still outside the box after 5 d
        # attempts as it is, and its z is then recomputed from x (a round trip in the last bits)
        for k in range(0, Z.shape[1] - 1, 2):
            np.testing.assert_allclose(Z[:, k + 1], -Z[:, k], rtol=1e-12, atol=1e-12)
        if lamb % 2 == 1 and Z.shape[1] == lamb:
            # the last individual is unpaired: not the mirror of the one before
            self.assertFalse(np.allclose(Z[:, -1], -Z[:, -2]))

    def test_pairs_with_resample(self):
        # A small box around the optimum: RESAMPLE rejects many points, and a rejected pair is
        # sampled again as a pair (before #75, 0 of 5 pairs were mirrored)
        settings = settings_from_dict(
            5, lambda0=10, mirrored="MIRRORED", bound_correction="RESAMPLE", seed=2,
            lb=np.full(5, -0.5), ub=np.full(5, 0.5), x0=np.full(5, 0.4), sigma0=0.5)
        es = ModularCMAES(settings)
        for _ in range(20):
            Z = sampled_generation(es, sphere)
            self.assert_pairs(Z, 10)

    def test_pairs_with_repelling(self):
        es = ModularCMAES(settings_from_dict(
            2, lambda0=6, mirrored="MIRRORED", repelling_type="COVERAGE",
            restart_strategy="RESTART", budget=20_000, seed=3))
        rejected = 0
        while es.p.stats.evaluations < 15_000:
            restarts = len(es.p.stats.centers)
            X = es.ask()
            if len(X) != 6 or len(es.p.stats.centers) != restarts:
                es.tell(np.array([rastrigin(x) for x in X]))  # a center of mass, hill-valley
                continue
            Z = es.p.pop.Z.copy()
            rejected += es.p.repelling.attempts
            es.tell(np.array([rastrigin(x) for x in X]))
            self.assert_pairs(Z, 6)
        self.assertGreater(len(es.p.repelling.archive), 0)
        self.assertGreater(rejected, 0)

    def test_pairs_with_sequential_selection_and_odd_lambda(self):
        # The generations stop early (after the first of a pair too) and lambda is odd: every
        # generation starts with a new pair, so its pairs stay in step
        es = ModularCMAES(settings_from_dict(
            5, lambda0=7, mu0=1, mirrored="MIRRORED", sequential_selection=True, seed=4))
        sizes = set()
        for _ in range(40):
            Z = sampled_generation(es, sphere)
            sizes.add(Z.shape[1])
            self.assert_pairs(Z, 7)
        self.assertTrue(any(n % 2 == 1 and n < 7 for n in sizes), sizes)

    def test_lambda_one_has_no_pairs(self):
        # With lambda = 1 the individual of a generation is unpaired: MIRRORED runs as NONE
        runs = []
        for mirrored in ("MIRRORED", "NONE"):
            es = ModularCMAES(settings_from_dict(3, lambda0=1, mirrored=mirrored, budget=300,
                                                 seed=5))
            es.run(sphere)
            runs.append((es.p.stats.global_best.y, es.p.stats.evaluations))
        self.assertEqual(runs[0], runs[1])

    def test_standalone_mirrored_starts_a_new_pair(self):
        sampler = sampling.Mirrored(sampling.Gaussian(3))
        z = sampler()
        sampler.begin_generation()
        self.assertFalse(np.allclose(sampler(), -z))


class TestOrthogonal(unittest.TestCase):
    """#76: the samples of every generation are orthogonal (F11), without a sign bias (F30)"""

    def assert_orthogonal(self, Z, d):
        for start in range(0, Z.shape[1], d):
            B = Z[:, start : start + d]
            B = B / np.linalg.norm(B, axis=0)
            off = B.T @ B - np.eye(B.shape[1])
            self.assertLess(np.abs(off).max(), 1e-12)

    def test_generations_are_orthogonal(self):
        for d, lamb, mirrored in ((10, 4, "NONE"), (4, 10, "NONE"), (6, 6, "NONE"),
                                  (10, 6, "MIRRORED"), (3, 10, "PAIRWISE")):
            es = ModularCMAES(settings_from_dict(d, lambda0=lamb, orthogonal=True,
                                                 mirrored=mirrored, ssa="TPA", seed=6))
            for _ in range(5):
                Z = sampled_generation(es, sphere)
                # with TPA the population is still lambda (its two points are extra)
                self.assertEqual(Z.shape[1], lamb)
                if mirrored != "NONE":
                    Z = Z[:, ::2]
                self.assert_orthogonal(Z, d)

    def test_no_sign_bias(self):
        # E[z_j[j]] = 0 for every position j of a block. Before #76 (d = 5, 5000 blocks):
        # -0.80, -0.77, -0.71, -0.70, +0.70; the standard error is about 0.014
        set_seed(11)
        d, blocks = 5, 5000
        sampler = sampling.Orthogonal(sampling.Gaussian(d), d)
        Z = np.array([[sampler() for _ in range(d)] for _ in range(blocks)])  # (blocks, j, coord)
        diagonal = Z[:, np.arange(d), np.arange(d)].mean(axis=0)
        self.assertLess(np.abs(diagonal).max(), 0.06, diagonal)
        # The first sample of each block keeps its direction (Gram-Schmidt)
        set_seed(12)
        gaussian = sampling.Gaussian(d)
        first = gaussian()
        set_seed(12)
        z = sampling.Orthogonal(sampling.Gaussian(d), d)()
        np.testing.assert_allclose(z, first, rtol=1e-12, atol=1e-12)


class TestExpectedLength(unittest.TestCase):
    """#76, F29: expected_length is E||z|| of each transformation"""

    TRANSFORMERS = ("UniformScaler", "LaplaceTransformer", "LogisticTransformer",
                    "DoubleWeibullTransformer", "IdentityTransformer", "GaussianTransformer")

    def test_within_one_percent_of_the_mean_norm(self):
        rng = np.random.default_rng(1)
        n = 200_000
        for name in self.TRANSFORMERS:
            for d in (2, 10, 100):
                expected = getattr(sampling, name)(sampling.Uniform(d)).expected_length()
                # The transformations act per coordinate: one long vector of n samples
                m = n // d if d < 100 else 20_000
                long = getattr(sampling, name)(sampling.Uniform(d * m))
                U = rng.random(d * m)
                Z = long.transform(U).reshape(m, d)
                mean = np.linalg.norm(Z, axis=1).mean()
                self.assertLess(abs(expected / mean - 1), 0.01, (name, d, expected, mean))
                if name != "GaussianTransformer":
                    self.assertLess(expected, math.sqrt(d))

    def test_closed_forms(self):
        # d = 1: E|z| of the uniform distribution on (-sqrt(3), sqrt(3)) and on (0, 1)
        self.assertAlmostEqual(
            sampling.UniformScaler(sampling.Uniform(1)).expected_length(), math.sqrt(3) / 2, 9)
        self.assertAlmostEqual(
            sampling.IdentityTransformer(sampling.Uniform(1)).expected_length(), 0.5, 9)
        # Laplace with b = 1/sqrt(2): E|z| = b
        self.assertAlmostEqual(
            sampling.LaplaceTransformer(sampling.Uniform(1)).expected_length(), math.sqrt(0.5), 7)
        # Double Weibull: Gamma(d + 1/2) / Gamma(d)
        for d in (1, 7, 300):
            self.assertAlmostEqual(
                sampling.DoubleWeibullTransformer(sampling.Uniform(d)).expected_length(),
                math.exp(math.lgamma(d + 0.5) - math.lgamma(d)), 10)

    def test_weights_use_it(self):
        for transformation, name in (("LAPLACE", "LaplaceTransformer"),
                                     ("SCALED_UNIFORM", "UniformScaler")):
            es = ModularCMAES(settings_from_dict(4, sample_transformation=transformation))
            expected = getattr(sampling, name)(sampling.Uniform(4)).expected_length()
            self.assertAlmostEqual(es.p.weights.expected_length_z, expected, 12)
            self.assertAlmostEqual(es.p.sampler.expected_length(), expected, 12)


class TestPortableRandomness(unittest.TestCase):
    """#116: the HALTON and SOBOL samplers draw the same numbers on every platform"""

    def test_shuffler_is_a_random_permutation(self):
        shuffler = cmaescpp._internal.Shuffler
        set_seed(1)
        for n in (5, 100, 128, 131072):
            s = shuffler(n)
            values = [s.next() for _ in range(n)]
            self.assertEqual(sorted(values), list(range(n)))
            self.assertEqual(s.multiplier % 8, 5)
            self.assertNotEqual(s.multiplier % s.modulus, 1)
            if n >= 100:
                # not a constant stride (before #116, every power of 2 was)
                self.assertGreater(len(set(np.diff(values[:64]) % n)), 2)


if __name__ == "__main__":
    unittest.main()
