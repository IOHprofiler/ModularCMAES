import unittest

import numpy as np

from modcma.c_maes import sampling, utils, ModularCMAES, Settings


class TestSampling(unittest.TestCase):
    """Test case for sampling methods of C++ version Modular CMA-ES."""

    def setUp(self):
        utils.set_seed(10)

    def sampler_test(self, sampler, expected):
        sample = sampler()
        self.assertIsInstance(sample, np.ndarray)
        self.assertEqual(len(sample), sampler.d)
        self.assertAlmostEqual(sample.sum(), expected)

    def test_base_sampler_gauss(self):
        sampler = sampling.Gaussian(5)
        self.sampler_test(sampler, -1.88320896)

    def quasi_random_test(self, make_sampler):
        # Halton and Sobol scramble with std::shuffle/std::uniform_int_distribution, whose
        # results differ between standard libraries, so there is no portable expected value
        # (the golden fixtures pin them per platform). Check range and reproducibility instead.
        utils.set_seed(10)
        first = np.array([make_sampler()() for _ in range(3)])
        utils.set_seed(10)
        second = np.array([make_sampler()() for _ in range(3)])
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (3, 5))
        self.assertTrue(np.all((first > 0) & (first < 1)))

    def test_base_sampler_halton(self):
        self.quasi_random_test(lambda: sampling.Halton(5))

    def test_base_sampler_sobol(self):
        self.quasi_random_test(lambda: sampling.Sobol(5))

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


if __name__ == "__main__":
    unittest.main()
