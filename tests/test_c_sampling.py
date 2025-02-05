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

    def test_base_sampler_halton(self):
        sampler = sampling.Halton(5)
        self.sampler_test(sampler, 2.373425998)

    def test_base_sampler_sobol(self):
        sampler = sampling.Sobol(5)
        self.sampler_test(sampler, 2.2788658)

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

        points = [[0.1, .1], [.2, .2]]
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