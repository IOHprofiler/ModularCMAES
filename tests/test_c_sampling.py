import unittest

import numpy as np

from modcma.c_maes import sampling, utils


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
        self.sampler_test(sampler,  1.107994899)

    def test_base_sampler_halton(self):
        sampler = sampling.Halton(5, 1)
        self.sampler_test(sampler, -3.675096792)

    def test_base_sampler_sobol(self):
        sampler = sampling.Sobol(5)
        self.sampler_test(sampler, -1.651717819)

    def test_samplers_are_random(self):
        for sampler in (sampling.Halton, sampling.Sobol):
            samples = np.array([sampler(5)() for _ in range(5)])
            self.assertGreater(abs(samples[0] - samples).sum(), 1e-10)

    def test_mirrored(self):
        sampler = sampling.Mirrored(sampling.Gaussian(5))
        sample = sampler()
        sample2 = sampler()
        self.assertTrue(np.all(sample == -sample2))


if __name__ == "__main__":
    unittest.main()
