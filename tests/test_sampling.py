import types
import unittest
import numpy as np
from modcma import sampling


class TestSampling(unittest.TestCase):
    _dim = 5

    def setUp(self):
        np.random.seed(12)

    def is_sampler(self, sampler):
        self.assertIsInstance(sampler, types.GeneratorType)
        for x in range(10):
            sample = next(sampler)
            self.assertIsInstance(sample, np.ndarray)
            self.assertEqual(sample.shape, (self._dim, 1, ))

    def test_gaussian(self):
        sampler = sampling.gaussian_sampling(self._dim)
        self.is_sampler(sampler)

    def test_sobol(self):
        sampler = sampling.sobol_sampling(self._dim)
        self.is_sampler(sampler)

    def test_halton(self):
        sampler = sampling.halton_sampling(self._dim)
        self.is_sampler(sampler)

    def test_orthogonal(self):
        for base_sampler in (
                sampling.gaussian_sampling(self._dim),
                sampling.sobol_sampling(self._dim),
                sampling.halton_sampling(self._dim)):
            for n_samples in (3, 6):
                sampler = sampling.orthogonal_sampling(
                    base_sampler, n_samples)
                self.is_sampler(sampler)

    def test_mirrored(self):
        for base_sampler in (
                sampling.gaussian_sampling(self._dim),
                sampling.sobol_sampling(self._dim),
                sampling.halton_sampling(self._dim)):
            sampler = sampling.mirrored_sampling(
                base_sampler
            )
            self.is_sampler(sampler)
            first_sample = next(sampler)
            second_sample = next(sampler)
            for i, j in zip(first_sample, second_sample):
                self.assertEqual(i, j * -1)


if __name__ == "__main__":
    unittest.main()
