"""Module containing tests for ModularCMA-ES samplers."""

import types
import unittest
import numpy as np
from modcma import sampling


class TestSampling(unittest.TestCase):
    """Test case for Modular CMA-ES samplers."""

    _dim = 5

    def setUp(self):
        """Test setup method."""
        np.random.seed(12)

    def is_sampler(self, sampler):
        """Test if a sampler is a sampler."""
        self.assertIsInstance(sampler, types.GeneratorType)
        for _ in range(10):
            sample = next(sampler)
            self.assertIsInstance(sample, np.ndarray)
            self.assertEqual(
                sample.shape,
                (
                    self._dim,
                    1,
                ),
            )

    def test_gaussian(self):
        """Test gaussian sampling."""
        sampler = sampling.gaussian_sampling(self._dim)
        self.is_sampler(sampler)

    def test_sobol(self):
        """Test sobol sampling."""
        sampler = sampling.sobol_sampling(sampling.Sobol(self._dim))
        self.is_sampler(sampler)

    def test_halton(self):
        """Test halton sampling."""
        sampler = sampling.halton_sampling(sampling.Halton(self._dim))
        self.is_sampler(sampler)

    def test_orthogonal(self):
        """Test orthogonal sampling."""
        for base_sampler in (
            sampling.gaussian_sampling(self._dim),
            sampling.sobol_sampling(sampling.Sobol(self._dim)),  
            sampling.halton_sampling(sampling.Halton(self._dim)),
        ):
            for n_samples in (3, 6):
                sampler = sampling.orthogonal_sampling(base_sampler, n_samples)
                self.is_sampler(sampler)

    def test_mirrored(self):
        """Test mirrored sampling."""
        for base_sampler in (
            sampling.gaussian_sampling(self._dim),
            sampling.sobol_sampling(sampling.Sobol(self._dim)),  
            sampling.halton_sampling(sampling.Halton(self._dim)),
        ):
            sampler = sampling.mirrored_sampling(base_sampler)
            self.is_sampler(sampler)
            first_sample = next(sampler)
            second_sample = next(sampler)
            for i, j in zip(first_sample, second_sample):
                self.assertEqual(i, j * -1)


if __name__ == "__main__":
    unittest.main()
