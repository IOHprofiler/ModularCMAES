"""Module containing tests for ModularCMA-ES Population."""

import unittest
import numpy as np
from modcma import population


class TestPopulation(unittest.TestCase):
    """Test case for Population object of Modular CMA-ES."""

    _dim = 5
    _lambda = 16
    _sigma = 0.5

    def setUp(self):
        """Test setup method."""
        np.random.seed(12)
        self.set_params()

    def set_params(self):
        """Set default parameters."""
        self.xmean = np.random.rand(self._dim, 1)
        self.B = np.eye(self._dim)
        self.D = np.ones((self._dim, 1))
        self.z = np.random.multivariate_normal(
            mean=np.zeros(self._dim), cov=np.eye(self._dim), size=self._lambda
        ).T
        self.y = np.dot(self.B, self.D * self.z)
        self.x = self.xmean + (self._sigma * self.y)
        self.f = np.array([sum(i) for i in self.x.T])
        self.pop = population.Population(self.x, self.y, self.f)

    def correct_copy(self, instance, other):
        """Test copy behaviour."""
        self.assertNotEqual(id(instance.x), id(other.x))
        self.assertNotEqual(id(instance.y), id(other.y))
        self.assertNotEqual(id(instance.f), id(other.f))

    def test_creation(self):
        """Test constructor behaviour."""
        self.assertIsInstance(self.pop, population.Population)
        self.correct_copy(self.pop, self.pop.copy())

    def test_sort(self):
        """Test sorting behaviour."""
        self.pop.sort()
        rank = np.argsort(self.f)
        for e in ("x", "y", ):
            self.assertListEqual(
                getattr(self, e)[:, rank].tolist(), getattr(self.pop, e).tolist()
            )
        self.assertListEqual(self.f[rank].tolist(), self.pop.f.tolist())

    def test_copy(self):
        """Test copy behaviour."""
        self.correct_copy(self.pop, self.pop.copy())

    def test_getitem(self):
        """Test whether errors are produced correctly."""
        with self.assertRaises(KeyError):
            _= self.pop["a"]
        with self.assertRaises(KeyError):
            _ = self.pop[0.1]

        self.assertIsInstance(self.pop[0], population.Population)
        self.assertIsInstance(self.pop[0:1], population.Population)
        self.assertIsInstance(self.pop[:-1], population.Population)
        self.assertIsInstance(self.pop[[1, 2, 3]], population.Population)

    def test_1d(self):
        """Test in 1d dimension."""
        self._dim = 1
        self.set_params()
        population.Population(self.x.ravel(), self.y.ravel(), self.f)

    def test_add(self):
        """Test addition.""" 
        self.pop += population.Population(self.x, self.y, self.f)
        self.assertEqual(
            self.pop.x.shape,
            (
                self._dim,
                self._lambda * 2,
            ),
        )
        self.assertEqual(
            self.pop.y.shape,
            (
                self._dim,
                self._lambda * 2,
            ),
        )
        self.assertEqual(self.pop.f.shape, (self._lambda * 2,))

        with self.assertRaises(TypeError):
            self.pop += 1

    def test_n(self):
        """Test n."""
        self.assertEqual(self._lambda, self.pop.n)

    def test_d(self):
        """Test d."""
        self.assertEqual(self._dim, self.pop.d)

    def test_repr(self):
        """Test representation.""" 
        self.assertEqual(type(repr(self.pop)), str)
        self.assertEqual(type(str(self.pop)), str)


if __name__ == "__main__":
    unittest.main()