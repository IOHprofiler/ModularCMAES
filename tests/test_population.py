import types
import unittest
import numpy as np
from ccmaes import population, utils


class TestPopulation(unittest.TestCase):
    _dim = 5
    _lambda = 16
    _sigma = .5

    def setUp(self):
        np.random.seed(12)
        self.xmean = np.random.rand(self._dim, 1)
        self.B = np.eye(self._dim)
        self.D = np.ones((self._dim, 1))
        self.z = np.random.multivariate_normal(
            mean=np.zeros(self._dim),
            cov=np.eye(self._dim),
            size=self._lambda
        ).T
        self.y = np.dot(self.B, self.D * self.z)
        self.x = self.xmean + (self._sigma * self.y)
        self.f = np.array([utils.sphere_function(i) for i in self.x.T])

    def correct_copy(self, instance, other):
        self.assertNotEqual(
            id(instance.x), id(other.x)
        )
        self.assertNotEqual(
            id(instance.y), id(other.y)
        )
        self.assertNotEqual(
            id(instance.f), id(other.f)
        )

    def test_creation(self):
        pop = population.Population(
            x=self.x,
            y=self.y,
            f=self.f,
        )
        self.assertIsInstance(pop, population.Population)
        self.correct_copy(pop, self)

    def test_sort(self):
        pop = population.Population(
            x=self.x,
            y=self.y,
            f=self.f,
        )
        pop.sort()
        rank = np.argsort(self.f)
        self.assertListEqual(
            self.x[:, rank].tolist(),
            pop.x.tolist()
        )
        self.assertListEqual(
            self.y[:, rank].tolist(),
            pop.y.tolist()
        )
        self.assertListEqual(
            self.f[rank].tolist(),
            pop.f.tolist()
        )

    def test_copy(self):
        pop = population.Population(
            x=self.x,
            y=self.y,
            f=self.f,
        )
        self.correct_copy(pop, pop.copy())

    def test_getitem(self):
        pop = population.Population(self.x, self.y, self.f)

        with self.assertRaises(KeyError):
            pop['a']
        with self.assertRaises(KeyError):
            pop[.1]

        self.assertIsInstance(pop[0], population.Population)
        self.assertIsInstance(pop[0:1], population.Population)
        self.assertIsInstance(pop[:-1], population.Population)


if __name__ == '__main__':
    unittest.main()
