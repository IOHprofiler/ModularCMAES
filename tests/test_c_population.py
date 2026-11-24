"""Module containing tests for ModularCMA-ES C++ Population."""

import unittest
import numpy as np
from modcma.c_maes import Population


class TestPopulation(unittest.TestCase):
    """Test case for Population object of Modular C++ CMA-ES."""

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
        self.Z = np.random.multivariate_normal(
            mean=np.zeros(self._dim), cov=np.eye(self._dim), size=self._lambda
        ).T
        self.Y = np.dot(self.B, self.D * self.Z)
        self.X = self.xmean + (self._sigma * self.Y)
        self.f = np.array([sum(i) for i in self.X.T])
        self.S = np.ones((self._dim, self._lambda)) * self._sigma
        self.pop = Population(self.X, self.Z, self.Y, self.f, self.S)

    def test_sort_ties_keep_index_order_nan_last(self):
        f = np.array([3.0, np.nan, 1.0, 3.0, -np.inf, 1.0, np.nan, 0.0])
        X = np.arange(2 * len(f), dtype=float).reshape(2, len(f))
        pop = Population(X, X.copy(), X.copy(), f, np.ones_like(X))
        pop.sort()
        np.testing.assert_array_equal(pop.X[0], X[0][[4, 7, 2, 5, 0, 3, 1, 6]])
        self.assertTrue(np.isnan(pop.f[-2:]).all())

    def test_resize_keeps_capacity(self):
        self.pop.resize_cols(3)
        self.assertEqual(self.pop.n, 3)
        self.assertEqual(self.pop.X.shape, (self._dim, 3))
        self.assertEqual(self.pop.capacity, self._lambda)

    def test_sort(self):
        """Test sorting behaviour."""
        self.pop.sort()
        rank = np.argsort(self.f)
        for e in ("X", "Y", "Z", "S"):
            self.assertListEqual(
                getattr(self, e)[:, rank].tolist(), getattr(self.pop, e).tolist()
            )

        self.assertListEqual(self.f[rank].tolist(), self.pop.f.tolist())

    def test_keeponly(self):
        """Test keeponly behaviour."""
        f1 = self.pop.f[1]
        # a copy: pop.X is a view of the C++ buffer, which keep_only reallocates
        x1 = self.pop.X[:, 1].copy()
        self.pop.keep_only([1])
        self.assertEqual(self.pop.f[0], f1)
        self.assertTrue(np.all(self.pop.X[:, 0] == x1))

    def test_resize(self):
        self.pop.resize_cols(2)
        self.assertEqual(self.pop.n, 2)
        self.assertEqual(self.pop.f.size, 2)
        self.assertEqual(self.pop.X.shape[1], 2)
        self.assertEqual(self.pop.S.shape[1], 2)
        self.assertEqual(self.pop.Y.shape[1], 2)
        self.assertEqual(self.pop.Z.shape[1], 2)

    def test_n_finite(self):
        self.assertEqual(self.pop.n_finite, self._lambda)
        self.pop.f = np.ones(self._lambda) * float("inf")
        self.assertEqual(self.pop.n_finite, 0)

    def test_add(self):
        """Test addition."""
        self.pop = self.pop + Population(self.X, self.Z, self.Y, self.f, self.S)
        self.assertEqual(
            self.pop.X.shape,
            (
                self._dim,
                self._lambda * 2,
            ),
        )
        self.assertEqual(
            self.pop.Y.shape,
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


if __name__ == "__main__":
    unittest.main()
