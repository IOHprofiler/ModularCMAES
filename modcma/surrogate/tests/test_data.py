from modcma.surrogate.data import *

import unittest
import random

class TestSurrogateData_V1(unittest.TestCase):
    def assertEqual(self, first: Any, second: Any, msg: Any = ...) -> None:
        if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            if first.shape == second.shape:
                if np.equal(first, second).all():
                    return
        return super().assertEqual(first, second, msg)

    def setUp(self):
        self.S = Parameters(5)
        self.A = SurrogateData_V1(self.S)

    def fillin(self, n=1):
        for _ in range(n):
            x = np.random.randn(3, 1)
            y = np.random.randn(1, 1)
            self.A.push(x, y)

    def test_voidlen(self):
        self.assertEqual(0, len(self.A))

    def test_voidPop(self):
        self.A.pop()
        self.assertEqual(0, len(self.A))
        self.A.pop(10)
        self.assertEqual(0, len(self.A))

    def test_overPop(self):
        self.fillin(10)
        self.A.pop(15)
        self.assertEqual(0, len(self.A))

    def test_push31(self):
        xs, ys = [], []
        for i in range(10):
            x = np.random.randn(3, 1)
            y = np.random.randn(1, 1)
            self.A.push(x, y)
            self.assertEqual(i+1, len(self.A))
            xs.append(x)
            ys.append(y)

        self.assertEqual(self.A.F.shape[0], 10)

        for i in range(10):
            x, y = self.A.pop()
            self.assertEqual(x, xs[i].T)
            self.assertEqual(y, ys[i])
            self.assertEqual(10-i-1, len(self.A))

    def test_push13(self):
        xs, ys = [], []
        for i in range(10):
            x = np.random.randn(1, 3)
            y = np.random.randn(1, 1)
            self.A.push(x, y)
            self.assertEqual(i+1, len(self.A))
            xs.append(x)
            ys.append(y)

        self.assertEqual(self.A.F.shape[0], 10)

        for i in range(10):
            x, y = self.A.pop()
            self.assertEqual(x, xs[i])
            self.assertEqual(y, ys[i])
            self.assertEqual(10-i-1, len(self.A))

    def test_pushmany(self):
        L = [random.randint(1, 10) for _ in range(10)]
        G = [(np.random.rand(length, 5), np.random.rand(length, 1)) for length in L]
        for (x,f) in G:
            self.A.push_many(x,f)
        self.assertEqual(len(self.A), sum(L))

        target = np.concatenate([x for (x,f) in G], axis=0)
        self.assertEqual(target, self.A.X)

        target = np.concatenate([f for (x,f) in G], axis=0)
        self.assertEqual(target, self.A.F)

    def test_sort(self):
        self.S = Parameters(5, surrogate_data_sorting='lq')
        self.A = SurrogateData_V1(self.S)
        x = np.array([
            [1, 1, 3, 5],
            [2, 3, 2, 3],
            [1, 1, 1, 2],
            [2, 9, 1, 2],
            [1, 8, 1, 2]
        ])
        y = np.array([[4, 3, 1, 5, 2]]).T
        xok = np.array([
            [2, 9, 1, 2],
            [1, 1, 3, 5],
            [2, 3, 2, 3],
            [1, 8, 1, 2],
            [1, 1, 1, 2],
        ])

        for i in range(len(y)):
            self.A.push(x[i], y[i])

        for i in [0, 1]:
            self.A.sort(i)
            self.assertEqual(self.A.X, x)
            self.assertEqual(self.A.F, y)

        for i in range(2, 5+1):
            target = list(y[:-i]) + \
                list(reversed(sorted(y[-i:])))
            target = np.array(target)

            self.A.sort(i)
            self.assertEqual(self.A.F, target)

        self.assertEqual(self.A.F, np.array([[5, 4, 3, 2, 1.]]).T)
        self.assertEqual(self.A.X, xok)

    def test_max_size(self):
        S = Parameters(5)

        for size in [3, 64, 101, 200]:
            S.surrogate_data_max_size = size
            A = SurrogateData_V1(S)

            X = np.random.rand(size + 101, 5)
            F = np.random.rand(size + 101, 1)

            A.push_many(X, F)

            self.assertEqual(len(A.F), size)
            self.assertEqual(len(A.X), size)
            self.assertEqual(A.X.shape[1], 5)

    def test_weight(self):
        S = Parameters(5)

        # FULL
        for size in [3, 64, 101, 200]:
            S.surrogate_data_max_size = size
            S.surrogate_data_min_weight = 2.5
            S.surrogate_data_max_weight = 100.
            A = SurrogateData_V1(S)

            X = np.random.rand(size + 101, 5)
            F = np.random.rand(size + 101, 1)
            A.push_many(X, F)

            self.assertEqual(len(A.W), size)
            self.assertEqual(A.W[0], S.surrogate_data_min_weight)
            self.assertEqual(A.W[-1], S.surrogate_data_max_weight)
            self.assertAlmostEqual(A.W[1] - A.W[0], A.W[-1] - A.W[-2])

        # NOT FILLED
        A = SurrogateData_V1(S)
        A.push(np.random.rand(1,5), np.random.rand(1,1))

        self.assertEqual(len(A.W), 1)
        self.assertEqual(A.W[0], S.surrogate_data_min_weight)

        A.push(np.random.rand(1,5), np.random.rand(1,1))
        self.assertEqual(len(A.W), 2)
        self.assertEqual(A.W[0], S.surrogate_data_min_weight)
        self.assertEqual(A.W[-1], S.surrogate_data_max_weight)

        A.push(np.random.rand(1,5), np.random.rand(1,1))
        self.assertEqual(len(A.W), 3)
        self.assertEqual(A.W[0], S.surrogate_data_min_weight)
        self.assertAlmostEqual(A.W[1], (S.surrogate_data_min_weight + S.surrogate_data_max_weight)/2)
        self.assertEqual(A.W[-1], S.surrogate_data_max_weight)

    def test_prune(self):
        S = Parameters(5)

        # FULL
        for size in [3, 4, 101, 200]:
            for add_size in [2, 5, 1001, 1024]:
                S.surrogate_data_max_size = size

                A = SurrogateData_V1(S)
                X = np.random.rand(size + add_size, 5)
                F = np.random.rand(size + add_size, 1)
                A.push_many(X, F)
                self.assertIsNotNone(A.F)
                self.assertIsNotNone(A.X)

                A.prune()

                self.assertEqual(len(A.F), size)
                self.assertEqual(len(A.X), size)
                self.assertEqual(A.F, F[-size:])
                self.assertEqual(A.X, X[-size:])

    @unittest.skip('TODO')
    def test_parameters(self):
        pass

