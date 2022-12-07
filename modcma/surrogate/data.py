import numpy as np
import math

from typing import Callable, List, Union, Self, Optional, Any
from numpy.typing import NDArray
from scipy.stats import kendalltau
from abc import ABCMeta, abstractmethod, abstractproperty


#  from modularcmaes import ModularCMAES
from ..parameters import Parameters
from ..typing_utils import XType, YType, xType, yType


class SurrogateData_V1(metaclass=ABCMeta):
    FIELDS = ['_X', '_F']

    def __init__(self, settings: Parameters):
        self.settings = settings

        self._X: Optional[XType] = None
        self._F: Optional[YType] = None

    def push(self, x, f: Union[YType, float]):
        ''' push elements to the archive '''
        x = np.atleast_2d(np.ravel(x))
        f = np.atleast_2d(np.ravel(f))

        if self._X is None or self._F is None:
            self._X, self._F = x, f
        else:
            self._X = np.vstack([self._X, x])
            self._F = np.vstack([self._F, f])

    def push_many(self, X, F):
        ''' same as push but with arrays '''
        if self._X is None or self._F is None:
            self._X, self._F = X, F
        else:
            self._X = np.vstack([self._X, X])
            self._F = np.vstack([self._F, F])

    def pop(self, number: int = 1):
        ''' removes n elements from the beggining of the stack (default=1)
            and returns them
        '''
        if self._X is None or self._F is None:
            return None, None
        x = self._X[:number]
        f = self._F[:number]
        self._X = self._X[number:]
        self._F = self._F[number:]
        return x, f

    def _sort_selection(self, selection: slice):
        ''' implemnts the sorting algorithm; returns order indices '''
        if self._F is None:
            return

        if self.settings.surrogate_data_sorting in ['LQ', 'lq' ]:
            measure = self._F[selection].ravel()
            order = np.argsort(measure)[::-1]
        else:
            raise NotImplementedError('Unknown sorting method')
        return order

    def sort(self, n: Optional[int] = None) -> None:
        ''' sorts top n elements default: sorts all elements '''

        if (n is not None and n <= 1) \
            or len(self) <= 1 \
            or self.settings.surrogate_data_sorting == 'time':
            return

        if n is None:
            select: slice = slice(None)
            other: slice = slice(0, 0)
        else:
            n = min(len(self), n)
            select: slice = slice(-n, None)
            other: slice = slice(None, -n)

        order = self._sort_selection(select)

        for name in self.FIELDS:
            data = getattr(self, name)
            new_data = [data[other], data[select, :][order, :]]
            setattr(self, name, np.vstack(new_data))

    def prune(self) -> None:
        ''' removes unwanted elements '''

        # MAX_SIZE
        if len(self) > self.model_size:
            self.pop(number=len(self) - self.model_size)

    def __len__(self) -> int:
        ''' number of saved samples (not nessesary for trainign purposes) '''
        if self._F is None:
            return 0
        return self._F.shape[0]

    @property
    def model_size(self) -> int:
        ''' number of samples selected for training a surrogate model '''
        size = len(self)

        # absolute max
        if self.settings.surrogate_data_max_size is not None:
            size = min(size, self.settings.surrogate_data_max_size)

        # relative max
        if self.settings.surrogate_data_max_relative_size is not None:
            if self.settings.surrogate_model_instance is not None:
                if self.settings.surrogate_model_instance.dof > 0:
                    size = min(size,
                               int(math.ceil(
                                   self.settings.surrogate_data_max_relative_size
                                   * self.settings.surrogate_model_instance.dof)))
        return size

        # truncation ratio
        #if self.settings.surrogate_data_truncation_ratio is not None:
        #    size = int(math.ceil(size * self.settings.surrogate_data_truncation_ratio))
        #return size


    def __getitem__(self, items):
        return
        pass

    # MODEL BUILDING BUSINESS

    @property
    def X(self) -> Optional[XType]:  # Covariates
        if self._X is None:
            return None
        return self._X[-self.model_size:]

    @property
    def F(self) -> Optional[YType]:  # Target Values
        if self._F is None:
            return None
        return self._F[-self.model_size:]

    @property
    def W(self):  # Weight
        if self.settings.surrogate_data_weighting == 'linear':
            return np.linspace(self.settings.surrogate_data_min_weight,
                               self.settings.surrogate_data_max_weight,
                               num=self.model_size)
        else:
            raise NotImplementedError("Couldnt interpret the weight_function")


if __name__ == '__main__':
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
                [2, 9, 1, 25],
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

                    A.prune()

                    self.assertEqual(len(self.F), size)
                    self.assertEqual(len(self.X), size)
                    self.assertEqual(self.F, F[-size:])
                    self.assertEqual(self.X, X[-size:])

        @unittest.skip('TODO')
        def test_parameters(self):
            pass

    unittest.main()


'''
#####################
# Population Storage Management

class FilterUnique(Filter):
    def __call__(self, pop: PopHistory) -> PopHistory:
        _, ind = np.unique(pop.x, axis=1, return_index=True)
        return pop[ind]


class FilterDistance(Filter):
    def __init__(self, parameters: Parameters, distance: float):
        self.max_distance = distance
        self.parameters = parameters

    @abstractmethod
    def _compute_distance(self, pop: PopHistory) -> npt.NDArray[np.float32]:
        pass

    def _get_mask(self, pop: PopHistory) -> npt.NDArray[np.bool_]:
        distance = self._compute_distance(pop)
        return distance <= self.max_distance

    def __call__(self, pop: PopHistory) -> PopHistory:
        mask = self._get_mask(pop)
        return pop[mask]


class FilterDistanceMahalanobis(FilterDistance):
    def __init__(self, parameters: Parameters, distance: float):
        super().__init__(parameters, distance)
        B = self.parameters.B
        sigma = self.parameters.sigma
        D = self.parameters.D

        self.transformation = np.linalg.inv(B) @ np.diag((1./sigma)*(1./D))

    def _compute_distance(self, pop: PopHistory) -> npt.NDArray[np.float32]:
        center_x = pop.x - self.parameters.m
        return np.sqrt(self.transformation @ center_x)


FILTER_TYPE = Union[
    FilterRealEvaluation,
    FilterUnique,
    FilterDistanceMahalanobis
]

'''
