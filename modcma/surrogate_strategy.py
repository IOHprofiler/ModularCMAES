import numpy as np

from typing import Callable, List, Union, Self, Optional, Any
from numpy.typing import NDArray
from typing_utils import XType, YType, xType, yType
from scipy.stats import kendalltau
from abc import ABCMeta, abstractmethod, abstractproperty


#  from modularcmaes import ModularCMAES
from parameters import SurrogateData_Settings


class SurrogateData_V1(metaclass=ABCMeta):
    FIELDS = ['_X', '_F']

    def __init__(self, settings: SurrogateData_Settings):
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

        if self.settings.sorting in ['LQ', ]:
            measure = self._F[selection].ravel()
            order = np.argsort(measure)[::-1]
        else:
            raise NotImplementedError('Unknown sorting method')
        return order

    def sort(self, n: Optional[int] = None) -> None:
        ''' sorts top n elements default: sorts all elements '''

        if (n is not None and n <= 1) or len(self) <= 1:
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
        if self.settings.max_size:
            to_remove = len(self) - self.settings.max_size
            if to_remove > 0:
                self.pop(number=to_remove)

    def __len__(self) -> int:
        ''' number of saved samples (not nessesary for trainign purposes) '''
        if self._F is None:
            return 0
        return self._F.shape[0]

    @property
    def model_size(self) -> int:
        ''' number of samples selected for training a surrogate model '''
        if self.settings.max_size is None:
            return len(self)
        else:
            return min(len(self), self.settings.max_size)

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
        if self.settings.weight_function == 'linear':
            return np.linspace(self.settings.weight_min,
                               self.settings.weight_max,
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
            self.S = SurrogateData_Settings()
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
            S = SurrogateData_Settings()

            for size in [3, 64, 101, 200]:
                S.max_size = size
                A = SurrogateData_V1(S)

                X = np.random.rand(size + 101, 5)
                F = np.random.rand(size + 101, 1)

                A.push_many(X, F)

                self.assertEqual(len(A.F), size)
                self.assertEqual(len(A.X), size)
                self.assertEqual(A.X.shape[1], 5)

        def test_weight(self):
            S = SurrogateData_Settings()

            # FULL
            for size in [3, 64, 101, 200]:
                S.max_size = size
                S.weight_min = 2.5
                S.weight_max = 100
                A = SurrogateData_V1(S)

                X = np.random.rand(size + 101, 5)
                F = np.random.rand(size + 101, 1)
                A.push_many(X, F)

                self.assertEqual(len(A.W), size)
                self.assertEqual(A.W[0], S.weight_min)
                self.assertEqual(A.W[-1], S.weight_max)
                self.assertAlmostEqual(A.W[1] - A.W[0], A.W[-1] - A.W[-2])

            # NOT FILLED
            A = SurrogateData_V1(S)
            A.push(np.random.rand(1,5), np.random.rand(1,1))

            self.assertEqual(len(A.W), 1)
            self.assertEqual(A.W[0], S.weight_min)

            A.push(np.random.rand(1,5), np.random.rand(1,1))
            self.assertEqual(len(A.W), 2)
            self.assertEqual(A.W[0], S.weight_min)
            self.assertEqual(A.W[-1], S.weight_max)

            A.push(np.random.rand(1,5), np.random.rand(1,1))
            self.assertEqual(len(A.W), 3)
            self.assertEqual(A.W[0], S.weight_min)
            self.assertAlmostEqual(A.W[1], (S.weight_min + S.weight_max)/2)
            self.assertEqual(A.W[-1], S.weight_max)

        def test_prune(self):
            S = SurrogateData_Settings()

            # FULL
            for size in [3, 4, 101, 200]:
                for add_size in [2, 5, 1001, 1024]:

                S.max_size = size

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


class SurrogateModelBase(metaclass=ABCMeta):
    def on_population_size_change(self, new_size) -> None:
        pass

    def sort(self, top=None) -> None:
        pass

    @abstractproperty
    def true_y(self) -> YType:
        return np.array(np.nan)

    @abstractproperty
    def true_x(self) -> XType:
        return np.array(np.nan)

    def __call__(self, X: XType) -> YType:
        return np.array(np.nan)


class PopulationRepresentation:
    """Manage incremental evaluation of a population of solutions.
    Evaluate solutions, add them to the model and keep track of which
    solutions were evaluated.
    """

    def __init__(self,
                 X: XType,
                 settings: LQSurrogateStrategySettings,
                 surrogate_model: SurrogateModelBase,
                 fitness_function: Callable
                 ):
        """all is based on the population (list of solutions) `X`"""
        self.X = X

        self.evaluated = np.zeros(len(X), dtype=np.bool_)
        self.fvalues = np.repeat(np.nan, len(X))
        self.surrogate: SurrogateModelBase = surrogate_model
        self.settings: LQSurrogateStrategySettings = settings
        self.fitness_function: Callable = fitness_function

    def _eval_sequence(self, number, X):
        """evaluate unevaluated entries until `number` of entries are evaluated *overall*.
        """
        F_model = self.surrogate(X)

        for i in np.argsort(F_model):
            if self.evaluations >= number:
                break
            if not self.evaluated[i]:
                self.fvalues[i] = self.fitness_function(self.X[i])
                self.evaluated[i] = True

        assert self.evaluations == number or \
            self.evaluations == len(self.X) < number

    def surrogate_values(self, true_values_if_all_available=True) -> YType:
        """return surrogate values """

        if true_values_if_all_available and self.evaluations == len(self.X):
            return self.fvalues

        F_model: YType = self.surrogate(self.X)
        m_offset = np.nanmin(F_model)
        f_offset = np.nanmin(self.fvalues)

        if np.isfinite(f_offset):
            return F_model - m_offset + f_offset
        else:
            return F_model

    @property
    def evaluations(self):
        return sum(self.evaluated)

    def __call__(self, X: XType) -> YType:
        """return population f-values.
        Evaluate at least one solution on the true fitness.
        The smallest returned value is never smaller than the smallest truly
        evaluated value.
        """

        number_evaluated = self.settings.number_of_evaluated

        while len(X) - sum(self.evaluated) > 0:  # all evaluated ?
            self._eval_sequence(number_evaluated, X)
            self.surrogate.sort(top=number_evaluated)


            tau, _ = kendalltau(
                self.surrogate.true_y,
                self.surrogate(self.surrogate.true_x)
            )
            if tau >= self.settings.tau_truth_threshold:
                break

            number_evaluated += int(np.ceil(
                number_evaluated
                * self.settings.increase_of_number_of_evaluated
            ))

        # popsi = len(x)
        # nevaluated = self.evaluations
        n_for_tau = lambda popsi, nevaluated: 
            int(max((15, min((1.2 * nevaluated, 0.75 * popsi)))))
            max(
        int(max((15, min((1.2 * nevaluated, 0.75 * popsi)))))


        self.surrogate.sort(top=self.evaluations)
        return self.surrogate_values(
            self.settings.return_true_fitness_if_all_evaluated
        )


        # TODO: Zjistit jestli je kendall stejny
        # TODO: Zjistit jestli je linearni regerese stejna
        # TODO: Nfortau







class LQ_SurrogateStrategy(ModularCMAES):
    pass

'''

