from sklearn.base import BaseEstimator, TransformerMixin
from .parameters import Parameters
from .population import Population
import numpy as np
import numpy.typing as npt

from typing import Any, Callable, Optional, Union, Tuple, Iterable

from abc import abstractmethod, ABCMeta

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from scipy.stats import kendalltau


#####################
# Population Storage Management

class PopHistory:
    '''Saves Entire Population History

        += Population to add another
    '''

    def __init__(self, x, f, f_true, generation):
        # copy constructor
        self.x: npt.NDArray[np.float64] = x
        self.f: npt.NDArray[np.float64] = f
        self.f_true: npt.NDArray[np.bool_] = f_true
        self.generation: npt.NDArray[np.int32] = generation

    @staticmethod
    def empty(d: int) -> "PopHistory":
        return PopHistory(
                x=np.empty(shape=(d, 0)),
                f=np.empty(shape=(1, 0)),
                f_true=np.empty(shape=(1, 0), dtype=np.bool_),
                generation=np.empty(shape=(1, 0), dtype=np.int32)
            )

    def __iadd__(self, parameters: Parameters, population: Population) -> None:
        self.x = np.hstack((self.x, population.x))
        self.f = np.append(self.f, population.f)
        self.f_true = np.append(self.f_true, population.f_true)
        self.generation = np.append(
            self.generation,
            parameters.t + np.zeros_like(population.f, dtype=np.int32)
        )

    def __getitem__(self, key: npt.NDArray[Union[np.bool_, np.int32]]) -> "PopHistory":
        return PopHistory(
            self.x[:, key],
            self.f[key],
            self.f_true[key],
            self.generation[key]
        )


#####################
# FILTER CLASSES for surrogate models

class Filter(metaclass=ABCMeta):
    '''
        Remove
    '''
    @abstractmethod
    def __call__(self, pop: PopHistory) -> PopHistory:
        pass


class FilterRealEvaluation(Filter):
    def __call__(self, pop: PopHistory) -> PopHistory:
        mask = pop.f_true
        return pop[mask]


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

###############################################################################
# Helper functions


class PureQuadraticFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> npt.NDArray[np.float64]:
        return np.hstack((X, np.square(X)))


###############################################################################
# Surrogate Model


class SurrogateModelBase(metaclass=ABCMeta):
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    def _get_training_samples(self) -> Tuple[npt.NDArray[np.float64],
                                             npt.NDArray[np.float64]]:
        archive: PopHistory = self.parameters.archive
        filters: Iterable[FILTER_TYPE] = self.parameters.filters

        for f in filters:
            archive = f(archive)

        return archive.x, archive.f

    def train(self) -> None:
        pass

    @abstractmethod
    def evaluate(self,
                 x: npt.NDArray[np.float64],
                 ) -> npt.NDArray[np.float64]:
        pass
        # return np.repeat(np.nan, [x.shape[0], 1])


class LQ_SurrogateModel(SurrogateModelBase):
    SAFETY_MARGIN: float = 1.1

    def train(self) -> bool:
        X, y = self._get_training_samples()
        assert X.shape[0] == self.parameters.d
        self.model = self.select_model(D=X.shape[0], N=X.shape[1])
        self.model = self.model.fit(X, y)
        return True

    def select_model(self, D: int, N: int) -> Pipeline:
        # model             degree of freedom
        # linear            D + 1
        # quadratic         2D + 1
        # full-quadratic    C_r(D, 1) + C_r(D, 2) = (D^2 + 3D)/2

        if N >= self.SAFETY_MARGIN * ((D**2 + 3*D) / 2 + 1):
            ppl = [('full-quadratic',
                    PolynomialFeatures(degree=2, include_bias=False))]
        elif N >= self.SAFETY_MARGIN * (2*D + 1):
            ppl = [('pure-quadratic', PureQuadraticFeatures())]
        else:  # N >= self.SAFETY_MARGIN * (D + 1):
            ppl = []
        return Pipeline(ppl + [('lm', LinearRegression())])


###############################################################################
# Special models

class SurrogateStrategyBase:
    @abstractmethod
    def evaluate(self,
                 x: npt.NDArray[np.float64],
                 y: npt.NDArray[np.float64],
                 s: npt.NDArray[np.float64],
                 n_offspring: int,
                 fitness_func: Callable,
                 mcmaes,
                 ) -> npt.NDArray[np.float64]:

        f = np.empty(n_offspring, object)
        for i in range(n_offspring):
            f[i] = mcmaes.fitness_func(x[:, i])
            if mcmaes.sequential_break_conditions(i, f[i]):
                f = f[:i]
                s = s[:i]
                x = x[:, :i]
                y = y[:, :i]
                break
        return x, y, f, s





####################
# Special models


class LQR2_SurrogateModel(SurrogateModel):
    # TODO: Adjusted R^2 to switch between models
    # TODO: Interaction only for PolynomialFeatures as an option
    # TODO: Add ^3
    pass


####################
# Other Surrogate Models

class SklearnSurrogateModelBase(SurrogateModelBase):
    @abstractmethod
    def train(self) -> bool:
        self.model = Pipeline([])
        return super().train()

    def evaluate(self, x: npt.NDArray[np.float64]) -> np.ndarray:
        return self.model.predict(x.T).T


class Linear_SurrogateModel(SklearnSurrogateModelBase):
    def train(self) -> bool:
        X, y = self._get_training_samples()
        self.model = LinearRegression().fit(X.T, y.T)
        return True


class QuadraticPure_SurrogateModel(SklearnSurrogateModelBase):
    def train(self) -> bool:
        X, y = self._get_training_samples()
        self.model = Pipeline([
            ('quad.f.', PureQuadraticFeatures()),
            ('lin. m.', LinearRegression())
        ]).fit(X.T, y.T)
        return True


class QuadraticInteraction_SurrogateModel(SklearnSurrogateModelBase):
    def train(self) -> bool:
        X, y = self._get_training_samples()
        self.model = Pipeline([
            ('quad.f.', PolynomialFeatures(degree=2,
                                           interaction_only=True,
                                           include_bias=False)),
            ('lin. m.', LinearRegression())
        ]).fit(X.T, y.T)
        return True


class Quadratic_SurrogateModel(SklearnSurrogateModelBase):
    def train(self) -> bool:
        X, y = self._get_training_samples()
        self.model = Pipeline([
            ('quad.f.', PolynomialFeatures(degree=2,
                                           include_bias=False)),
            ('lin. m.', LinearRegression())
        ]).fit(X.T, y.T)
        return True
