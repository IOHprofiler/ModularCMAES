from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from .parameters import Parameters
from .population import Population
import numpy as np
import numpy.typing as npt

from typing import Any, Optional, Union, Tuple

from abc import abstractmethod, ABCMeta

import math
import sklearn
import sklearn.linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

#####################
# Population Storage Management


class PopHistory:
    '''Saves Entire Population History

        += Population to add another
    '''

    def __init__(self, x, y, f, f_true, s, generation=None):
        self.population: Population = Population(
                x=x, y=y, f=f, f_true=f_true, s=s
            )
        self.generation: npt.NDArray[np.int32] = \
            np.empty(shape=(1, 0), dtype=np.int32) if generation is None else \
            generation

    @staticmethod
    def empty(d: int) -> "PopHistory":
        return PopHistory(
                x=np.empty(shape=(d, 0)),
                y=np.empty(shape=(d, 0)),
                f=np.empty(shape=(1, 0)),
                f_true=np.empty(shape=(1, 0)),
                s=np.empty(shape=(1, 0))
            )

    def __iadd__(self, parameters: Parameters, population: Population) -> None:
        self.population += population
        # extra info
        self.generation = np.append(
            self.generation,
            parameters.t + np.zeros_like(population.f, dtype=np.int32)
        )

    def __getitem__(self, key: npt.NDArray[np.bool_]) -> "PopHistory":
        return PopHistory(
            self.population.x[:, key],
            self.population.y[:, key],
            self.population.f[key],
            self.population.f_true[key],
            self.population.s[key],
            self.generation[key]
        )


#####################
# FILTER CLASSES for surrogate models

class FilterRealEvaluation:
    def __call__(self, pop: PopHistory) -> PopHistory:
        mask = pop.population.f_true
        return pop[mask]


class FilterDistance(metaclass=ABCMeta):
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
        center_x = pop.population.x - self.parameters.m
        return np.sqrt(self.transformation @ center_x)


FILTER_TYPE = Union[FilterRealEvaluation, FilterDistanceMahalanobis]

####################
# Helper functions


class PureQuadraticFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> npt.NDArray[np.float64]:
        return np.hstack((X, np.square(X)))


####################
# Surrogate Model


class SurrogateModel(metaclass=ABCMeta):
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    def _get_training_samples(self) -> Tuple[npt.NDArray[np.float64],
                                             npt.NDArray[np.float64]]:
        # TODO: get data
        # TODO: make data unique 
        return np.empty((0,0)), np.empty((0,0))

    @abstractmethod
    def train(self) -> bool:
        return False

    @abstractmethod
    def evaluate(self, x: npt.NDArray[np.float64]) -> np.ndarray:
        return np.repeat(np.nan, [x.shape[0], 1])


class Sklearn_SurrogateModels(SurrogateModel):
    @abstractmethod
    def train(self) -> bool:
        self.model = Pipeline([])
        return super().train()

    def evaluate(self, x: npt.NDArray[np.float64]) -> np.ndarray:
        return self.model.predict(x.T).T


class LQ_SurrogateModel(Sklearn_SurrogateModels):
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
            ppl = [PolynomialFeatures(degree=2, include_bias=False)]
        elif N >= self.SAFETY_MARGIN * (2*D + 1):
            ppl = [PureQuadraticFeatures()]
        else:  # N >= self.SAFETY_MARGIN * (D + 1):
            ppl = []
        return Pipeline(ppl + [LinearRegression()])


####################
# Special models


class LQR2_SurrogateModel(SurrogateModel):
    # TODO: Adjusted R^2 to switch between models
    # TODO: Interaction only for PolynomialFeatures as an option
    # TODO: Add ^3
    pass


####################
# Other models


class Linear_SurrogateModel(Sklearn_SurrogateModels):
    def train(self) -> bool:
        X, y = self._get_training_samples()
        self.model = LinearRegression().fit(X.T, y.T)
        return True


class QuadraticPure_SurrogateModel(Sklearn_SurrogateModels):
    def train(self) -> bool:
        X, y = self._get_training_samples()
        self.model = Pipeline([
            ('quad.f.', PureQuadraticFeatures()),
            ('lin. m.', LinearRegression())
        ]).fit(X.T, y.T)
        return True


class QuadraticInteraction_SurrogateModel(Sklearn_SurrogateModels):
    def train(self) -> bool:
        X, y = self._get_training_samples()
        self.model = Pipeline([
            ('quad.f.', PolynomialFeatures(degree=2,
                                           interaction_only=True,
                                           include_bias=False)),
            ('lin. m.', LinearRegression())
        ]).fit(X.T, y.T)
        return True


class Quadratic_SurrogateModel(Sklearn_SurrogateModels):
    def train(self) -> bool:
        X, y = self._get_training_samples()
        self.model = Pipeline([
            ('quad.f.', PolynomialFeatures(degree=2,
                                           include_bias=False)),
            ('lin. m.', LinearRegression())
        ]).fit(X.T, y.T)
        return True
