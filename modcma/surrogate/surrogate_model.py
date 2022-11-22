from sklearn.base import BaseEstimator, TransformerMixin
from parameters import Parameters

import numpy as np
import numpy.typing as npt

from typing import Any, Callable, Optional, Union, Tuple, Iterable

from abc import abstractmethod, ABCMeta

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


from typing_utils import *

class SurrogateModelBase(metaclass=ABCMeta):
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    @abstractmethod
    def fit(self, X: XType, F: YType) -> None:
        pass

    @abstractmethod
    def predict(self, X: XType) -> YType:
        return np.tile(np.nan, (len(X), 1))

    @abstractmethod
    def predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        F = self.predict(X)
        return (F, np.tile(np.nan, F.shape))


class LQ_SurrogateModel(SurrogateModelBase):
    SAFETY_MARGIN: float = 1.1

    def train(self, X, F) -> None:
        X, y = self._get_training_samples()
        assert X.shape[0] == self.parameters.d
        self.model = self.select_model(D=X.shape[0], N=X.shape[1])
        self.model = self.model.fit(X, y)

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
