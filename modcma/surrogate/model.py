import inspect
import sys

import numpy as np
import numpy.typing as npt

from typing import Any, Callable, Optional, Union, Tuple, Iterable, Type
from typing_extensions import override

from abc import abstractmethod, abstractproperty, ABCMeta

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.base import BaseEstimator, TransformerMixin

from ..parameters import Parameters
from ..typing_utils import XType, YType


def normalize_string(s: str):
    return s.lower().replace(' ', '_')

class SurrogateModelBase(metaclass=ABCMeta):
    ModelName = "Base"

    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    @abstractmethod
    def fit(self, X: XType, F: YType, W: YType) -> None:
        self.fitted = True

    @property
    def fitted(self):
        return self._fitted

    @fitted.setter
    def fitted(self, value):
        assert value is False
        self._fitted = False

    @abstractmethod
    def predict(self, X: XType) -> YType:
        return np.tile(np.nan, (len(X), 1))

    @abstractmethod
    def predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        F = self.predict(X)
        return (F, np.tile(np.nan, F.shape))

    @abstractproperty
    def df(self):
        return 0

    @property
    def max_df(self):
        return self.df

    @classmethod
    def name(cls) -> str:
        return normalize_string(cls.ModelName)

class LQ_SurrogateModel(SurrogateModelBase):
    ModelName = 'LQ'
    SAFETY_MARGIN: float = 1.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: Optional[Pipeline] = None
        self._dof: int = self.parameters.d + 1
        self.i_model: int = 0

    def _select_model(self, D: int, N: int) -> Pipeline:
        # model             degree of freedom
        # linear            D + 1
        # quadratic         2D + 1
        # full-quadratic    C_r(D, 1) + C_r(D, 2) = (D^2 + 3D)/2 + 1

        if N >= self.SAFETY_MARGIN * ((D**2 + 3*D) / 2 + 1):
            ppl = [('full-quadratic',
                    PolynomialFeatures(degree=2, include_bias=False))]
            self._dof = (self.parameters.d**2 + 3*self.parameters.d + 2)//2
            self.i_model = 2
        elif N >= self.SAFETY_MARGIN * (2*D + 1):
            ppl = [('pure-quadratic', PureQuadraticFeatures())]
            self._dof = 2*self.parameters.d + 1
            self.i_model = 1
        else:
            ppl = []
            self._dof = self.parameters.d + 1
            self.i_model = 0
        return Pipeline(ppl + [('lm', LinearRegression())])

    @override
    def fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = self._select_model(D=X.shape[0], N=X.shape[1])
        self.model = self.model.fit(X, F, sample_weight=W)
        super().fit(X, F, W)

    @override
    def predict(self, X: XType) -> YType:
        if self.model is None:
            return super().predict(X)
        return self.model.predict(X)

    @property
    def df(self):
        return self._dof

    @property
    def max_df(self):
        return (self.parameters.d**2 + 3*self.parameters.d)/2 + 1


####################
# Other Surrogate Models


class SklearnSurrogateModelBase(SurrogateModelBase):
    @override
    def predict(self, X: XType) -> YType:
        self.model: Pipeline
        return self.model.predict(X)

    def predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        return super().predict_with_confidence(X)


class Linear_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'Linear'

    @override
    def fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('linear', LinearRegression())]).fit(X, F, sample_weight=W)
        super().fit(X, F, W)

    @property
    def df(self):
        return self.parameters.d + 1


class QuadraticPure_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'QuadraticPure'

    @override
    def fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('quad.f.', PureQuadraticFeatures()),
            ('lin. m.', LinearRegression())
        ]).fit(X, F, sample_weight=W)
        super().fit(X, F, W)

    @property
    def df(self):
        return 2*self.parameters.d + 1


class QuadraticInteraction_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'QuadraticInteraction'

    @override
    def fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('quad.f.', PolynomialFeatures(degree=2,
                                           interaction_only=True,
                                           include_bias=False)),
            ('lin. m.', LinearRegression())
        ]).fit(X, F, sample_weight=W)
        super().fit(X, F, W)

    @property
    def df(self):
        return (self.parameters.d*(self.parameters.d + 1) + 2)//2


class Quadratic_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'Quadratic'

    @override
    def fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('quad.f.', PolynomialFeatures(degree=2,
                                           include_bias=False)),
            ('lin. m.', LinearRegression())
        ]).fit(X, F)
        super().fit(X, F, W)

    @property
    def df(self):
        return (self.parameters.d + 2)*(self.parameters.d + 1) // 2


def get_model(parameters: Parameters) -> SurrogateModelBase:
    to_find = normalize_string(parameters.surrogate_strategy)
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    for (modelname, model) in clsmembers:
        if issubclass(model, SurrogateModelBase):
            model: Type[SurrogateModelBase]
            if model.name() == to_find:
                return model(parameters)
    raise NotImplementedError(
        f'Cannot find model with name "{parameters.surrogate_strategy}"')

####################
# Helper functions


class PureQuadraticFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> npt.NDArray[np.float64]:
        return np.hstack((X, np.square(X)))


if __name__ == '__main__':
    import unittest

    class TestInterface(unittest.TestCase):
        def test_Linear(self):
            parameters = Parameters(d=2)
            model = Linear_SurrogateModel(parameters)

            X = np.hstack([np.array([np.linspace(0, 1, 10)]).T, np.random.rand(10, 2)])
            F = X[:, 0] * 2
            W = np.ones_like(F)
            model.fit(X, F, W)

            X = np.array([[20., 3.1, 3.4]])
            Fh = model.predict(X)
            self.assertAlmostEqual(Fh[0], 40.)

    unittest.main()


'''
####################
# Special models

class LQR2_SurrogateModel(SurrogateModelBase):
    # TODO: Adjusted R^2 to switch between models
    # TODO: Interaction only for PolynomialFeatures as an option
    # TODO: Add ^3
    pass
'''
