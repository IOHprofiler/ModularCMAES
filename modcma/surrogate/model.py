import inspect
import sys

import numpy as np
from numpy.random import sample
import numpy.typing as npt

from typing import Any, Callable, Optional, Union, Tuple, Iterable, Type
from typing_extensions import override

from abc import abstractmethod, abstractproperty, ABCMeta

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.base import BaseEstimator, TransformerMixin

from modcma.parameters import Parameters
from modcma.typing_utils import XType, YType
from modcma.utils import normalize_string


####################
# Helper functions


class PureQuadraticFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> npt.NDArray[np.float64]:
        return np.hstack((X, np.square(X)))


def normalize_X(X: XType, d):
    assert X.shape[1] == d
    return X


def normalize_F(Y: YType):
    return Y.ravel()


normalize_W = normalize_F


####################
# Models


class SurrogateModelBase(metaclass=ABCMeta):
    ModelName = "Base"

    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    def fit(self,
            X: Optional[XType],
            F: Optional[YType],
            W: Optional[YType] = None):
        ''' fit the surrogate '''
        if X is None or F is None:
            self.fitted = False
            return self

        X = normalize_X(X, self.parameters.d)
        F = normalize_F(F)

        if W is None:
            W = np.ones_like(F)
        else:
            W = normalize_W(W)
        self._fit(X, F, W)
        self.fitted = True
        return self

    @abstractmethod
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        pass

    def predict(self, X: XType) -> YType:
        F = self._predict(X)
        return normalize_F(F)

    @abstractmethod
    def _predict(self, X: XType) -> YType:
        return np.tile(np.nan, (len(X), 1))

    def predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        F = self.predict(X)
        return (F, np.tile(np.nan, F.shape))

    @property
    def fitted(self) -> bool:
        return self._fitted

    @fitted.setter
    def fitted(self, value: bool):
        self._fitted = value

    @abstractproperty
    def df(self) -> int:
        return 0

    @property
    def max_df(self) -> int:
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

    def _select_model(self, N: int, D: int) -> Pipeline:
        # model             degree of freedom
        # linear            D + 1
        # quadratic         2D + 1
        # full-quadratic    C_r(D, 1) + C_r(D, 2) = (D^2 + 3D)/2 + 1

        margin = self.parameters.surrogate_model_lq_margin

        if N >= margin * ((D ** 2 + 3 * D) / 2 + 1):
            ppl = [('full-quadratic',
                    PolynomialFeatures(degree=2, include_bias=False))]
            self._dof = (self.parameters.d ** 2 + 3 * self.parameters.d + 2) // 2
            self.i_model = 2
        elif N >= margin * (2 * D + 1):
            ppl = [('pure-quadratic', PureQuadraticFeatures())]
            self._dof = 2 * self.parameters.d + 1
            self.i_model = 1
        else:
            ppl = []
            self._dof = self.parameters.d + 1
            self.i_model = 0
        return Pipeline(ppl + [('linearregression', LinearRegression())])

    @override
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        (N, D) = X.shape
        self.model = self._select_model(N, D)
        self.model = self.model.fit(X, F, linearregression__sample_weight=W)

    @override
    def _predict(self, X: XType) -> YType:
        if self.model is None:
            return super().predict(X)
        return self.model.predict(X)

    @property
    def df(self) -> int:
        return self._dof

    @property
    def max_df(self) -> int:
        return (self.parameters.d ** 2 + 3 * self.parameters.d) // 2 + 1


####################
# Other Surrogate Models


class SklearnSurrogateModelBase(SurrogateModelBase):
    @override
    def _predict(self, X: XType) -> YType:
        self.model: Pipeline
        return self.model.predict(X)

    '''
    def predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        return super().predict_with_confidence(X)
    '''


class Linear_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'Linear'

    @override
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('linearregression', LinearRegression())
        ]).fit(X, F, linearregression__sample_weight=W)

    @property
    def df(self) -> int:
        return self.parameters.d + 1


class QuadraticPure_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'QuadraticPure'

    @override
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('quad.f.', PureQuadraticFeatures()),
            ('linearregression', LinearRegression())
        ]).fit(X, F, linearregression__sample_weight=W)

    @property
    def df(self) -> int:
        return 2 * self.parameters.d + 1


class QuadraticInteraction_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'QuadraticInteraction'

    @override
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('quad.f.', PolynomialFeatures(degree=2,
                                           interaction_only=True,
                                           include_bias=False)),
            ('linearregression', LinearRegression())
        ]).fit(X, F, linearregression__sample_weight=W)

    @property
    def df(self) -> int:
        return (self.parameters.d * (self.parameters.d + 1) + 2) // 2


class Quadratic_SurrogateModel(SklearnSurrogateModelBase):
    ModelName = 'Quadratic'

    @override
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = Pipeline([
            ('quad.f.', PolynomialFeatures(degree=2,
                                           include_bias=False)),
            ('linearregression', LinearRegression())
        ]).fit(X, F, linearregression__sample_weight=W)

    @property
    def df(self):
        return (self.parameters.d + 2) * (self.parameters.d + 1) // 2


def get_model(parameters: Parameters) -> SurrogateModelBase:
    to_find = normalize_string(parameters.surrogate_model)
    cls_members = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    for (model_name, model) in cls_members:
        if issubclass(model, SurrogateModelBase):
            model: Type[SurrogateModelBase]
            if model.name() == to_find:
                return model(parameters)
    raise NotImplementedError(
        f'Cannot find model with name "{parameters.surrogate_model}"')


'''
####################
# Special models

class LQR2_SurrogateModel(SurrogateModelBase):
    # TODO: Adjusted R^2 to switch between models
    # TODO: Interaction only for PolynomialFeatures as an option
    # TODO: Add ^3
    pass
'''
