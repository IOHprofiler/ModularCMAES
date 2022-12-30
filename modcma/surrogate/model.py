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

from ..parameters import Parameters
from ..typing_utils import XType, YType


####################
# Helper functions


class PureQuadraticFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> npt.NDArray[np.float64]:
        return np.hstack((X, np.square(X)))


def normalize_string(s: str):
    return s.lower().replace(' ', '_')


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
        self.fit

    def fit(self,
            X: Union[XType, None],
            F: Union[YType, None],
            W: Union[YType, None]) -> None:
        ''' fit the surrogate '''
        if X is None or F is None:
            self.fitted = False
            return

        X = normalize_X(X, self.parameters.d)
        F = normalize_F(F)

        if W is None:
            W = np.ones_like(F)
        else:
            W = normalize_W(W)
        self._fit(X, F, W)
        self.fitted = True

    @abstractmethod
    def _fit(self, X: XType, F: YType, W: YType):
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
    def fitted(self):
        return self._fitted

    @fitted.setter
    def fitted(self, value):
        self._fitted = False

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
        return Pipeline(ppl + [('linearregression', LinearRegression())])

    @override
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        self.model = self._select_model(D=X.shape[0], N=X.shape[1])
        self.model = self.model.fit(X, F, linearregression__sample_weight=W)

    @override
    def _predict(self, X: XType) -> YType:
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
    def df(self):
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
    def df(self):
        return 2*self.parameters.d + 1


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
    def df(self):
        return (self.parameters.d*(self.parameters.d + 1) + 2)//2


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
        return (self.parameters.d + 2)*(self.parameters.d + 1) // 2


def get_model(parameters: Parameters) -> SurrogateModelBase:
    to_find = normalize_string(parameters.surrogate_model)
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    for (modelname, model) in clsmembers:
        if issubclass(model, SurrogateModelBase):
            model: Type[SurrogateModelBase]
            if model.name() == to_find:
                return model(parameters)
    raise NotImplementedError(
        f'Cannot find model with name "{parameters.surrogate_strategy}"')


###################
# TESTS


if __name__ == '__main__':
    import unittest
    from numpy.testing import assert_array_equal, assert_array_almost_equal

    class Test_get_model(unittest.TestCase):
        def test_empty(self):
            p = Parameters(2)
            m = get_model(p)
            self.assertIsInstance(m, Linear_SurrogateModel)

        def test_Quandratic(self):
            p = Parameters(2)
            p.surrogate_model = 'Quadratic'
            m = get_model(p)
            self.assertIsInstance(m, Quadratic_SurrogateModel)

        def test_LQ(self):
            p = Parameters(2)
            p.surrogate_model = 'LQ'
            m = get_model(p)
            self.assertIsInstance(m, LQ_SurrogateModel)

    class TestModelsBase(unittest.TestCase):
        def train_model(self, X, Y):
            self.model: SurrogateModelBase
            self.model.fit(X, Y, None)

        def train_try_model(self, X, Y):
            self.train_model(X, Y)
            self.try_model(X, Y)

        def try_model(self, X, Y):
            Yt = self.model.predict(X)
            self.assertIsNone(assert_array_almost_equal(Y, Yt))

        def try_ne_model(self, X, Y):
            Yt = self.model.predict(X)
            self.assertFalse(np.allclose(Y, Yt))

    class Test_Linear_SurrogateModel(TestModelsBase):
        def test_1(self) -> None:
            p = Parameters(2)
            self.model = get_model(p)

            X = np.array([
                [1, 4], [2, 2],
                [3, 7], [4, 0],
            ])
            Y = X[:,0] + X[:,1]*3 + 1
            self.train_try_model(X, Y)

            X = np.array([[2, 2], [0, 0]])
            Y = np.array([9, 1])
            self.try_model(X, Y)

    class Test_QuadraticPure_SurrogateModel(TestModelsBase):
        def test_1(self) -> None:
            p = Parameters(1)
            p.surrogate_model = 'QuadraticPure'
            self.model = get_model(p)

            X = np.array([
                [1], [2], [3], [4],
            ])
            Y = np.array([3, 7,  13, 21])
            self.train_try_model(X, Y)

            X = np.array([[5], [6]])
            Y = np.array([31, 43])
            self.try_model(X, Y)

    class Test_QuadraticInteraction_SurrogateModel(TestModelsBase):
        def test_1(self) -> None:
            p = Parameters(2)
            p.surrogate_model = 'QuadraticInteraction'
            self.model = get_model(p)

            X = np.array([
                [0, 0], [1, 0], [0, 1], [1, 1]
            ])
            Y = np.array([1, 3, 4, 11])
            self.train_try_model(X, Y)

            X = np.array([[5, 2], [3, 4]])
            Y = np.array([67, 79])
            self.try_model(X, Y)

    class Test_Quadratic_SurrogateModel(TestModelsBase):
        def test_1(self) -> None:
            # Intercept=1, 2 3 5 7 11
            p = Parameters(2)
            p.surrogate_model = 'Quadratic'
            self.model = get_model(p)

            X = np.array([
                [0, 0], [1, 0], [0, 1], [1, 1], [3, 0], [0, 3]
            ])
            Y = np.array([1, 8, 11, 29, 52, 73])
            self.train_try_model(X, Y)

            X = np.array([[3, 1], [1, 3]])
            Y = np.array([95, 113])
            self.try_model(X, Y)

            X = np.array([[5, 1], [8, 9]])
            Y = np.array([201, 1723])
            self.try_model(X, Y)

    class Test_LQ_SurrogateModel(TestModelsBase):
        def test_1_small(self):
            p = Parameters(2)
            p.surrogate_model = 'LQ'
            self.model = get_model(p)

            X = np.array([[0, 0], [1, 0]])
            Y = np.array([1, 3])
            self.train_try_model(X, Y)

            X = np.array([[-1, 0], [2, 3]])
            Y = np.array([-1, 5])
            self.try_model(X, Y)

        def test_2_full_linear(self):
            p = Parameters(2)
            p.surrogate_model = 'LQ'
            self.model = get_model(p)

            X = np.array([[0, 0], [1, 0], [0, 1]])
            Y = np.array([1, 3, 2])
            self.train_try_model(X, Y)

            X = np.array([[-1, 0], [2, 3]])
            Y = np.array([-1, 8])
            self.try_model(X, Y)

        def test_2_full_linear_neg_case(self):
            p = Parameters(2)
            p.surrogate_model = 'LQ'
            self.model = get_model(p)

            X = np.array([[0, 0], [1, 0], [2, 0]])
            Y = np.array([1, 1+1, 1+4])
            self.train_model(X, Y)

            X = np.array([[3, 0], [1, 0]])
            Y = np.array([1+9, 1+1])
            self.try_ne_model(X, Y)

        @unittest.skip("TODO")
        def test_2_quadratic_part(self):
            p = Parameters(2)
            p.surrogate_model = 'LQ'
            self.model = get_model(p)

            X = np.array([[0, 0], [1, 0], [0, 1], []])
            Y = np.array([1, 3, 2])
            self.train_try_model(X, Y)

            X = np.array([[-1, 0], [2, 3]])
            Y = np.array([-1, 8])
            self.try_model(X, Y)

    unittest.main(verbosity=2)


'''
####################
# Special models

class LQR2_SurrogateModel(SurrogateModelBase):
    # TODO: Adjusted R^2 to switch between models
    # TODO: Interaction only for PolynomialFeatures as an option
    # TODO: Add ^3
    pass
'''
