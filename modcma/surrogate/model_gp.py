from abc import abstractmethod, ABCMeta
from typing import Tuple, Optional, Type, List
import operator


import numpy as np
from ..typing_utils import XType, YType

import tensorflow as tf
import tensorflow_probability as tfp

from .model import SurrogateModelBase
from ..parameters import Parameters

# import kernels
from gp_kernels import _basic_kernels, _functor_kernels, GP_kernel_concrete_base
for k in _basic_kernels + _functor_kernels:
    locals()[k.__name__] = k

# Stuff for statitc typing
MaternFiveHalves: Type[GP_kernel_concrete_base]
MaternOneHalf: Type[GP_kernel_concrete_base]
MaternThreeHalves: Type[GP_kernel_concrete_base]
RationalQuadratic: Type[GP_kernel_concrete_base]
ExponentiatedQuadratic: Type[GP_kernel_concrete_base]
ExpSinSquared: Type[GP_kernel_concrete_base]
Linear: Type[GP_kernel_concrete_base]
Quadratic: Type[GP_kernel_concrete_base]
Cubic: Type[GP_kernel_concrete_base]
Parabolic: Type[GP_kernel_concrete_base]
ExponentialCurve: Type[GP_kernel_concrete_base]
Constant: Type[GP_kernel_concrete_base]

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def create_positive_variable(default, dtype=tf.float64, name=None):
    if isinstance(default, (float, int)):
        default = tf.constant(default, dtype=dtype)

    bijector = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

    return tfp.util.TransformedVariable(
        initial_value=default,
        bijector=bijector,
        dtype=dtype,
        name=name,
    )


def create_constant(default, dtype=tf.float64, name: Optional[str] = None):
    return tf.constant(default, dtype=dtype, name=name)

# ###############################################################################
# ### MODEL BUILDING COMPOSITION MODELS
# ###############################################################################


class _ModelBuildingBase(metaclass=ABCMeta):
    def __init__(self, parameters, kernel):
        self.parameters = parameters
        self.kernel = kernel
        self.mean_fn = None

    @abstractmethod
    def build_for_training(self,
                           observation_index_points=None,
                           observations=None) -> tfp.distributions.GaussianProcess:
        self.observation_index_points = observation_index_points
        self.observations = observations

    @abstractmethod
    def build_for_regression(self,
                             X,
                             observation_index_points=None,
                             observations=None
                             ) -> tfp.distributions.GaussianProcessRegressionModel:
        pass

    @staticmethod
    def create_class(parameters: Parameters):
        if parameters.surrogate_model_gp_noisy_samples:
            return _ModelBuilding_Noisy
        else:
            return _ModelBuilding_Noiseless


class _ModelBuilding_Noiseless(_ModelBuildingBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_noise_variance = create_constant(0.)

    def build_for_training(self,
                           observation_index_points=None,
                           observations=None):
        super().build_for_training(observation_index_points, observations)

        return tfd.GaussianProcess(
            kernel=self.kernel,
            mean_fn=self.mean_fn,
            index_points=self.observation_index_points,
            observation_noise_variance=self.observation_noise_variance
        )

    def build_for_regression(self,
                             X,
                             observation_index_points=None,
                             observations=None):
        observation_index_points = observation_index_points or self.observation_index_points
        observations = observations or self.observations

        return tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            mean_fn=self.mean_fn,
            index_points=X,
            observation_index_points=observation_index_points,
            observations=observations,
            observation_noise_variance=self.observation_noise_variance,
            predictive_noise_variance=self.observation_noise_variance,
        )


class _ModelBuilding_Noisy(_ModelBuilding_Noiseless):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_noise_variance = \
            create_positive_variable(1., name='observation_noise_variance')


# ###############################################################################
# ### MODEL TRAINING COMPOSITION MODELS
# ###############################################################################


class _ModelTrainingBase(metaclass=ABCMeta):
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    @abstractmethod
    def train(self,
              observation_index_points,
              observations,
              model: _ModelBuildingBase):
        pass

    @staticmethod
    def create_class(parameters: Parameters):
        return _ModelTraining_MaximumLikelihood


class _ModelTraining_MaximumLikelihood(_ModelTrainingBase):
    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        self.LEARNING_RATE = self.parameters.surrogate_model_gp_learning_rate
        self.MAX_ITERATIONS = self.parameters.surrogate_model_gp_max_iterations
        self.EARLY_STOPPING_DELTA = self.parameters.surrogate_model_gp_early_stopping_delta
        self.EARLY_STOPPING_PATIENCE = self.parameters.surrogate_model_gp_early_stopping_patience

    def train(self,
              observation_index_points,
              observations,
              model: _ModelBuildingBase):
        ''' '''
        gp = model.build_for_training(observation_index_points, observations)
        optimizer = tf.optimizers.Adam(learning_rate=self.LEARNING_RATE)

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                loss = -gp.log_prob(observations)
            grads = tape.gradient(loss, gp.trainable_variables)
            optimizer.apply_gradients(zip(grads, gp.trainable_variables))
            return loss

        minimal_neg_log_likelihood = np.inf
        minimal_index = 0

        neg_log_likelihood = np.nan
        for i in range(self.MAX_ITERATIONS):
            neg_log_likelihood = step()
            neg_log_likelihood = neg_log_likelihood.numpy()
            # nan
            if np.isnan(neg_log_likelihood):
                break

            if minimal_neg_log_likelihood - self.EARLY_STOPPING_DELTA > neg_log_likelihood:
                minimal_neg_log_likelihood = neg_log_likelihood
                minimal_index = i
            elif minimal_index + self.EARLY_STOPPING_PATIENCE < i:
                break
        return neg_log_likelihood

# ###############################################################################
# ### MODELS
# ###############################################################################

class _GaussianProcessModel:
    ''' gaussian process wihtout known kernel '''

    def __init__(self, parameters: Parameters, kernel_cls):
        self.parameters = parameters
        self.KERNEL_CLS = kernel_cls

        self.MODEL_GENERATION_CLS = _ModelBuildingBase.create_class(self.parameters)
        self.MODEL_TRAINING_CLS = _ModelTrainingBase.create_class(self.parameters)

    def _fit(self, X: XType, F: YType, W: YType):
        # kernel
        self._kernel_obj = self.KERNEL_CLS(self.parameters)
        self._kernel = self._kernel_obj.kernel()

        self.model_generation = self.MODEL_GENERATION_CLS(self.parameters, self._kernel)
        self.model_training = self.MODEL_TRAINING_CLS(self.parameters)

        self._train_loss = self.model_training.train(
            observation_index_points=X, observations=F, model=self.model_generation)
        return self

    def _predict(self, X: XType) -> YType:
        gprm = self.model_generation.build_for_regression(X)
        return gprm.mean().numpy()

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        gprm = self.model_generation.build_for_regression(X)
        mean = gprm.mean().numpy()
        stddev = gprm.stddev().numpy()
        return mean, stddev

    @property
    def loss(self):
        return self._train_loss

    def df(self) -> int:
        return 0


class GaussianProcess(_GaussianProcessModel, SurrogateModelBase):
    def __init__(self, parameters: Parameters):
        SurrogateModelBase.__init__(self, parameters)
        KERNEL_CLS = eval(self.parameters.surrogate_model_gp_kernel)

        _GaussianProcessModel.__init__(self, parameters, KERNEL_CLS)


class GaussianProcessBasicSelection(SurrogateModelBase):
    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        # the selection ...
        self._model_classes: List[Type[GP_kernel_concrete_base]] = [
            MaternFiveHalves,
            MaternOneHalf,
            MaternThreeHalves,
            RationalQuadratic,
            ExponentiatedQuadratic,
            ExpSinSquared,
            Linear,
            Quadratic,
            #Cubic,
            #Parabolic,
            #ExponentialCurve,
            #Constant,
        ]

    def _partial_fit(self, kernel_cls, X, F, W) -> _GaussianProcessModel:
        return _GaussianProcessModel(self.parameters, kernel_cls)._fit(X, F, W)

    def _full_fit(self, X, F, W):
        return [self._partial_fit(k, X, F, W) for k in self._model_classes]

    def _fit(self, X: XType, F: YType, W: YType):
        models = self._full_fit(X, F, W)
        losses = np.array(list(map(operator.attrgetter('loss'), models)))
        best_index = np.nanargmin(losses)
        self.best_model = models[best_index]
        return self

    def _predict(self, X: XType) -> YType:
        return self.best_model._predict(X)

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        return self.best_model._predict_with_confidence(X)


class GaussianProcessPenalizedSelection(SurrogateModelBase):
    TRAIN_MAX_MODELS = 30
    TRAON_MAX_TIME_S = 120

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        # the selection ...
        self._base_classes: List[Type[GP_kernel_concrete_base]] = [
            MaternFiveHalves,
            MaternOneHalf,
            MaternThreeHalves,
            RationalQuadratic,
            ExponentiatedQuadratic,
            ExpSinSquared,
            Linear,
        ]

    def _fit(self, X: XType, F: YType, W: YType) -> None:

        self.best_model;
        return super()._fit(X, F, W)

    def _predict(self, X: XType) -> YType:
        return self.best_model._predict(X)

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        return self.best_model._predict_with_confidence(X)



if __name__ == '__main__':
    import unittest

    class TestBasic(unittest.TestCase):
        def test_gp_init(self):
            parameters = Parameters(3)
            model = GaussianProcess(parameters)
def test_gp_linear(self):
            ''' L should be linear '''
            parameters = Parameters(3)

            X = np.random.rand(200, 3)
            Y = X[:,0]*2. + X[:,1] - X[:,2]

            model = GaussianProcess(parameters)
            model.fit(X, Y)

            Xt = np.random.randn(10, 3)
            Yt = Xt[:,0]*2. + Xt[:,1] - Xt[:,2]
            Yp = model.predict(Xt)

            for i in range(len(Xt)):
                self.assertAlmostEqual(Yp[i], Yt[i], places=2)

        def test_multiple_kernel_LL(self):
            ''' L + L = L '''
            parameters1 = Parameters(2)
            parameters2 = Parameters(2)
            parameters2.surrogate_model_gp_kernel = 'Linear + Linear'

            X  = np.random.rand(200, 2)
            Xt = np.random.rand(30, 2)
            Y  = X[:,0]*2. - X[:,1]*3
            Yt = Xt[:,0]*2. - Xt[:,1]*3

            model1 = GaussianProcess(parameters1)
            model2 = GaussianProcess(parameters2)
            model1.fit(X, Y)
            model2.fit(X, Y)

            p1 = model1.predict(Xt)
            p2 = model2.predict(Xt)
            for i in range(len(Xt)):
                self.assertAlmostEqual(p1[i], p2[i], places=2)
                self.assertAlmostEqual(p1[i], Yt[i], places=2)

        def test_multiple_kernel_LL_Q(self):
            ''' L * L = Q '''
            parameters1 = Parameters(2)
            parameters2 = Parameters(2)
            parameters1.surrogate_model_gp_kernel = 'Quadratic + Linear'
            parameters2.surrogate_model_gp_kernel = 'Linear * Linear'

            X  = np.random.rand(200, 2) * 3
            Xt = np.random.rand(30, 2)
            Y  = X[:,0]*2. - X[:,1]*3 + X[:,0]**2 + X[:,0]*X[:,1]
            Yt  = Xt[:,0]*2. - Xt[:,1]*3 + Xt[:,0]**2 + Xt[:,0]*Xt[:,1]

            model1 = GaussianProcess(parameters1)
            model2 = GaussianProcess(parameters2)
            model1.fit(X, Y)
            model2.fit(X, Y)

            p1 = model1.predict(Xt)
            p2 = model2.predict(Xt)
            for i in range(len(Xt)):
                self.assertAlmostEqual(p1[i], p2[i], places=2)
                self.assertAlmostEqual(p1[i], Yt[i], places=2)



    unittest.main()





