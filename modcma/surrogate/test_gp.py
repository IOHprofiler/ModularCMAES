from abc import abstractmethod, ABCMeta
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import os

from ..parameters import Parameters

from tensorflow.python.ops.variable_scope import init_ops
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_probability as tfp

from .model import SurrogateModelBase
from ..typing_utils import XType, YType

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


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


def create_constant(default, dtype=tf.float64, name=None):
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
        self.observation_index_points = observation_index_points or self.observation_index_points
        self.observations = observations or self.observations

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
        super().build_for_regression(X, observation_index_points, observations)
        return tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            mean_fn=self.mean_fn,
            index_points=X,
            observation_index_points=self.observation_index_points,
            observations=self.observations,
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


# ###############################################################################
# ### MODELS
# ###############################################################################

class GaussianProcess(SurrogateModelBase):
    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        self.KERNEL_CLS = self.parameters.surrogate_model_gp_kernel
        self.KERNEL_CLS = eval(self.KERNEL_CLS)

        self.MODEL_GENERATION_CLS = _ModelBuildingBase.create_class(parameters)
        self.MODEL_TRAINING_CLS = _ModelTrainingBase.create_class(parameters)


    def _fit(self, X: XType, F: YType, W: YType) -> None:
        # kernel
        self._kernel_obj = self.KERNEL_CLS(self.parameters)
        self._kernel = self._kernel_obj.kernel()

        self.model_generation = self.MODEL_GENERATION_CLS(self.parameters, self._kernel)
        self.model_training = self.MODEL_TRAINING_CLS(self.parameters)

        gp = self.model_generation.build_for_training(
                observation_index_points=X,
                observations=F)

        self.model_training.train(observation_index_points=X, observations=F, model=gp)

    def _predict(self, X: XType) -> YType:
        gprm = self.model_generation.build_for_regression(X)
        return gprm.mean().numpy()

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        gprm = self.model_generation.build_for_regression(X)
        mean = gprm.mean().numpy()
        stddev = gprm.stddev().numpy()
        return mean, stddev

    def df(self) -> int:
        return 0






        



'''
from ..parameters import Parameters

p = Parameters(1)
gp = GP_ML_ExponentiatedQuadratic(p)

f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
X = np.random.uniform(-1., 1., 50)[..., np.newaxis]
Y = f(X) + np.random.normal(0., .05, 50)


gp.fit(X, Y)

index_points = np.linspace(-1., 1., 100)[..., np.newaxis]
#samples = gp._predict(index_points).sample(10).numpy()
samples, var = gp._predict_with_confidence(index_points)


import matplotlib.pyplot as plt

plt.scatter(np.squeeze(X), Y)
plt.plot(index_points, samples, alpha = 1.)
plt.plot(index_points, samples - np.sqrt(var)*3, alpha = .5)
plt.plot(index_points, samples + np.sqrt(var)*3, alpha = .5)

#plt.plot(np.stack([index_points[:, 0]]*10).T, samples.T, c='r', alpha=.2)
plt.show()
'''
