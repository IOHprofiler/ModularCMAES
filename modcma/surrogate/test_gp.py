from abc import abstractmethod, ABCMeta

import matplotlib.pyplot as plt
import numpy as np

import os

#from typing import override


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
    def __init__(self, kernel):
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
    def __init__(self, settings):

        pass

    @abstractmethod
    def train(self,
              observation_index_points,
              observations,
              model: _ModelBuildingBase):
        pass


class _ModelTrainingBase_MaximumLikelihood(metaclass=ABCMeta):
    LEARNING_RATE: float = 0.01
    MAX_ITERATIONS: int = 1000

    EARLY_STOPPING_DELTA = 1e-6
    EARLY_STOPPING_PATIENCE = 20

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
    KERNEL = ???

    def __init__(self):
        pass


class GP_RegressionModel(SurrogateModelBase):
    ModelName = 'GP_Base'

    def __init__(self, d, kernel_cls):
        self.d = d

        # create variables ...
        self._kernel_obj = kernel_cls(self)
        self._kernel = self._kernel_obj.kernel()

        self.observation_noise_variance = create_positive_variable(
            1., name='observation_noise_variance'
        )

        self.observation_index_points = None
        self.observations = None

    @abstractmethod
    def _fit(self, X: XType, F: YType, W: YType) -> None:
        self.observation_index_points = tf.constant(X, dtype=tf.float64)
        self.observations = tf.constant(F, dtype=tf.float64)

        # TODO - how can you weight the GP ??
        # self.weights = tf.constant(W, tf.float64)
        pass


'''
    @abstractmethod
    def _predict(self, X: XType) -> YType:
        return np.tile(np.nan, (len(X), 1))

    @abstractproperty
    def df(self) -> int:
        return 0
'''
    @abstractmethod
    def train_model(self, observation_index_points, observations) -> None:
        pass

    def _generate_train_model(self):
        self.gp = tfd.GaussianProcess(
            kernel=self._kernel,
            index_points=self.observation_index_points,
            observation_noise_variance=self.observation_noise_variance
        )
    def _generate_regression_model(self, X):
        self.gprm = tfd.GaussianProcessRegressionModel(
            kernel=self.gp.kernel,
            mean_fn=self.gp.mean_fn,
            index_points=X,
            observation_index_points=self.observation_index_points,
            observations=self.observations,
            observation_noise_variance=self.observation_noise_variance,
            predictive_noise_variance=self.observation_noise_variance,
        )


class GP_ModelBase(SurrogateModelBase):
    ModelName = 'Base_GP'

    def __init__(self, ):
        super().__init__(parameters)
        self.default_observation_noise_variance = tf.constant(np.exp(-5), dtype=tf.float64)
        self.observation_noise_variance = tfp.util.TransformedVariable(
            self.default_observation_noise_variance, tfb.Exp(), name='observation_noise_variance')

        self.kernel = self._kernel()

    @property
    def kernel(self):
        if self._kernel_obj is None:
            self.parameters.surrogate_m
        pass

    @property
    def df(self):
        return 0

    @abstractmethod
    def _fit(self, X: XType, F: YType, W: YType):
        self.observation_index_points = tf.constant(X, dtype=np.float64)
        self.observations = tf.constant(Y, dtype=np.float64)
        return

    def _generate_gp_model(self):
        self.gp = tfd.GaussianProcess(
            kernel=self.kernel,
            index_points=self.observation_index_points,
            observation_noise_variance=self.observation_noise_variance)


    def _generate_regression_model(self, X) -> tfp.distributions.GaussianProcessRegressionModel:
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=self.gp.kernel,
            mean_fn=self.gp.mean_fn,
            index_points=X,
            observation_index_points=self.observation_index_points,
            observations=self.observations,
            observation_noise_variance=self.observation_noise_variance)
        return gprm

    def _predict(self, X):
        gprm = self._generate_regression_model(X)
        return gprm.mean().numpy()

    def _predict_with_confidence(self, X):
        gprm = self._generate_regression_model(X)
        mean = gprm.mean().numpy()
        var = gprm.variance().numpy()
        return mean, var


class GP_ML_ModelBase(GP_ModelBase):
    ModelName = 'Base_GP_ML'

    @abstractmethod
    def reset(self):
        self.observation_noise_variance.assign(self.default_observation_noise_variance)

    def _fit(self, X, Y, W):
        super()._fit(X, Y, W)
        self.reset()

        gp = tfd.GaussianProcess(
            kernel=self.kernel,
            index_points=self.observation_index_points,
            observation_noise_variance=self.observation_noise_variance)

        optimizer = tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)

        #@tf.function
        def optimize_gaussian_process():
            with tf.GradientTape() as tape:
                loss = -gp.log_prob(self.observations)
                tf.print(loss)
            grads = tape.gradient(loss, gp.trainable_variables)
            optimizer.apply_gradients(zip(grads, gp.trainable_variables))
            return loss

        for i in range(1000):
            neg_log_likelihood = optimize_gaussian_process()
            if i % 100 == 0:
                print("Step {}: NLL = {}".format(i, neg_log_likelihood))
        print('-'*80)
        print(gp.trainable_variables)
        print('-'*80)


        

class GP_ML_ExponentiatedQuadratic(GP_ML_ModelBase):
    ModelName = 'ExponentiatedQuadratic'

    def __init__(self, *args, **kwargs):
        self.default_amplitude = tf.constant(1., dtype=tf.float64)
        self.default_length_scale = self.default_amplitude

        self.amplitude = tfp.util.TransformedVariable(
            self.default_amplitude, tfb.Exp(), name='amplitude')
        self.length_scale = tfp.util.TransformedVariable(
            self.default_length_scale, tfb.Exp(), name='length_scale')

        super().__init__(*args, **kwargs)

    def reset(self):
        self.amplitude.assign(self.default_amplitude)
        self.length_scale.assign(self.default_length_scale)
        super().reset()

    def _kernel(self):
        return psd_kernels.ExponentiatedQuadratic(self.amplitude, self.length_scale)


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
