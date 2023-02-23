from abc import abstractmethod, ABCMeta
from typing import Tuple, Optional, Type, List, Generator
import operator
import itertools
import time

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from modcma.typing_utils import XType, YType
from modcma.surrogate.model import SurrogateModelBase
from modcma.parameters import Parameters

# import kernels
from gp_kernels import basic_kernels, functor_kernels, GP_kernel_concrete_base
for k in basic_kernels + functor_kernels:
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
              model: _ModelBuildingBase) -> float:
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
        return float(neg_log_likelihood)

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
        self._train_loss = float(np.nan)

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
    def loss(self) -> float:
        return self._train_loss

    @loss.setter
    def loss(self, other):
        self._train_loss = other

    def df(self) -> int:
        return 0


class GaussianProcess(_GaussianProcessModel, SurrogateModelBase):
    def __init__(self, parameters: Parameters):
        SurrogateModelBase.__init__(self, parameters)
        KERNEL_CLS = eval(self.parameters.surrogate_model_gp_kernel)

        _GaussianProcessModel.__init__(self, parameters, KERNEL_CLS)


class _GaussianProcessModelMixtureBase:
    TRAIN_MAX_MODELS: Optional[int] = None
    TRAIN_MAX_TIME_S: Optional[int] = None

    def __init__(self, parameters: Parameters) -> None:
        self.parameters = parameters

        # the selection ...
        self._building_blocks: List[Type[GP_kernel_concrete_base]] = [
            MaternFiveHalves,
            MaternOneHalf,
            MaternThreeHalves,
            RationalQuadratic,
            ExponentiatedQuadratic,
            ExpSinSquared,
            Linear,
            Quadratic,
            ####
            #Cubic,
            #Parabolic,
            #ExponentialCurve,
            #Constant,
        ]

    def _partial_fit(self, kernel_cls, X, F, W) -> _GaussianProcessModel:
        return _GaussianProcessModel(self.parameters, kernel_cls)._fit(X, F, W)

    def _penalized_partial_fit(self, kernel_cls, X, F, W) -> _GaussianProcessModel:
        gpm = self._partial_fit(kernel_cls, X, F, W)
        gpm.loss = self.penalize_kernel(gpm.loss, gpm._kernel_obj)
        return gpm

    def restricted_full_fit(self, X, F, W):
        time_start = time.time()
        trained_models = []

        iterator = self.generate_kernel_space()
        if self.TRAIN_MAX_MODELS:
            iterator = itertools.islice(iterator, self.TRAIN_MAX_MODELS)

        for kernel_cls in iterator:
            model = self._penalized_partial_fit(kernel_cls, X, F, W)
            trained_models.append(model)

            if self.TRAIN_MAX_TIME_S:
                if time.time() - time_start > self.TRAIN_MAX_TIME_S:
                    break

        self.best_model = self.restricted_full_fit_postprocessing(trained_models)

    def restricted_full_fit_postprocessing(self, trained_models: List[_GaussianProcessModel]):
        losses = np.array(list(map(operator.attrgetter('loss'), trained_models)))
        best_index = np.nanargmin(losses)
        return trained_models[best_index]

    @abstractmethod
    def penalize_kernel(self, loss: float, kernel_obj: GP_kernel_concrete_base):
        return loss

    @abstractmethod
    def generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        return self._building_blocks


class GaussianProcessBasicSelection(SurrogateModelBase, _GaussianProcessModelMixtureBase):
    def __init__(self, parameters: Parameters):
        SurrogateModelBase.__init__(self, parameters)
        _GaussianProcessModelMixtureBase.__init__(self, parameters)

    def _fit(self, X: XType, F: YType, W: YType):
        self.restricted_full_fit(X, F, W)
        return self

    def _predict(self, X: XType) -> YType:
        return self.best_model._predict(X)

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        return self.best_model._predict_with_confidence(X)

    def penalize_kernel(self, loss, kernel_obj):
        return super().penalize_kernel(loss, kernel_obj)

    def generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        return super().generate_kernel_space()

    def df(self):
        return 0


class GaussianProcessBasicAdditiveSelection(GaussianProcessBasicSelection):
    def generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super().generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) in [(Linear, Linear), (Quadratic, Quadratic), ]:
                continue
            yield a + b


class GaussianProcessBasicMultiplicativeSelection:
    def generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super().generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) in [(Linear, Linear), ]:
                continue
            yield a * b


class GaussianProcessBasicBinarySelection:
    def generate_kernel_space(self) -> Generator[Type[GP_kernel_concrete_base], None, None]:
        yield from super().generate_kernel_space()
        for (a, b) in itertools.combinations_with_replacement(self._building_blocks, 2):
            if (a, b) not in [(Linear, Linear), ]:
                yield a * b
            if (a, b) not in [(Linear, Linear), (Quadratic, Quadratic), ]:
                yield a + b





class GaussianProcessPenalizedAdditiveSelection(GaussianProcessBasicSelection):
    def penalize_kernel(self, loss, kernel_obj):
        return super().penalize_kernel(loss, kernel_obj)

    def _predict(self, X: XType) -> YType:
        return self.best_model._predict(X)

    def _predict_with_confidence(self, X: XType) -> Tuple[YType, YType]:
        return self.best_model._predict_with_confidence(X)
