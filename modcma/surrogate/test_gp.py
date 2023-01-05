from abc import abstractmethod
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp


from .model import SurrogateModelBase

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


class GP_ML_OptimizerBase(SurrogateModelBase):
    ModelName = 'GP'

    def __init__(self):
        self.default_observation_noise_variance = tf.constant(np.exp(-5), dtype=tf.float64)
        self.observation_noise_variance = tfp.util.TransformedVariable(
            self.default_observation_noise_variance, tfb.Exp(), name='observation_noise_variance')

        self.kernel = self._kernel()

    @property
    def df(self):
        return 0

    @abstractmethod
    def _kernel(self):
        pass

    @abstractmethod
    def reset(self):
        self.observation_noise_variance.assign(self.default_observation_noise_variance)

    def _fit(self, X, Y):
        self.reset()
        self.observation_index_points = tf.constant(X, dtype=np.float64)
        self.observations = tf.constant(Y, dtype=np.float64)

        gp = tfd.GaussianProcess(
            kernel=self.kernel,
            index_points=self.observation_index_points,
            observation_noise_variance=self.observation_noise_variance)

        optimizer = tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)

        @tf.function
        def optimize_gaussian_process():
            with tf.GradientTape() as tape:
                loss = -gp.log_prob(self.observations)
            grads = tape.gradient(loss, gp.trainable_variables)
            optimizer.apply_gradients(zip(grads, gp.trainable_variables))
            return loss

        for i in range(10000):
            neg_log_likelihood = optimize_gaussian_process()
            if i % 100 == 0:
                print("Step {}: NLL = {}".format(i, neg_log_likelihood))
        print('-'*80)
        print(gp.trainable_variables)
        print('-'*80)

    def _generate_regression_model(self, X) -> tfp.distributions.GaussianProcessRegressionModel:
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
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

class GP_ML_ExponentiatedQuadratic(GP_ML_OptimizerBase):
    ModelName = 'ExponentiatedQuadratic'

    def __init__(self):
        self.default_amplitude = tf.constant(1., dtype=tf.float64)
        self.default_length_scale = self.default_amplitude

        self.amplitude = tfp.util.TransformedVariable(
            self.default_amplitude, tfb.Exp(), name='amplitude')
        self.length_scale = tfp.util.TransformedVariable(
            self.default_length_scale, tfb.Exp(), name='length_scale')

        super().__init__()

    def reset(self):
        self.amplitude.assign(self.default_amplitude)
        self.length_scale.assign(self.default_length_scale)
        super().reset()

    def _kernel(self):
        return psd_kernels.ExponentiatedQuadratic(self.amplitude, self.length_scale)




gp = GP_ML_ExponentiatedQuadratic()

f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
X = np.random.uniform(-1., 1., 50)[..., np.newaxis]
Y = f(X) + np.random.normal(0., .05, 50)


gp._fit(X, Y)

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
