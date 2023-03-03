
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from modcma.surrogate.model_gp import *

import unittest
import math

if False:
    # For debugging purposes
    import matplotlib.pyplot as plt

    parameters = Parameters(1)
    parameters.learning_rate = 0.10
    parameters.surrogate_model_gp_max_iterations = 10000000
    parameters.surrogate_model_gp_early_stopping_patience = 3
    parameters.surrogate_model_gp_early_stopping_delta = 0.1
    parameters.surrogate_model_gp_noisy_samples = False
    #parameters.surrogate_model_gp_noisy_samples = True
    parameters.surrogate_model_gp_kernel = 'ExpSinSquared'
    #parameters.surrogate_model_gp_kernel = 'MaternOneHalf'


    def function(x):
        return -0.3 + np.abs(np.sin(x/3.5 + 0.12)) * 0.5

    data = np.random.rand(72, 1) * 100
    target = function(data)

    model = GaussianProcess(parameters)
    model.fit(data, target)

    test_data = np.linspace(0, 100., 1000)[:, np.newaxis]
    test_target = function(test_data)
    test_prediction = model.predict(test_data)

    plt.plot(test_data, test_target)
    plt.plot(test_data, test_prediction)
    plt.scatter(data, target)
    plt.show()
    exit(0)


class TestBasic(unittest.TestCase):
    def setUp(self) -> None:
        #tf.get_logger().setLevel('INFO')
        pass


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


class Test_GaussianProcessBasicSelection(unittest.TestCase):
    def test_quadratic(self):
        data = np.random.rand(50, 1)
        target = data[:,0]**2

        parameters = Parameters(1)
        parameters.surrogate_model_gp_noisy_samples = False
        model = GaussianProcessBasicSelection(parameters)
        model.fit(data, target)

        self.assertIsInstance(model.best_model._kernel_obj, Quadratic)

    def test_linear(self):
        data = np.random.rand(50, 2)
        target = data[:,0] + data[:,1] * 10

        parameters = Parameters(2)
        parameters.surrogate_model_gp_noisy_samples = False
        model = GaussianProcessBasicSelection(parameters)
        model.fit(data, target)

        self.assertIsInstance(model.best_model._kernel_obj, Linear)

    def test_sin(self):
        data = np.random.rand(272, 1) * 3.14 * 100

        target = np.sin(data/4 + 0.14)

        parameters = Parameters(1)
        model = GaussianProcessBasicSelection(parameters)
        model.fit(data, target)

        self.assertIsInstance(model.best_model._kernel_obj, ExpSinSquared)

    def test_generator(self):
        parameters = Parameters(1)
        model = GaussianProcessBasicSelection(parameters)
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks)
        )


class Test_GaussianProcessExtendedSelection(unittest.TestCase):
    def combination_number(self, a):
        return math.factorial(a + 2 - 1) // math.factorial(2) // math.factorial(a - 1)

    def test_generator_additive(self):
        parameters = Parameters(1)
        model = GaussianProcessBasicSelection(parameters)
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks) +
            self.combination_number(len(model._building_blocks)) - 2
            # Lin + Lin == Lin
            # Qua + Qua == Qua
        )

    def test_generator_multiplicative(self):
        parameters = Parameters(1)
        model = GaussianProcessBasicSelection(parameters)
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks) +
            self.combination_number(len(model._building_blocks)) - 1
            # Lin + Lin == Qua
        )

    def test_generator_both(self):
        parameters = Parameters(1)
        model = GaussianProcessBasicSelection(parameters)
        self.assertEqual(
            len(list(model._generate_kernel_space())),
            len(model._building_blocks) +
            self.combination_number(len(model._building_blocks)) - 3
        )


if __name__ == '__main__':
    unittest.main()
