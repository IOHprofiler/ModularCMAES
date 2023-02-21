
import unittest
import math
from numpy.testing import assert_array_equal, assert_array_almost_equal

from modcma.surrogate.model import *

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

        self.assertEqual(0, self.model.i_model)
        self.assertEqual(3, self.model.df)

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

        self.assertEqual(0, self.model.i_model)
        self.assertEqual(3, self.model.df)

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

        self.assertEqual(0, self.model.i_model)
        self.assertEqual(3, self.model.df)

    def test_2_quadratic_pure_exist_quad(self):
        p = Parameters(2)
        p.surrogate_model = 'LQ'
        self.model = get_model(p)

        X = np.array([[0, 0], [1, 0],
                      [2, 0], [3, 0],
                      [1, 1], [2, 2],
                      ])
        Y = 11*X[:, 0]**2 + 3*X[:, 1] + 1
        self.train_try_model(X, Y)

        self.assertEqual(1, self.model.i_model)
        self.assertEqual(5, self.model.df)

        X = np.array([[-1, 0], [-2, 1]])
        Y = 11*X[:, 0]**2 + 3*X[:, 1] + 1
        self.try_model(X, Y)

    def test_2_quadratic_pure_ne_exist(self):
        p = Parameters(2)
        p.surrogate_model = 'LQ'
        self.model = get_model(p)

        X = np.array([[0, 0], [1, 0],
                      [2, 0], [3, 0],
                      [1, 1], [2, 2],
                      ])
        Y = + 11*X[:, 0]**2 \
            + 31*X[:, 1]**2 \
            + 3*X[:, 0] \
            + 2*X[:, 1] \
            - 7
        self.train_try_model(X, Y)
        self.assertEqual(1, self.model.i_model)
        self.assertEqual(5, self.model.df)
        self.assertEqual(
            self.model.df,
            len(self.model.model['linearregression'].coef_) + 1
        )

        X = np.array([[0, 0], [1, 0],
                      [2, 0], [0, 1],
                      [1, 2], [3, 2],
                      ])
        Y = + 11*X[:, 0]**2 \
            + 31*X[:, 1]**2 \
            + 3*X[:, 0] \
            + 2*X[:, 1] \
            - 7 \
            + 5*X[:, 0]*X[:, 1] \

        self.train_model(X, Y)
        self.try_ne_model(X, Y)

    def test_3_quadratic_full(self):
        p = Parameters(3)
        p.surrogate_model = 'LQ'
        self.model = get_model(p)

        X = np.random.randn(12, 3)
        Y = + 11*X[:, 0]**2 \
            + 31*X[:, 1]**2 \
            + 17*X[:, 2]**2 \
            + 3*X[:, 0] \
            + 2*X[:, 1] \
            + 5*X[:, 2] \
            + 13*X[:, 0]*X[:, 1] \
            + 19*X[:, 0]*X[:, 2] \
            + 23*X[:, 1]*X[:, 2] \
            - 7
        self.train_try_model(X, Y)
        self.assertEqual(2, self.model.i_model)
        self.assertEqual(10, self.model.df)
        self.assertEqual(
            self.model.df,
            len(self.model.model['linearregression'].coef_) + 1
        )

    def test_changes(self):
        def LM_th():
            return (4 + 1)
            # linear + intercept

        def QP_th():
            return (4 + 4 + 1)
            # quadratic + linear + intercept

        def QF_th():
            return (6 + 4 + 4 + 1)
            # all combinations + quadratic + linear + intercept

        def train_and_return(N):
            X, Y = np.random.randn(N, 4), np.random.randn(N, 1)
            self.train_model(X, Y)
            return self.model.i_model

        def myfloor(th):
            thf = math.floor(th)
            if th == thf:
                return thf - 1
            return thf
        p = Parameters(4)
        p.surrogate_model = 'LQ'


        for margin in [1.5, 2.0, 1.]:
            p.surrogate_model_lq_margin = margin
            self.model = get_model(p)

            #N = 1
            #self.assertEqual(0, train_and_return(N))
            N = myfloor(LM_th() * margin)
            self.assertEqual(0, train_and_return(N))
            N += 1
            self.assertEqual(0, train_and_return(N))
            N = myfloor(QP_th() * margin)
            self.assertEqual(0, train_and_return(N))
            N += 1
            self.assertEqual(1, train_and_return(N))
            N = myfloor(QF_th() * margin)
            self.assertEqual(1, train_and_return(N))
            N += 1
            self.assertEqual(2, train_and_return(N))

if __name__ == '__main__':
    unittest.main(verbosity=2)
