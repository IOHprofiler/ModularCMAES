import numpy as np
import math
from abc import ABCMeta, abstractmethod

from scipy.stats import kendalltau
import pulp

import functools
import scipy.optimize

def singleton_args(cls):
    instances = {}

    def wrapper(*args):
        if args not in instances:
            instances[args] = cls(*args)
        return instances[args]

    return wrapper


_all_loss_classes = {}

class _RegisterLossMeta(ABCMeta):
    def __new__(cls, clsname, bases, attrs):
        newclass = super().__new__(cls, clsname, bases, attrs)
        if hasattr(newclass, 'name'):
            assert(newclass.name not in _all_loss_classes)
            _all_loss_classes[newclass.name] = newclass
            return newclass
        return newclass

class _MakeFull(_RegisterLossMeta):
    def __new__(cls, clsname, bases, attrs):
        def __init__(self):
            pass

        def __call__(self, predict, target, **kwargs):
            self.mu = len(target)
            return bases[0].__call__(self, predict, target)

        if 'name' not in attrs:
            assert(len(bases) == 1)
            assert(hasattr(bases[0], 'name'))
            newname = bases[0].name + 'Full'
            attrs['name'] = newname
        newclass = super().__new__(cls, clsname, bases, attrs)

        setattr(newclass, '__init__', __init__)
        setattr(newclass, '__call__', __call__)
        return newclass


class _MakeAuto(_RegisterLossMeta):
    def __new__(cls, clsname, bases, attrs):
        def __init__(self):
            pass

        def __call__(self, predict, target, **kwargs):
            self.mu = max(len(target) // 2, 1)
            return bases[0].__call__(self, predict, target)

        if 'name' not in attrs:
            assert(len(bases) == 1)
            assert(hasattr(bases[0], 'name'))
            newname = bases[0].name + 'Auto'
            attrs['name'] = newname
        newclass = super().__new__(cls, clsname, bases, attrs)

        setattr(newclass, '__init__', __init__)
        setattr(newclass, '__call__', __call__)
        return newclass


@singleton_args
class OrderSolver:
    def __init__(self, variables: int):
        #print(f'Created OrderSolver: {variables}')
        self.variables = variables

        # objective function == sum of absolute values
        self.c = np.concatenate(
            [np.zeros(variables), np.ones(variables)])

        # bounds
        self.bounds = [(None, None)] * variables + [(0., None)] * variables

        # absolute values - inequalities
        e = np.eye(variables)
        A_ub_abs = np.block([[+e, -e], [-e, -e]])

        self.b_ub_abs = np.zeros(variables*2)

        # order inequalities
        e = np.eye(variables - 1)
        g = np.zeros((variables - 1, 1))
        A_ub_order = np.hstack([
            np.hstack([e, g]) - np.hstack([g, e]),
            np.zeros((variables-1, variables))
        ])

        self.A_ub = np.vstack([A_ub_abs, A_ub_order])

    def __call__(self, vector):
        assert(len(vector) == self.variables)

        b_ub_order = vector[1:] - vector[:-1]
        b_ub = np.concatenate([self.b_ub_abs, b_ub_order])

        result = scipy.optimize.linprog(
            self.c,
            A_ub=self.A_ub,
            b_ub=b_ub,
            bounds=self.bounds,
            method='highs-ds',  # fast [1.08, 19.9]
            # method='highs-ipm', [slover, 4.33, 84.3]
            # method='highs', [very slow, 88s, ?]
        )

        return np.sum(result.x[self.variables:])

@singleton_args
class WeakOrderSolver:
    def __init__(self, variables: int, weak_variables: int):
        #print(f'Created WeakOrderSolver: {variables}, {weak_variables}')
        all_variables = variables + weak_variables
        self.variables = variables
        self.weak_variables = weak_variables

        # objective function == sum of absolute values
        self.c = np.concatenate(
            [np.zeros(all_variables), np.ones(all_variables)])

        # bounds
        self.bounds = [(None, None)] * all_variables \
            + [(0., None)] * all_variables

        # absolute values - inequalities
        e = np.eye(all_variables)
        A_ub_abs = np.block([[+e, -e], [-e, -e]])

        self.b_ub_abs = np.zeros(all_variables*2)

        # order inequalities
        e = np.eye(variables - 1)
        g = np.zeros((variables - 1, 1))
        A_ub_order = np.hstack([
            np.hstack([e, g]) - np.hstack([g, e]),
            np.zeros((variables-1, all_variables + weak_variables))
        ])

        A_ub_weak = np.hstack([
            np.zeros((weak_variables, variables-1)),
            np.ones((weak_variables, 1)),
            -np.eye(weak_variables),
            np.zeros((weak_variables, all_variables))
        ])

        self.A_ub = np.vstack([A_ub_abs, A_ub_order, A_ub_weak])

    def __call__(self, vector):
        assert(len(vector) == self.variables + self.weak_variables)

        b_ub_order = vector[1:self.variables] - vector[:self.variables-1]
        b_ub_weak = vector[self.variables:] - vector[self.variables-1]
        assert(len(b_ub_weak) == self.weak_variables)
        b_ub = np.concatenate([self.b_ub_abs, b_ub_order, b_ub_weak])

        result = scipy.optimize.linprog(
            self.c,
            A_ub=self.A_ub,
            b_ub=b_ub,
            bounds=self.bounds,
            method='highs-ds',
        )

        return np.sum(result.x[self.variables + self.weak_variables:])


class Loss(metaclass=_RegisterLossMeta):
    @abstractmethod
    def __call__(self, predict, target, **kwargs):
        assert isinstance(predict, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert predict.shape == target.shape
        assert len(predict.shape) == 1
        return None

class L1(Loss):
    name = 'L1'

    def __call__(self, predict, target, **kwargs):
        super().__call__(predict, target)
        return np.mean(np.abs(predict - target))


class L2(Loss):
    name = 'L2'

    def __call__(self, predict, target, **kwargs):
        super().__call__(predict, target)
        return np.mean(np.square(predict - target))


class L1Drop10P(Loss):
    name = 'L1Drop10P'

    def __call__(self, predict, target, **kwargs):
        super().__call__(predict, target)
        num_of_res = int(round(float(target.shape[0])*(9./10.)))
        residuals = np.abs(predict - target)
        residuals = np.sort(residuals, axis=0)[:num_of_res]
        return np.mean(residuals)


class L2Drop10P(Loss):
    name = 'L2Drop10P'

    def __call__(self, predict, target, **kwargs):
        super().__call__(predict, target)
        num_of_res = int(round(float(target.shape[0])*(9./10.)))
        residuals = np.square(predict - target)
        residuals = np.sort(residuals, axis=0)[:num_of_res]
        return np.mean(residuals)


class Kendall(Loss):
    ''' Kendall Loss '''
    name = 'Kendall'

    def __call__(self, predict, target, **kwargs):
        super().__call__(predict, target)
        c, _ = kendalltau(predict, target)
        return c


class RDE(Loss, metaclass=type):
    ''' Ranking Difference Error '''
    name = 'RDE'
    cache = {}

    def __init__(self, mu):
        assert(mu > 0)
        self.mu = mu

    def _compute_normalization_coefficient(self, lam, mu):
        assert mu <= lam

        prvni_sloupec = np.arange(1, -mu, step=-1)[:, np.newaxis]
        assert len(prvni_sloupec) == mu + 1

        radek = np.arange(1, mu+1)[np.newaxis, :]
        radek_obraceny = np.arange(lam, lam-mu, step=-1)[np.newaxis, :]
        assert radek.shape[1] == mu
        assert radek_obraceny.shape[1] == mu

        tabulka = prvni_sloupec + (radek - 1)
        tabulka = np.where(tabulka > 0, tabulka, radek_obraceny)
        vysledek = np.amax(np.sum(np.abs(tabulka - radek), axis=1))
        return vysledek

    def __call__(self, predict, target, **kwargs):
        super().__call__(predict, target)
        assert self.mu is not None
        lam = len(predict)
        try:
            err_max = self.cache[(lam, self.mu)]
        except KeyError:
            err_max = self._compute_normalization_coefficient(lam, self.mu)
            self.cache[(lam, self.mu)] = err_max

        si_predict = np.argsort(predict)
        si_target = np.argsort(target)[:self.mu]

        inRank = np.zeros(lam)
        inRank[si_predict] = np.arange(lam)

        r1 = inRank[si_target[:self.mu]]
        r2 = np.arange(self.mu)
        return np.sum(np.abs(r1 - r2))/err_max


class SRDE(Loss):
    name = 'SRDE'
    def __init__(self, mu):
        assert(mu > 0)
        self.mu = mu

    def __call__(self, predict, target, **kwargs):
        super().__call__(predict, target)
        mu = min(self.mu, len(predict))
        order_predicted = np.argpartition(predict, mu-1)[:self.mu]
        target = target[order_predicted]
        predict = predict[order_predicted]
        order = np.argsort(target)
        predict = predict[order]
        return self._solve_problem(predict) / min(self.mu, len(predict))

    def _scale(self, predict):
        scale = np.max(predict) - np.min(predict)
        if scale > 0:
            predict = predict / scale
        return predict

    def _solve_problem(self, predict):
        predict = self._scale(predict)
        solver = OrderSolver(len(predict))
        return solver(predict)


class ESRDE(SRDE):
    name = 'ESRDE'
    def __call__(self, predict, target, **kwargs):
        super().__call__(predict, target)
        order = np.argsort(target)
        predict = predict[order]
        predict = self._scale(predict)

        novar = len(predict)
        mu = min(novar, self.mu)

        if novar == 0:
            return 0.

        solver = WeakOrderSolver(mu, novar - mu)
        g = solver(predict)
        return g / mu


class RDE_auto(RDE, metaclass=_MakeAuto):
    pass


class RDE_full(RDE, metaclass=_MakeFull):
    pass


class SRDE_full(SRDE, metaclass=_MakeFull):
    pass


class SRDE_auto(SRDE, metaclass=_MakeAuto):
    pass


class ESRDE_full(ESRDE, metaclass=_MakeFull):
    pass


class ESRDE_auto(ESRDE, metaclass=_MakeAuto):
    pass


def get_cls_by_name(name):
    return _all_loss_classes[name]


if __name__ == '__main__':
    import unittest

    class TestLosses(unittest.TestCase):

        def test_SRDE_full_size_one(self):
            loss = SRDE_full()
            predict = np.array([1.])
            target = np.array([10.])
            self.assertEqual(loss(predict, target), 0.)

        def test_ESRDE_full_size_one(self):
            loss = ESRDE_full()
            predict = np.array([1.])
            target = np.array([10.])
            self.assertEqual(loss(predict, target), 0.)

        def test_SRDE_mu_extra(self):
            loss = SRDE(10)
            predict = np.array([1.])
            target = np.array([10.])
            self.assertEqual(loss(predict, target), 0.)

        def test_ESRDE_mu_extra(self):
            loss = ESRDE(10)
            predict = np.array([1.])
            target = np.array([10.])
            self.assertEqual(loss(predict, target), 0.)


        '''

        def test_SRDE_full_size_two(self):
            loss = SRDE_full()
            predict = np.array([1., 2.])
            target = np.array([10., 100.])
            self.assertEqual(loss(predict, target), 0./2)

            predict = np.array([2., 1.])
            target = np.array([10., 100.])
            self.assertEqual(loss(predict, target), 1./2)

            predict = np.array([3., 1.])
            target = np.array([10., 100.])
            self.assertEqual(loss(predict, target), 2./2)

        def test_SRDE_full_size_three(self):
            loss = SRDE_full()
            predict = np.array([1., 2., 2.])
            target = np.array([1., 2., 3.])
            self.assertEqual(loss(predict, target), 0./3)

            predict = np.array([1., 1., 1.])
            target = np.array([1., 2., 3.])
            self.assertEqual(loss(predict, target), 0./3)

            predict = np.array([1., 3., 2.])
            target = np.array([1., 2., 3.])
            self.assertEqual(loss(predict, target), 1./3)

            predict = np.array([30., 20., 10.])
            target = np.array([1., 2., 3.])
            self.assertEqual(loss(predict, target), 20./3)

        def test_SRDE_full_order(self):
            loss = SRDE_full()
            predict = np.array([2., 2., 1.])
            target = np.array([2., 3., 1.])
            self.assertEqual(loss(predict, target), 0./3)

            predict = np.array([1., 1., 1.])
            target = np.array([3., 1., 2.])
            self.assertEqual(loss(predict, target), 0./3)

            predict = np.array([1., 2., 3.])
            target = np.array([1., 3., 2.])
            self.assertEqual(loss(predict, target), 1./3)

            predict = np.array([20., 30., 10.])
            target = np.array([2., 1., 3.])
            self.assertEqual(loss(predict, target), 20./3)

        def test_SRDE_auto(self):
            loss = SRDE_auto()
            predict = np.array([1., 2., 3., 4., 6., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 0./3)

            loss = SRDE_full()
            predict = np.array([1., 2., 3., 4., 6., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 1./6)

            loss = SRDE_auto()
            predict = np.array([1., 2., 3., 6., 4., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 0./3)

            loss = SRDE_auto()
            predict = np.array([1., 2., 1., 6., 4., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertNotEqual(loss(predict, target), 0./3)

        def test_SRDE(self):
            loss = SRDE(2)
            predict = np.array([1., 2., 0., 4., 6., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 0./2)

            predict = np.array([2., 1., 0., 4., 6., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 1./2)

        def test_SRDE_safety_mu(self):
            loss = SRDE(100)
            predict = np.array([1., 2., 0.])
            target = np.array([1., 2., 3.])
            self.assertEqual(loss(predict, target), 2./3)

        def test_SSRDE(self):
            loss = Scaled_SRDE(2)
            predict = np.array([1., 2., 0., 4., 6., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 0./2)

            predict = np.array([2., 1., 0., 4., 6., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 1./2)

            predict = np.array([3., 1., 0., 4., 6., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 1./2)

            loss = Scaled_SRDE(4)
            predict = np.array([0., 2., 1., 4., 6., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 0.25/4)

        def test_ESRDE(self):
            loss = ESRDE(2)
            predict = np.array([2., 1., 0., 4., 6., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 2./2.)

            predict = np.array([2., 1., -4., 4., 6., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 6./2.)

        def test_Scaled_ESRDE(self):
            loss = Scaled_ESRDE(2)
            predict = np.array([2., 1., 0., 4., 6., 5.])
            target = np.array([1., 2., 3., 4., 5., 6.])
            self.assertEqual(loss(predict, target), 2./2./2.)

        def test_EERDE(self):
            loss = EESRDE(2)
            predict = np.array([20., 10., 0., 0.])
            target = np.array([1., 2., 3., 4.])
            self.assertEqual(loss(predict, target), (7.5*4)/2.)

        def test_Scaled_EERDE(self):
            loss = Scaled_EESRDE(2)
            predict = np.array([20., 10., 0., 0.])
            target = np.array([1., 2., 3., 4.])
            self.assertEqual(loss(predict, target), (7.5*4)/2./20.)

        def test_RScaled_EERDE(self):
            loss = RScaled_EESRDE(2)
            predict = np.array([20., 10., 0., 0.])
            target = np.array([1., 2., 3., 4.])
            self.assertEqual(loss(predict, target), (np.mean(predict)*4)/2./10.)

            predict = np.array([20., 10., 0., 2.])
            target = np.array([1., 2., 3., 4.])
            self.assertAlmostEqual(loss(predict, target),
                             np.sum(np.abs(predict - np.mean(predict)))/2./10.)
         '''


        def test_RDE(self):
            vec = np.array([0.8147, 0.9058, 0.1270, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649])
            tar = np.array([0.1576, 0.9706, 0.9572, 0.4854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595])

            vysledek = [0, 0.2500, 0.1905, 0.3333, 0.4286, 0.4062, 0.4444, 0.5500, 0.5111, 0.5200]

            for mu in range(1, 10+1):
                loss = RDE(mu)
                res = loss(vec, tar)
                self.assertAlmostEqual(res, vysledek[mu-1], places=4)

        def test_RDE_2(self):
            vec = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712,
                0.7060, 0.0318, 0.2769, 0.0462, 0.0971, 0.8235, 0.6948, 0.3171, 0.9502, 0.0344, 0.4387])
            tar = np.array([0.3816, 0.7655, 0.7952, 0.1869, 0.4898, 0.4456, 0.6463, 0.7094, 0.7547, 0.2760,
                  0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404, 0.5853, 0.2238, 0.7513, 0.2551, ])
            vysledek = [0.1500, 0.2105, 0.4630, 0.6176, 0.5875, 0.5222, 0.5510, 0.5524, 0.5893, 0.5750,
                0.5859, 0.5809, 0.5694, 0.6209, 0.5864, 0.5965, 0.6500, 0.6526, 0.7000, 0.6714, 0.6545]

            for mu in range(1, 21+1):
                loss = RDE(mu)
                res = loss(vec, tar)
                self.assertAlmostEqual(res, vysledek[mu-1], places=4)

        def test_l1(self):
            vec = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712,
                0.7060, 0.0318, 0.2769, 0.0462, 0.0971, 0.8235, 0.6948, 0.3171, 0.9502, 0.0344, 0.4387])
            tar = np.array([0.3816, 0.7655, 0.7952, 0.1869, 0.4898, 0.4456, 0.6463, 0.7094, 0.7547, 0.2760,
                  0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404, 0.5853, 0.2238, 0.7513, 0.2551, ])
            loss = L1()
            loss(vec, tar)

        def test_l2(self):
            vec = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712,
                0.7060, 0.0318, 0.2769, 0.0462, 0.0971, 0.8235, 0.6948, 0.3171, 0.9502, 0.0344, 0.4387])
            tar = np.array([0.3816, 0.7655, 0.7952, 0.1869, 0.4898, 0.4456, 0.6463, 0.7094, 0.7547, 0.2760,
                  0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404, 0.5853, 0.2238, 0.7513, 0.2551, ])
            loss = L2()
            a = loss(vec, tar)


        def test_kendall(self):
            vec = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712,
                0.7060, 0.0318, 0.2769, 0.0462, 0.0971, 0.8235, 0.6948, 0.3171, 0.9502, 0.0344, 0.4387])
            tar = np.array([0.3816, 0.7655, 0.7952, 0.1869, 0.4898, 0.4456, 0.6463, 0.7094, 0.7547, 0.2760,
                  0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404, 0.5853, 0.2238, 0.7513, 0.2551, ])
            loss = Kendall()
            loss(vec, tar)

    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)


# In[ ]:


