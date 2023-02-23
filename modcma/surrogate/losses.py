import numpy as np
import math
from abc import ABCMeta, abstractmethod

from scipy.stats import kendalltau
import pulp


class Loss(metaclass=ABCMeta):
    name = 'error'

    @abstractmethod
    def __call__(self, predict, target):
        assert isinstance(predict, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert predict.shape == target.shape
        assert len(predict.shape) == 1
        return None


class L1(Loss):
    name = 'L1'

    def __call__(self, predict, target):
        super().__call__(predict, target)
        return np.mean(np.abs(predict - target))


class L2(Loss):
    name = 'L2'

    def __call__(self, predict, target):
        super().__call__(predict, target)
        return np.mean(np.square(predict - target))


class L1Drop10P(Loss):
    name = 'L1Drop10P'

    def __call__(self, predict, target):
        super().__call__(predict, target)
        num_of_res = int(round(float(target.shape[0])*(9./10.)))
        residuals = np.abs(predict - target)
        residuals = np.sort(residuals, axis=0)[:num_of_res]
        return np.mean(residuals)


class L2Drop10P(Loss):
    name = 'L2Drop10P'

    def __call__(self, predict, target):
        super().__call__(predict, target)
        num_of_res = int(round(float(target.shape[0])*(9./10.)))
        residuals = np.square(predict - target)
        residuals = np.sort(residuals, axis=0)[:num_of_res]
        return np.mean(residuals)


class Kendall(Loss):
    ''' Kendall Loss '''
    name = 'Kendall'

    def __call__(self, predict, target):
        super().__call__(predict, target)
        c, _ = kendalltau(predict, target)
        return c


class _MakeFull(ABCMeta):
    def __new__(cls, clsname, bases, attrs):
        def __init__(self):
            pass

        def __call__(self, predict, target):
            self.mu = len(target)
            return bases[0].__call__(self, predict, target)

        newclass = super().__new__(cls, clsname, bases, attrs)

        setattr(newclass, '__init__', __init__)
        setattr(newclass, '__call__', __call__)
        return newclass


class _MakeAuto(ABCMeta):
    def __new__(cls, clsname, bases, attrs):
        def __init__(self):
            pass

        def __call__(self, predict, target):
            self.mu = max(len(target) // 2, 1)
            return bases[0].__call__(self, predict, target)

        newclass = super().__new__(cls, clsname, bases, attrs)

        setattr(newclass, '__init__', __init__)
        setattr(newclass, '__call__', __call__)
        return newclass


class RDE(Loss, metaclass=type):
    ''' Ranking Difference Error Loss '''
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

    def __call__(self, predict, target):
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
    ''' Soft Ranking Difference Error '''
    name = 'SRDE'

    def __init__(self, mu):
        assert(mu > 0)
        self.mu = mu

    def __call__(self, predict, target):
        super().__call__(predict, target)
        order = np.argsort(target)

        # Choose top mu variables (based on target)
        predict = predict[order][:self.mu]
        return self._solve_problem(predict) / min(self.mu, len(predict))

    def _scale(self, predict):
        return predict

    def _solve_problem(self, predict):
        ''' returns the minimal sum of absolute elements of a vector
            that after adding to predict makes vector in ascending order
        '''
        predict = self._scale(predict)

        mu = len(predict)
        if len(predict) == 0:
            return 0.

        # Create optimization problem
        problem = pulp.LpProblem("SmoothRDE", pulp.LpMinimize)

        # Creates the variables (mu times)
        var = [pulp.LpVariable(name=str(index))
               for index in range(mu)]

        for index in range(mu - 1):
            upravena_hodnota = predict[index] + var[index]
            nasledujici_hodnota = predict[index+1] + var[index+1]

            # it adds constrain to the problem
            problem += upravena_hodnota <= nasledujici_hodnota

        # Now we want sum(abs(vars)), but it is not possible
        # => create substitute variables
        abs_var = [pulp.LpVariable(name='a'+str(index), lowBound=0.)
                   for index in range(mu)]

        # to couple the var and abs_var we need to add two constrains per variable
        for index in range(mu):
            problem += +var[index] <= abs_var[index]
            problem += -var[index] <= abs_var[index]

        # And define the loss in this way
        problem += pulp.lpSum(abs_var)

        problem.solve(pulp.PULP_CBC_CMD(msg=0))
        return float(sum(abs_var).value())

class ESRDE(SRDE):
    def __call__(self, predict, target):
        super().__call__(predict, target)
        order_true = np.argsort(target)
        order_othe = np.argsort(predict)

        selected_indexes = np.zeros(len(predict), dtype=bool)
        selected_indexes[order_true[:self.mu]] = True
        selected_indexes[order_othe[:self.mu]] = True

        predict = predict[selected_indexes]
        target = target[selected_indexes]

        order = np.argsort(target)
        predict = predict[order]

        return self._solve_problem(predict) / min(self.mu, len(predict))

class EESRDE(SRDE):
    def __call__(self, predict, target):
        super().__call__(predict, target)

        order = np.argmax(target)
        predict = predict[order]
        predict = self._scale(predict)

        novar = len(predict)
        mu = min(novar, self.mu)

        if novar == 0:
            return 0.

        # Create optimization problem
        problem = pulp.LpProblem("SmoothEESRDE", pulp.LpMinimize)

        # Creates the variables (novar times)
        var = [pulp.LpVariable(name=str(index))
               for index in range(novar)]

        # adds consecutive value constrains (the first mu)
        for index in range(mu - 1):
            value = predict[index] + var[index]
            next_value = predict[index+1] + var[index+1]

            # it adds constrain to the problem
            problem += value <= next_value 

        # add other constrains (in this case, we dont care about order between them)
        value = next_value
        for index in range(mu, novar):
            next_value = predict[index] + var[index]
            problem += value <= next_value

        # Now we want sum(abs(vars)), but it is not possible
        # => create substitute variables
        abs_var = [pulp.LpVariable(name='a'+str(index), lowBound=0.)
                   for index in range(novar)]

        # to couple the var and abs_var we need to add two constrains per variable
        for index in range(novar):
            problem += +var[index] <= abs_var[index]
            problem += -var[index] <= abs_var[index]

        # And define the loss in this way
        problem += pulp.lpSum(abs_var)

        problem.solve(pulp.PULP_CBC_CMD(msg=0))
        return float(sum(abs_var).value()) / mu


class Scaled_SRDE(SRDE):
    ''' Soft Scaled Ranking Difference Error '''
    name = 'SSRDE'

    def _scale(self, predict):
        scale = np.max(predict) - np.min(predict)
        if scale > 0:
            predict = predict / scale
        return predict

class Scaled_ESRDE(ESRDE):
    ''' Soft Scaled Ranking Difference Error '''
    name = 'SSRDE'

    def _scale(self, predict):
        scale = np.max(predict) - np.min(predict)
        if scale > 0:
            predict = predict / scale
        return predict

class RScaled_ESRDE(ESRDE):
    ''' Soft Scaled Ranking Difference Error '''
    name = 'SSRDE'

    def _scale(self, predict):
        scale = np.max(predict[:self.mu]) - np.min(predict[:self.mu])
        if scale > 0:
            predict = predict / scale
        return predict




class RDE_auto(RDE, metaclass=_MakeAuto):
    pass

class RDE_full(RDE, metaclass=_MakeFull):
    pass

class SRDE_full(SRDE, metaclass=_MakeFull):
    pass

class SRDE_auto(SRDE, metaclass=_MakeAuto):
    pass

class SSRDE_full(SRDE, metaclass=_MakeFull):
    pass

class SSRDE_auto(SRDE, metaclass=_MakeAuto):
    pass

if __name__ == '__main__':
    import unittest

    class TestLosses(unittest.TestCase):

        def test_SRDE_full_size_one(self):
            loss = SRDE_full()
            predict = np.array([1.])
            target = np.array([10.])
            self.assertEqual(loss(predict, target), 0.)

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

