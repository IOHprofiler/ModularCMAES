'''
TODO: Implementing modules

'''

import numpy as np
from Optimizer import Optimizer
from Utils import evaluate
from Sampling import GaussianSampling, MirroredSampling, OrthogonalSampling


class Boolean:
    def __init__(self, name=''):
        self.name = name

    def __get__(self, instance, instance_type):
        return instance.__dict__.get(self.name) or False

    def __set__(self, instance, value):
        if type(value) != bool:
            raise TypeError("{} should be bool".format(self.name))
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class NpArray:
    def __init__(self, name=''):
        self.name = name

    def __get__(self, instance, instance_type):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if type(value) != np.ndarray:
            raise TypeError("{} should be numpy.ndarray".format(self.name))
        instance.__dict__[self.name] = value.copy()

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class AnyOf:
    def __init__(self, name='', options=None):
        self.name = name
        self.options = options

    def __get__(self, instance, instance_type):
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        if value not in self.options:
            raise TypeError("{} should any of {}".format(
                self.name, self.options
            ))
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class Parameters:
    active = Boolean('active')
    elitist = Boolean('elitist')
    mirrored = Boolean('mirrored')
    orthogonal = Boolean('orthogonal')
    sequential = Boolean('sequential')
    threshold_convergence = Boolean(
        'threshold_convergence')  # this is bugged
    tpa = Boolean('tpa')
    selection = AnyOf('selection', (None, 'pairwise',))
    weights_option = AnyOf("weights_option", (None, '1/n',))
    base_sampler = AnyOf(
        "base_sampler", (None, 'quasi-sobol', 'quasi-halton',))
    local_restart = AnyOf("local_restart", (None, 'IPOP', 'BIPOP',))

    def __init__(self, d, absolute_target, rtol, **kwargs):
        self.target = absolute_target + rtol
        self.d = d
        self.init_meta_parameters()
        self.init_selection_parameters()
        self.init_modules(**kwargs)
        self.init_adaptation_parameters()
        self.init_dynamic_parameters()
        self.init_population()

    def init_modules(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.sampler = self.get_sampler()

    def get_sampler(self):
        sampler = GaussianSampling(self.d)

        if self.orthogonal:
            sampler = OrthogonalSampling(self.d, self.lambda_, sampler)

        if self.mirrored:
            sampler = MirroredSampling(sampler)
        return sampler

    def init_meta_parameters(self):
        self.used_budget = 0
        self.fopt = float("inf")
        self.budget = 1e4 * self.d
        self.eigeneval = 0

    def init_selection_parameters(self, seq_cutoff_factor=1):
        self.m = np.random.rand(self.d, 1)
        self.m_old = self.m.copy()
        self.lambda_ = (4 + np.floor(3 * np.log(self.d))).astype(int)
        self.mu = self.lambda_ // 2

        self.seq_cutoff_factor = seq_cutoff_factor
        self.seq_cutoff = self.mu * self.seq_cutoff_factor

    def init_adaptation_parameters(self):
        self.weights = (np.log((self.lambda_ + 1) / 2) -
                        np.log(np.arange(1, self.lambda_ + 1)))

        self.pweights = self.weights[:self.mu]
        self.nweights = self.weights[self.mu:]

        self.mueff = (
            self.pweights.sum()**2 /
            (self.pweights ** 2).sum()
        )
        self.mueff_neg = (
            self.nweights.sum()**2 /
            (self.nweights ** 2).sum()
        )
        self.c1 = 2 / ((self.d + 1.3)**2 + self.mueff)
        self.cmu = (
            2 * (self.mueff - 2 + 1 / self.mueff) /
            ((self.d + 2)**2 + 2 * self.mueff / 2)
        )
        self.pweights = self.pweights / self.pweights.sum()
        amu_neg = 1 + (self.c1 / self.mu)
        amueff_neg = 1 + ((2 * self.mueff_neg) / (self.mueff + 2))
        aposdef_neg = (1 - self.c1 - self.cmu) / (self.d * self.cmu)
        self.nweights = (min(amu_neg, amueff_neg, aposdef_neg) /
                         np.abs(self.nweights).sum()) * self.nweights
        self.weights = np.append(self.pweights, self.nweights)

        self.cc = (
            (4 + (self.mueff / self.d)) /
            (self.d + 4 + (2 * self.mueff / self.d))
        )
        self.cs = (self.mueff + 2) / (self.d + self.mueff + 5)
        self.damps = (
            1. + (2. * max(0., np.sqrt((self.mueff - 1) / (self.d + 1)) - 1) + self.cs)
        )
        self.chiN = (
            self.d ** .5 * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))
        )

    def init_dynamic_parameters(self):
        self.sigma = .5
        self.pc = np.zeros((self.d, 1))
        self.ps = np.zeros((self.d, 1))
        self.B = np.eye(self.d)
        self.C = np.eye(self.d)
        self.D = np.ones((self.d, 1))
        self.invC = np.eye(self.d)

    def init_population(self):
        # only placeholders
        self.old_population = None
        self.population = Population(
            np.zeros(self.d, self.lambda_),
            np.zeros(self.d, self.lambda_),
            np.zeros(self.d)
        )
        self.dm = np.zeros(self.d)

    def adapt(self):
        self.dm = (self.m - self.m_old) / self.sigma
        self.ps = ((1 - self.cs) * self.ps + (np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * self.invC @ self.dm))
        self.sigma = self.sigma * np.exp(
            (self.cs / self.damps) * ((np.linalg.norm(self.ps) / self.chiN) - 1)
        )
        # cov matrix adapation
        hs = (
            np.linalg.norm(self.ps) /
            np.sqrt(1 - np.power(1 - self.cs, 2 *
                                 (self.used_budget / self.lambda_)))
        ) < (1.4 + (2 / (self.d + 1))) * self.chiN

        dhs = (1 - hs) * self.cc * (2 - self.cc)

        self.pc = (1 - self.cc) * self.pc + (hs * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        )) * self.dm

        rank_one = (self.c1 * self.pc * self.pc.T)
        old_C = (1 - (self.c1 * dhs) - self.c1 -
                 (self.cmu * self.pweights.sum())) * self.C

        if self.active:
            # punish bad direction by their weighted distance traveled
            weights = self.weights[::].copy()
            weights = weights[:self.population.y.shape[1]]
            weights[weights < 0] = weights[weights < 0] * (
                self.d /
                np.power(np.linalg.norm(
                    self.invC @  self.population.y[:, weights < 0], axis=0), 2)
            )
            rank_mu = self.cmu * \
                (weights * self.population.y @ self.population.y.T)
        else:
            rank_mu = (self.cmu *
                       (self.pweights * self.population.y[:, :self.mu] @
                        self.population.y[:, :self.mu].T))
        self.C = old_C + rank_one + rank_mu

        if np.isinf(self.C).any() or np.isnan(self.C).any() or (not 1e-16 < self.sigma < 1e6):
            self.init_dynamic_parameters()
        else:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D.astype(complex).reshape(-1, 1)).real
            self.invC = np.dot(self.B, self.D ** -1 * self.B.T)

    @property
    def threshold(self):
        # TODO: We need to check these values
        init_threshold = 0.2
        decay_factor = 0.995
        ub, lb = 5, -5
        diameter = np.linalg.norm(ub - (lb))

        return init_threshold * diameter * (
            (self.budget - self.used_budget) / self.budget
        ) ** decay_factor


class Population:
    x = NpArray('x')
    y = NpArray('y')
    f = NpArray('f')

    def __init__(self, x, y, f):
        self.x = x
        self.y = y
        self.f = f

    def sort(self):
        fidx = np.argsort(self.f)
        self.x = self.x[:, fidx]
        self.y = self.y[:, fidx]
        self.f = self.f[fidx]

    def copy(self):
        return Population(**self.__dict__)

    def __add__(self, other):
        assert isinstance(other, self.__class__)
        return Population(
            np.hstack([self.x, other.x]),
            np.hstack([self.y, other.y]),
            np.append(self.f, other.f)
        )

    def __getitem__(self, key):
        if isinstance(key, int):
            return Population(
                self.x[:, key].reshape(-1, 1),
                self.y[:, key].reshape(-1, 1),
                np.array([self.f[key]])
            )
        elif isinstance(key, slice):
            return Population(
                self.x[:, key.start: key.stop: key.step],
                self.y[:, key.start: key.stop: key.step],
                self.f[key.start: key.stop: key.step]
            )
        else:
            raise KeyError("Key must be non-negative integer or slice, not {}"
                           .format(type(key)))

    def __repr__(self):
        return str(self.x.shape)


class ModularCMA(Optimizer):
    def __init__(
            self,
            fitness_func,
            absolute_target,
            d,
            rtol,
            **kwargs
    ):
        self.parameters = Parameters(d, absolute_target, rtol, **kwargs)
        self._fitness_func = fitness_func

    def mutate(self):
        y, x, f = [], [], []
        for i in range(self.parameters.lambda_):
            z = self.parameters.sampler.next()
            if self.parameters.threshold_convergence:
                length = np.linalg.norm(z)
                if length < self.parameters.threshold:
                    new_length = self.parameters.threshold + (
                        self.parameters.threshold - length)
                    z *= (new_length / length)
            y.append(np.dot(
                self.parameters.B, self.parameters.D * z))
            x.append(self.parameters.m +
                     (self.parameters.sigma * y[-1]))
            f.append(self.fitness_func(x[-1]))
            if self.sequential_break_conditions(i, f[-1]):
                break
        self.parameters.population = Population(
            np.hstack(x),
            np.hstack(y),
            np.array(f))

    def select(self):
        if self.parameters.elitist and self.parameters.old_population:
            self.parameters.population += self.parameters.old_population[
                :self.parameters.mu]

        self.parameters.population.sort()

        self.parameters.population = self.parameters.population[
            :self.parameters.lambda_]

        self.parameters.old_population = self.parameters.population.copy()

        self.parameters.fopt = min(
            self.parameters.fopt, self.parameters.population.f[0])

    def recombine(self):
        self.parameters.m_old = self.parameters.m.copy()
        self.parameters.m = self.parameters.m_old + (1 * (
            (self.parameters.population.x[:, :self.parameters.mu] -
                self.parameters.m_old) @
            self.parameters.pweights).reshape(-1, 1)
        )

    def step(self):
        self.mutate()
        self.select()
        self.recombine()
        self.parameters.adapt()
        return not any(self.break_conditions)

    def sequential_break_conditions(self, i: int, f) -> bool:
        '''Add docstring'''
        if self.parameters.sequential:
            return (f < self.parameters.fopt and
                    i > self.parameters.seq_cutoff)
        return False


def test_modules():
    from CannonicalCMA import CannonicalCMA
    iterations = 10
    for i in [1, 2]:  # , 5, 6, 8, 9, 10, 11, 12]:
        print("cannon")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, CannonicalCMA, iterations=iterations)
        print("new")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, ModularCMA, iterations=iterations)

        print("active")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, ModularCMA, iterations=iterations, active=True)

        print("elist")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, ModularCMA, iterations=iterations, elitist=True)

        print("mirrored")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, ModularCMA, iterations=iterations, mirrored=True)

        print("Orthogonal")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, ModularCMA, iterations=iterations, orthogonal=True)

        print("Sequential")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, ModularCMA, iterations=iterations, sequential=True)

        print()
        print()


def run_once(fid=1, **kwargs):
    evals, fopts = evaluate(
        fid, 5, ModularCMA, iterations=10, **kwargs)


if __name__ == "__main__":
    # test_modules()
    run_once(fid=1, threshold_convergence=False)
    run_once(fid=1, threshold_convergence=True)

    # functions = [20, 22, 23, 24]
    # [3, 4, 16, 17, 18, 19, 21]

    # [1, 2, 5, 6] + list(range(8, 15))  # + [20, 22, 23, 24]
    # d = 20
    # for fid in range(1, 25):
    # evals, fopts = evaluate(
    # fid, d, ModularCMA, logging=True, label="canon")
