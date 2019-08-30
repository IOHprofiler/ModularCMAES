import numpy as np
from scipy.linalg import fractional_matrix_power
from Optimizer import Optimizer
from Utils import evaluate


class Parameters:
    def __init__(self, d, absolute_target, rtol):
        self.target = absolute_target + rtol
        self.d = d
        self.init_meta_parameters()
        self.init_selection_parameters()
        self.init_adaptation_parameters()
        self.normalize_weights()
        self.init_dynamic_parameters()

        self.active = False

    def init_meta_parameters(self):
        self.used_budget = 0
        self.fopt = float("inf")
        self.budget = 1e4 * self.d
        self.eigeneval = 0

    def init_selection_parameters(self):
        self.xmean = np.random.rand(self.d, 1)
        self.lambda_ = (4 + np.floor(3 * np.log(self.d))).astype(int)

        self.mu = self.lambda_ // 2
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

    def normalize_weights(self):
        # Weights normalization
        self.pweights = self.pweights / self.pweights.sum()
        amu_neg = 1 + (self.c1 / self.mu)
        amueff_neg = 1 + ((2 * self.mueff_neg) / (self.mueff + 2))
        aposdef_neg = (1 - self.c1 - self.cmu) / (self.d * self.cmu)
        self.nweights = (min(amu_neg, amueff_neg, aposdef_neg) /
                         np.abs(self.nweights).sum()) * self.nweights
        self.weights = np.append(self.pweights, self.nweights)

    def init_adaptation_parameters(self):
        self.c1 = 2 / ((self.d + 1.3)**2 + self.mueff)
        self.cmu = (
            2 * (self.mueff - 2 + 1 / self.mueff) /
            ((self.d + 2)**2 + 2 * self.mueff / 2)
        )
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
        self.z = np.zeros(self.d, self.lambda_)
        self.y = np.zeros(self.d, self.lambda_)
        self.yw = np.zeros(self.d)
        self.x = np.zeros(self.d, self.lambda_)
        self.f = np.zeros(self.d)

    def adapt(self):
        self.ps = ((1 - self.cs) * self.ps + (np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * self.invC @ self.yw))

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
        )) * self.yw

        # punish bad direction by their weighted distance traveled
        weights = self.weights[::].copy()
        weights[weights < 0] = weights[weights < 0] * (
            self.d /
            np.power(np.linalg.norm(
                self.invC @  self.y[:, self.weights < 0], axis=0), 2)
        )
        old_C = (1 - (self.c1 * dhs) - self.c1 -
                 (self.cmu * self.pweights.sum())) * self.C

        rank_one = (self.c1 * self.pc * self.pc.T)

        if self.active:
            rank_mu = self.cmu * (weights * self.y @ self.y.T)
        else:
            rank_mu = (self.cmu *
                       (self.pweights * self.y[:, :self.mu] @ self.y[:, :self.mu].T))
        self.C = old_C + rank_one + rank_mu

        if np.isinf(self.C).any() or np.isnan(self.C).any() or (not 1e-16 < self.sigma < 1e6):
            self.init_dynamic_parameters()
        else:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D.astype(complex).reshape(-1, 1)).real
            self.invC = np.dot(self.B, self.D ** -1 * self.B.T)


class ModularCMA(Optimizer):
    def __init__(
            self,
            fitness_func,
            absolute_target,
            d,
            rtol):
        self.parameters = Parameters(d, absolute_target, rtol)
        self._fitness_func = fitness_func

    def mutate(self):
        self.parameters.z = np.random.multivariate_normal(
            mean=np.zeros(self.parameters.d),
            cov=np.eye(self.parameters.d),
            size=self.parameters.lambda_
        ).T
        self.parameters.y = np.dot(
            self.parameters.B, self.parameters.D * self.parameters.z)
        self.parameters.x = self.parameters.xmean + \
            (self.parameters.sigma * self.parameters.y)
        self.parameters.f = np.array(
            [self.fitness_func(i) for i in self.parameters.x.T])

    def select(self):
        fidx = np.argsort(self.parameters.f)
        self.parameters.y = self.parameters.y[:, fidx]
        self.parameters.x = self.parameters.x[:, fidx]
        self.parameters.fopt = min(
            self.parameters.fopt, self.parameters.f[fidx[0]])

    def recombine(self):
        self.parameters.yw = (self.parameters.y[:, :self.parameters.mu]
                              @ self.parameters.pweights).reshape(-1, 1)
        self.parameters.xmean = self.parameters.xmean + \
            (1 * self.parameters.sigma * self.parameters.yw)

    def step(self):
        self.mutate()
        self.select()
        self.recombine()
        self.parameters.adapt()
        return not any(self.break_conditions)


if __name__ == "__main__":
    from CannonicalCMA import CannonicalCMA
    iterations = 10
    for i in [1, 2, 5, 6, 8, 9, 10, 11, 12]:
        print("cannon")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, CannonicalCMA, iterations=iterations)
        print("new")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, ModularCMA, iterations=iterations)
        # expect 1: ~ 650
        # expect 1: ~ 2090
