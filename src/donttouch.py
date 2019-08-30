import numpy as np
from scipy.linalg import fractional_matrix_power
from Optimizer import Optimizer
from Utils import evaluate


class CannonicalCMA(Optimizer):
    def __init__(
            self,
            fitness_func,
            asolute_target,
            d,
            rtol,
            active=False,
            eigendecomp=True):

        self._fitness_func = fitness_func
        self.target = asolute_target + rtol
        self.d = d
        self.initialize()
        self.active = active
        self.eigendecomp = eigendecomp

    def initialize(self):
        self.used_budget = 0
        self.fopt = float("inf")
        self.xmean = np.random.rand(self.d, 1)
        self.sigma = .5
        self.budget = 1e4 * self.d

        # selection parameters
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
        self.c1 = 2 / ((self.d + 1.3)**2 + self.mueff)
        self.cmu = (
            2 * (self.mueff - 2 + 1 / self.mueff) /
            ((self.d + 2)**2 + 2 * self.mueff / 2)
        )
        # Weights normalization
        self.pweights = self.pweights / self.pweights.sum()
        amu_neg = 1 + (self.c1 / self.mu)
        amueff_neg = 1 + ((2 * self.mueff_neg) / (self.mueff + 2))
        aposdef_neg = (1 - self.c1 - self.cmu) / (self.d * self.cmu)
        self.nweights = (min(amu_neg, amueff_neg, aposdef_neg) /
                         np.abs(self.nweights).sum()) * self.nweights
        self.weights = np.append(self.pweights, self.nweights)

        # adaptation parameters
        self.cc = (
            (4 + (self.mueff / self.d)) /
            (self.d + 4 + (2 * self.mueff / self.d))
        )
        self.cs = (self.mueff + 2) / (self.d + self.mueff + 5)
        self.damps = (
            1. + (2. * max(0., np.sqrt((self.mueff - 1) / (self.d + 1)) - 1) + self.cs)
        )

        # dynamic parameters
        self.pc = np.zeros((self.d, 1))
        self.ps = np.zeros((self.d, 1))
        self.B = np.eye(self.d)
        self.D = np.eye(self.d)
        self.C = self.B * self.D * (self.B * self.D).T
        self.eigeneval = 0
        self.chiN = (
            self.d ** .5 * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))
        )
        self.invC = np.dot(self.B, self.D ** -1 * self.B.T)

    def step(self):
        # generate and evaluate offspring
        y = np.random.multivariate_normal(
            mean=np.zeros(self.d),
            cov=self.C,
            size=self.lambda_
        ).T
        x = self.xmean + (self.sigma * y)
        f = np.array([self.fitness_func(i) for i in x.T])

        # selection & recombination
        fidx = np.argsort(f)
        y = y[:, fidx]
        x = x[:, fidx]
        yw = (y[:, :self.mu] @ self.pweights).reshape(-1, 1)
        self.xmean = self.xmean + (1 * self.sigma * yw)

        # step size control
        self.ps = ((1 - self.cs) * self.ps + (np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * self.invC @ yw))

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

        # print(self.pc)
        self.pc = (1 - self.cc) * self.pc + (hs * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        )) * yw

        # punish bad direction by their weighted distance traveled
        weights = self.weights[::].copy()
        weights[weights < 0] = weights[weights < 0] * (
            self.d /
            np.power(np.linalg.norm(
                self.invC @  y[:, self.weights < 0], axis=0), 2)
        )

        old_C = (1 - (self.c1 * dhs) - self.c1 -
                 (self.cmu * self.pweights.sum())) * self.C

        rank_one = (self.c1 * self.pc * self.pc.T)

        if self.active:
            rank_mu = self.cmu * (weights * y @ y.T)
        else:
            rank_mu = (self.cmu *
                       (self.pweights * y[:, :self.mu] @ y[:, :self.mu].T))
        self.C = old_C + rank_one + rank_mu

        if np.isinf(self.C).any() or np.isnan(self.C).any() or (not 1e-16 < self.sigma < 1e6):
            self.sigma = .5
            self.pc = np.zeros((self.d, 1))
            self.ps = np.zeros((self.d, 1))
            self.C = np.eye(self.d)

        if self.eigendecomp:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D.astype(complex).reshape(-1, 1)).real
            self.invC = np.dot(self.B, self.D ** -1 * self.B.T)
        else:
            self.invC = fractional_matrix_power(self.C, -.5)

        self.fopt = min(self.fopt, f[fidx[0]])
        return not any(self.break_conditions)

    def fitness_func(self, x) -> float:
        '''Add docstring'''
        self.used_budget += 1
        return self._fitness_func(x.flatten())


if __name__ == "__main__":
    np.random.seed(12)
    print("W eigendecomp")
    for i in range(1, 25):
        evals, fopts = evaluate(
            i, 5, CannonicalCMA, eigendecomp=True)

    # np.random.seed(12)
    # print("W/o eigendecomp")
    # evals, fopts = evaluate(
    #     2, 3, CannonicalCMA, eigendecomp=False)
