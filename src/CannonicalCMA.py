import numpy as np
from scipy.linalg import fractional_matrix_power
from Optimizer import Optimizer
from Utils import evaluate


class Parameters:
    target = None
    fopt = None
    budget = None
    used_budget = None


class CannonicalCMA(Optimizer):
    def __init__(
            self,
            fitness_func,
            asolute_target,
            d,
            rtol):

        self._fitness_func = fitness_func
        self.parameters = Parameters()
        self.parameters.target = asolute_target + rtol
        self.d = d
        self.initialize()

    def initialize(self):
        self.parameters.used_budget = 0
        self.parameters.fopt = float("inf")
        self.xmean = np.random.rand(self.d, 1)
        self.sigma = .5
        self.parameters.budget = 1e4 * self.d

        # selection parameters
        self.lambda_ = (4 + np.floor(3 * np.log(self.d))).astype(int)
        self.mu = self.lambda_ // 2

        self.weights = (np.log((self.lambda_ + 1) / 2) -
                        np.log(np.arange(1, self.lambda_ + 1)))

        self.weights = self.weights[:self.mu]

        self.mueff = (
            self.weights.sum()**2 /
            (self.weights ** 2).sum()
        )
        # Weights normalization
        self.weights = self.weights / self.weights.sum()

        self.c1 = 2 / ((self.d + 1.3)**2 + self.mueff)
        self.cmu = (
            2 * (self.mueff - 2 + 1 / self.mueff) /
            ((self.d + 2)**2 + 2 * self.mueff / 2)
        )
        # adaptation parameters
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

        # dynamic parameters
        self.pc = np.zeros((self.d, 1))
        self.ps = np.zeros((self.d, 1))
        self.B = np.eye(self.d)
        self.C = np.eye(self.d)
        self.D = np.ones((self.d, 1))
        self.invC = np.eye(self.d)
        self.eigeneval = 0

    def step(self):
        # generate and evaluate offspring
        z = np.random.multivariate_normal(
            mean=np.zeros(self.d),
            cov=np.eye(self.d),
            size=self.lambda_
        ).T
        y = np.dot(self.B, self.D * z)
        x = self.xmean + (self.sigma * y)
        f = np.array([self.fitness_func(i) for i in x.T])

        # selection & recombination
        fidx = np.argsort(f)
        y = y[:, fidx]
        x = x[:, fidx]
        yw = (y[:, :self.mu] @ self.weights).reshape(-1, 1)
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
                                 (self.parameters.used_budget / self.lambda_)))
        ) < (1.4 + (2 / (self.d + 1))) * self.chiN

        dhs = (1 - hs) * self.cc * (2 - self.cc)

        # print(self.pc)
        self.pc = (1 - self.cc) * self.pc + (hs * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        )) * yw

        old_C = (1 - (self.c1 * dhs) - self.c1 -
                 (self.cmu * self.weights.sum())) * self.C

        rank_one = (self.c1 * self.pc * self.pc.T)

        rank_mu = (self.cmu *
                   (self.weights * y[:, :self.mu] @ y[:, :self.mu].T))
        self.C = old_C + rank_one + rank_mu

        if np.isinf(self.C).any() or np.isnan(self.C).any() or (not 1e-16 < self.sigma < 1e6):
            self.sigma = .5
            self.pc = np.zeros((self.d, 1))
            self.ps = np.zeros((self.d, 1))
            self.C = np.eye(self.d)
            self.B = np.eye(self.d)
            self.D = np.ones((self.d, 1))
            self.invC = np.eye(self.d)
        else:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D.astype(complex).reshape(-1, 1)).real
            self.invC = np.dot(self.B, self.D ** -1 * self.B.T)

        self.parameters.fopt = min(self.parameters.fopt, f[fidx[0]])
        return not any(self.break_conditions)


if __name__ == "__main__":
    np.random.seed(1242)
    for i in range(1, 3):
        evals, fopts = evaluate(
            i, 5, CannonicalCMA, iterations=200)
