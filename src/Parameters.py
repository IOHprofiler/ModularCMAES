import numpy as np

from Utils import Boolean, AnyOf, InstanceOf
from Sampling import (
    GaussianSampling,
    OrthogonalSampling,
    MirroredSampling
)
from Population import Population


import warnings
warnings.filterwarnings("error")


class Parameters:
    active = Boolean('active')
    elitist = Boolean('elitist')
    mirrored = Boolean('mirrored')
    orthogonal = Boolean('orthogonal')
    sequential = Boolean('sequential')
    threshold_convergence = Boolean('threshold_convergence')
    bound_correction = Boolean('bound_correction')
    tpa = Boolean('tpa')

    selection = AnyOf('selection', (None, 'pairwise',))
    weights_option = AnyOf("weights_option", (None, '1/n',))
    base_sampler = AnyOf(
        "base_sampler", (None, 'quasi-sobol', 'quasi-halton',))
    local_restart = AnyOf("local_restart", (None, 'IPOP', 'BIPOP',))

    # Other parameters
    population = InstanceOf("population", Population)
    old_population = InstanceOf("old_population", Population)

    def __init__(self, d, absolute_target, rtol, **kwargs):
        self.target = absolute_target + rtol
        self.d = d
        self.init_meta_parameters()
        self.init_selection_parameters()
        self.init_modules(**kwargs)
        self.init_adaptation_parameters()
        self.init_dynamic_parameters()

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
        # TODO: we need to check these values
        self.ub, self.lb = 5, -5
        self.init_threshold = 0.2
        self.decay_factor = 0.995
        self.diameter = np.linalg.norm(self.ub - (self.lb))

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
        self.dm = np.zeros(self.d)
        self.pc = np.zeros((self.d, 1))
        self.ps = np.zeros((self.d, 1))
        self.B = np.eye(self.d)
        self.C = np.eye(self.d)
        self.D = np.ones((self.d, 1))
        self.invC = np.eye(self.d)

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
            try:
                self.invC = np.dot(self.B, self.D ** -1 * self.B.T)
            except RuntimeWarning:
                # breakpoint()
                pass

    @property
    def threshold(self):
        return self.init_threshold * self.diameter * (
            (self.budget - self.used_budget) / self.budget
        ) ** self.decay_factor
