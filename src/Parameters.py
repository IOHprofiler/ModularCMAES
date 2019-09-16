'''
see if we can define dependencies between modules
    we cannot select pairwise selection if mirrored selection is turned off
    This should only effect recombination.
'''
import numpy as np

from Utils import AnnotatedStruct
from Population import Population

from Sampling import (
    gaussian_sampling,
    orthogonal_sampling,
    mirrored_sampling,
    sobol_sampling,
    halton_sampling,
)


class Parameters(AnnotatedStruct):
    d: int
    absolute_target: float
    rtol: float

    # selection
    lambda_: int = None
    mu: int = None

    # TPA
    a_tpa: float = .5
    b_tpa: float = 0.
    c_sigma: float = .3

    # Sequential selection
    seq_cutoff_factor: int = 1

    # Threshold convergence TODO: we need to check these values
    ub: int = 5
    lb: int = -5
    init_threshold: float = 0.2
    decay_factor: float = 0.995
    diameter: float = float(np.linalg.norm(ub - (lb)))

    # Parameters options
    active: bool = False
    elitist: bool = False
    mirrored: bool = False
    sequential: bool = False
    threshold_convergence: bool = False
    bound_correction: bool = False
    orthogonal: bool = False
    base_sampler: str = ('gaussian', 'quasi-sobol', 'quasi-halton',)
    weights_option: str = ('default', '1/mu', '1/2^mu', )
    selection: str = ('best', 'pairwise',)
    step_size_adaptation: str = ('csa', 'tpa', 'msr', )
    # TODO:
    local_restart: str = (None, 'IPOP', 'BIPOP',)

    # Other parameters with type checking
    population: Population = None
    old_population: Population = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_meta_parameters()
        self.init_selection_parameters()
        self.init_adaptation_parameters()
        self.init_dynamic_parameters()

    def get_sampler(self):
        sampler = {
            "gaussian": gaussian_sampling,
            "quasi-sobol": sobol_sampling,
            "quasi-halton": halton_sampling,
        }.get(self.base_sampler, gaussian_sampling)(self.d)

        if self.orthogonal:
            n_samples = max(1, (
                self.lambda_ // (2 - (not self.mirrored))) - (
                    2 * self.step_size_adaptation == 'tpa')
            )
            sampler = orthogonal_sampling(sampler, n_samples)

        if self.mirrored:
            sampler = mirrored_sampling(sampler)
        return sampler

    def init_meta_parameters(self):
        self.used_budget = 0
        self.fopt = float("inf")
        self.budget = 1e4 * self.d
        self.eigeneval = 0
        self.target = self.absolute_target + self.rtol

    def init_selection_parameters(self):
        self.m = np.random.rand(self.d, 1)
        self.m_old = None
        self.lambda_ = self.lambda_ or (
            4 + np.floor(3 * np.log(self.d))).astype(int)
        self.mu = self.mu or self.lambda_ // 2
        if self.mu > self.lambda_:
            raise AttributeError(
                "\u03BC ({}) cannot be larger than \u03bb ({})".format(
                    self.mu, self.lambda_
                ))

        self.seq_cutoff = self.mu * self.seq_cutoff_factor
        self.sampler = self.get_sampler()

    def init_adaptation_parameters(self):
        if self.weights_option == '1/mu':
            self.weights = np.ones(self.mu) / self.mu
            self.weights[self.mu:] *= -1
        elif self.weights_option == '1/2^mu':
            ws = 1 / 2**np.arange(1, self.mu + 1) + (
                (1 / (2**self.mu)) / self.mu)
            self.weights = np.append(ws, ws[::-1] * -1)
        else:
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
        # For MSR
        self.ds = 2 - 2 / self.d

    def init_dynamic_parameters(self):
        self.sigma = .5
        self.dm = np.zeros(self.d)
        self.pc = np.zeros((self.d, 1))
        self.ps = np.zeros((self.d, 1))
        self.B = np.eye(self.d)
        self.C = np.eye(self.d)
        self.D = np.ones((self.d, 1))
        self.invC = np.eye(self.d)

        # Two point step size adaptation
        self.s = 0
        self.rank_tpa = None

    def adapt_sigma(self):
        if self.step_size_adaptation == 'tpa' and self.old_population:
            self.s = ((1 - self.c_sigma) * self.s) + (
                self.c_sigma * self.rank_tpa
            )
            self.sigma *= np.exp(self.s)

        elif self.step_size_adaptation == 'msr' and self.old_population:
            k_succ = (self.population.f < np.median(
                self.old_population.f)).sum()
            z = (2 / self.lambda_) * (k_succ - ((self.lambda_ + 1) / 2))

            self.s = ((1 - self.c_sigma) * self.s) + (
                self.c_sigma * z)
            self.sigma *= np.exp(self.s / self.ds)
        else:
            self.sigma = self.sigma * np.exp(
                (self.cs / self.damps) *
                ((np.linalg.norm(self.ps) / self.chiN) - 1)
            )

    def adapt(self):
        self.dm = (self.m - self.m_old) / self.sigma

        self.ps = ((1 - self.cs) * self.ps + (np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * self.invC @ self.dm))

        self.adapt_sigma()

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

        self.old_population = self.population.copy()

    @property
    def threshold(self):
        return self.init_threshold * self.diameter * (
            (self.budget - self.used_budget) / self.budget
        ) ** self.decay_factor


if __name__ == "__main__":
    p = Parameters(2, 1., 1.)
    # b.selection = 'a'?