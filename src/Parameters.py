'''
see if we can define dependencies between modules
    we cannot select pairwise selection if mirrored selection is turned off
    This should only effect recombination.
'''
import numpy as np
from pprint import pprint

from Utils import AnnotatedStruct
from Population import Population

from Sampling import (
    gaussian_sampling,
    orthogonal_sampling,
    mirrored_sampling,
    sobol_sampling,
    halton_sampling,
)
from collections import deque


class Parameters(AnnotatedStruct):
    d: int
    absolute_target: float
    rtol: float

    # selection
    lambda_: int = None
    mu: int = None
    init_sigma: float = .5
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

    # local_restart
    ipop_factor: int = 2
    tolx: float = pow(10, -12)
    tolup_sigma: float = pow(10, 20)
    condition_cov: float = pow(10, 14)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_meta_parameters()
        self.init_selection_parameters()
        self.init_adaptation_parameters()
        self.init_dynamic_parameters()
        self.init_local_restart_parameters()

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
        self.target = self.absolute_target + self.rtol
        self.t = 0
        self.sigma_over_time = []
        self.best_fopts = []
        self.median_fitnesses = []
        self.best_fitnesses = []
        self.flat_fitnesses = deque(maxlen=self.d)
        self.n_restarts = 0

    def init_selection_parameters(self):
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
        self.diameter = np.linalg.norm(self.ub - (self.lb))

    def init_local_restart_parameters(self):
        self.last_restart = self.t
        self.max_iter = 100 + 50 * (self.d + 3)**2 / np.sqrt(self.lambda_)
        self.nbin = 10 + int(np.ceil(30 * self.d / self.lambda_))

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
        self.cmu = min(1 - self.c1, (
            2 * ((self.mueff - 2 + (1 / self.mueff)) /
                 ((self.d + 2)**2 + (2 * self.mueff / 2)))
        ))

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
            1. + (2. * max(0., np.sqrt((self.mueff - 1) /
                                       (self.d + 1)) - 1) + self.cs)
        )
        self.chiN = (
            self.d ** .5 * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))
        )
        # For MSR
        self.ds = 2 - (2 / self.d)

    def init_dynamic_parameters(self, sigma=None, m=None):
        self.sigma = sigma or self.init_sigma
        self.m = m if type(m) != type(None) else np.random.rand(self.d, 1)

        self.dm = np.zeros(self.d)
        self.pc = np.zeros((self.d, 1))
        self.ps = np.zeros((self.d, 1))
        self.B = np.eye(self.d)
        self.C = np.eye(self.d)
        self.D = np.ones((self.d, 1))
        self.invC = np.eye(self.d)

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

    def adapt_covariance_matrix(self):
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

    def adapt(self):
        self.dm = (self.m - self.m_old) / self.sigma
        self.ps = ((1 - self.cs) * self.ps + (np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * self.invC @ self.dm))

        self.adapt_sigma()
        self.adapt_covariance_matrix()
        self.record_statistics()
        self.old_population = self.population.copy()
        if any(self.termination_criteria.values()):
            self.perform_local_restart()

    def perform_local_restart(self):
        if self.local_restart:
            if self.local_restart == 'IPOP':
                self.mu = None
                self.lambda_ *= self.ipop_factor
            elif self.local_restart == 'BIPOP':
                pass
            self.n_restarts += 1
            self.init_selection_parameters()
            self.init_adaptation_parameters()
            self.init_dynamic_parameters()
            self.init_local_restart_parameters()
        else:
            print("Warning!: Termination criterion met:", ",".join(
                name for name, value in self.termination_criteria if name
            ))

    @property
    def threshold(self):
        return self.init_threshold * self.diameter * (
            (self.budget - self.used_budget) / self.budget
        ) ** self.decay_factor

    def record_statistics(self):
        k = int(np.ceil(.1 + self.lambda_ / 4))
        self.flat_fitnesses.append(
            self.fopt == self.population.f[k]
        )
        self.t += 1
        self.sigma_over_time.append(self.sigma)
        self.best_fopts.append(self.fopt)
        self.best_fitnesses.append(np.max(self.population.f))
        self.median_fitnesses.append(np.median(self.population.f))

        d_sigma = self.sigma / self.init_sigma
        _t = (self.t % self.d)
        diag_C = np.diag(self.C.T)
        n_stagnation = min(int(120 + (30 * self.d / self.lambda_)), 20000)

        # only use values starting from last restart
        # to compute termination criteria
        best_fopts = self.best_fitnesses[self.last_restart:]
        median_fitnesses = self.median_fitnesses[self.last_restart:]
        self.termination_criteria = {
            "max_iter": (
                self.t - self.last_restart > self.max_iter
            ),
            "equalfunvalues": (
                len(best_fopts) > self.nbin and
                np.ptp(best_fopts[-self.nbin:]) == 0
            ),
            "flat_fitness": (
                self.t - self.last_restart > self.flat_fitnesses.maxlen and
                len(self.flat_fitnesses) == self.flat_fitnesses.maxlen and
                np.sum(self.flat_fitnesses) > (self.d / 3)
            ),
            "tolx": np.all((
                np.append(self.pc.T, diag_C)
                * d_sigma) < (self.tolx * self.init_sigma)
            ),
            "tolupsigma": (
                d_sigma > self.tolup_sigma * np.sqrt(self.D.max())
            ),
            "conditioncov": np.linalg.cond(self.C) > self.condition_cov,
            "noeffectaxis": np.all((1 * self.sigma * np.sqrt(
                self.D[_t, 0]) * self.B[:, _t] + self.m) == self.m
            ),
            "noeffectcoor": np.any(
                (.2 * self.sigma * np.sqrt(diag_C) + self.m) == self.m
            ),
            "stagnation": (
                self.t - self.last_restart > n_stagnation and (
                    np.median(best_fopts[-int(.3 * self.t):]) >=
                    np.median(best_fopts[:int(.3 * self.t)]) and
                    np.median(median_fitnesses[-int(.3 * self.t):]) >=
                    np.median(median_fitnesses[:int(.3 * self.t)])
                )
            )
        }


if __name__ == "__main__":
    p = Parameters(2, 1., 1.)
    # b.selection = 'a'?
