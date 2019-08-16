from __future__ import annotations
from typing import Optional, Any
import numpy as np


class BaseParameters(dict):
    alpha_mu = 2

    ## Threshold Convergence ##
    init_threshold = .2
    decay_factor = .995

    ## TPA ##
    alpha = 0.5
    tpa_factor = 0.5
    beta_tpa = 0
    c_alpha = 0.3

    ## Break conditions ##
    rtol = 1e-8

    def __getattr__(self, attr: str) -> Any:
        return self.get(attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        self[attr] = value


class Parameters(BaseParameters):
    # TODO: think of more semantic variable names
    bool_default_opts = dict.fromkeys(
        ['active', 'elitist', 'mirrored',
         'orthogonal', 'sequential', 'threshold_convergence', 'tpa'], False)

    string_default_opts = dict.fromkeys(
        ['base-sampler', 'selection', 'weights_option'])

    def __init__(
        self,
        d: int,
        mu: Optional[int] = None,
        lambda_: Optional[int] = None,
        sigma: Optional[float] = None,
        budget: Optional[int] = None,
        absolute_target: Optional[float] = None,
        seq_cutoff_factor: Optional[int] = 1,
        **kwargs
    ) -> None:
        # call these functions once #
        self.init_parameters(**kwargs)
        self.init_meta_variables(d, absolute_target, budget)
        self.init_bounds()

        # maybe call this on set_modules
        self.init_selection_variables(lambda_, mu, seq_cutoff_factor)

        # def. call this on set_modules
        self.init_adaptive_variables()

        # call this on degeneration, alg. restart
        self.init_dynamic_variables(sigma)

    def init_parameters(self, **kwargs) -> None:
        # TODO: make this configurable
        self.__dict__.update(
            {**self.bool_default_opts,
             **self.string_default_opts,
             **self,
             **kwargs}
        )

    def init_meta_variables(self, d: int, absolute_target: float, budget: int) -> None:
        self.d = d
        self.absolute_target = absolute_target
        self.target = absolute_target + self.rtol
        self.budget = budget or 1e4 * self.d
        self.used_budget = 0
        self.fitness_over_time = np.array([])
        self.fmin = float("inf")

    def init_bounds(self) -> None:
        # why this range? seems aribitrary
        # might be an idea to remove the bounds
        self.lb = -5
        self.ub = 5
        self.diameter = np.sqrt(
            np.sum(
                np.square((
                    (np.ones((self.d, 1)) * self.ub) -
                    (np.ones((self.d, 1)) * self.lb)
                ))
            )
        )
        self.wcm = (np.random.randn(self.d, 1) *
                    (self.ub - self.lb)) + self.lb

    def init_selection_variables(self, lambda_: int, mu: int,
                                 seq_cutoff_factor: int) -> None:
        self.lambda_ = lambda_ or int(4 + np.floor(3 * np.log(self.d)))
        self.mu = mu or int(1 + np.floor((self.lambda_ - 1) * .5))

        self.seq_cutoff_factor = seq_cutoff_factor
        self.seq_cutoff = self.mu * self.seq_cutoff_factor

    def init_adaptive_variables(self) -> None:
        self.recombination_weights = self.get_recombination_weights()
        self.mu_eff = 1 / np.sum(np.square(self.recombination_weights))

        # These things are different for when mu = 1
        self.cc = ((4 + self.mu_eff / self.d) /
                   (self.d + 4 + 2 * self.mu_eff / self.d))

        self.cs = (self.mu_eff + 2) / (self.mu_eff + self.d + 5)
        self.c1 = 2 / ((self.d + 1.3)**2 + self.mu_eff)

        self.c_mu = min(1 - self.c1, self.alpha_mu * (
            (self.mu_eff - 2 + 1 / self.mu_eff) / (
                (self.d + 2)**2 + self.alpha_mu * self.mu_eff / 2)))

        self.damps = 1 + 2 * \
            np.max([0, np.sqrt((self.mu_eff - 1) / (self.d + 1)) - 1]) + self.cs
        self.chi_N = self.d**.5 * (1 - 1 / (4 * self.d) + 1 / (21 * self.d**2))

    def init_dynamic_variables(self, sigma: Optional[float] = None) -> None:
        self.sigma = self.sigma_old = sigma or 1.
        self.pc = np.zeros((self.d, 1))
        self.C = np.eye(self.d)
        self.ps = np.zeros((self.d, 1))

        # Diagonalization
        self.B = np.eye(self.d)
        self.D = np.ones((self.d, 1))
        self.inv_sqrt_C = np.eye(self.d)

    def adapt(self) -> None:
        self.ps = (
            (1 - self.cs) * self.ps +
            np.sqrt(
                self.cs * (2 - self.cs) * self.mu_eff
            ) *
            np.dot(
                self.inv_sqrt_C, (self.wcm - self.wcm_old) / self.sigma
            )
        )
        # this migth be different, try to replace with the orig one.
        hsig = (
            (self.ps ** 2).sum() /
            (
                (1 - (1 - self.cs) **
                    (2 * self.used_budget / self.lambda_)
                 )
            ) / self.d) < 2 + 4 / (self.d + 1)

        self.pc = (
            (1 - self.cc) * self.pc + hsig *
            np.sqrt(
                self.cc * (2 - self.cc) * self.mu_eff
            ) * (self.wcm - self.wcm_old) / self.sigma
        )
        # there is a new setting with negative weights,
        # look in  to this, might boots performance
        self.C = (
            (1 - self.c1 - self.c_mu) * self.C + self.c1 *
            (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc *
             (2 - self.cc) * self.C) + self.c_mu *
            np.dot(self.offset[:, :self.mu], self.recombination_weights *
                   self.offset[:, :self.mu].T)
        )

        if self.active:
            self.active_update(pop)

        if self.tpa:
            self.tpa_update()
        else:
            self.sigma_update()

        self.sigma_old = self.sigma

        try:
            self.diagonalize()
        except np.linalg.LinAlgError as err:
            self.init_dynamic_variables()

    def diagonalize(self) -> None:
        self.C = np.triu(self.C) + np.triu(self.C, 1).T

        self.D, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(self.D.astype(complex).reshape(-1, 1))
        if np.isinf(self.C).any() or (not 1e-16 < self.sigma_old < 1e6):
            raise np.linalg.LinAlgError(
                'The Covariance matrix has degenerated')

        if np.isinf(self.D).any() or (~np.isreal(self.D)).any():
            raise np.linalg.LinAlgError(
                'The eigenvalues of the Covariance matrix are infinite or not real')

        self.D = np.real(self.D)
        self.inv_sqrt_C = np.dot(self.B, self.D ** -1 * self.B.T)

    def tpa_update(self) -> None:
        # tpa_result, alpha_s and beta_tpa are still undefined
        alpha = self.tpa_result * self.alpha + (
            self.beta_tpa * (self.tpa_result > 1)
        )
        self.alpha_s += self.c_alpha * (alpha - self.alpha_s)
        self.sigma *= np.exp(self.alpha_s)

    def sigma_update(self) -> None:
        self.sigma *= np.exp((
            (np.linalg.norm(self.ps) / self.chi_N - 1) *
            (self.cs / self.damps)
        ))
        if np.isinf(self.sigma):
            self.sigma = self.sigma_old

    def active_update(self, pop: "Population") -> None:
        if pop.n >= (2 * self.mu):
            offset = pop.mutation_vectors
            self.C -= self.c_mu * \
                np.dot(self.offset[:, -self.mu:],
                       self.rw * self.offset[:, -self.mu:].T)

    def get_recombination_weights(self) -> np.ndarray:
        if self.weights_option == '1/n':
            return np.ones((self.mu, 1)) * (1 / self.mu)
        else:
            _mu_prime = (self.lambda_ - 1) / 2.0
            weights = np.log(_mu_prime + 1.0) - \
                np.log(np.arange(1, self.mu + 1)[:, np.newaxis])
            return weights / np.sum(weights)

    def record_statistics(self, pop: "Population") -> None:
        # this might be better include the entire population
        self.fitness_over_time = np.append(
            self.fitness_over_time,
            pop.fitnesses
        )
        self.fmin = min(pop.best_individual.fitness, self.fmin)

    @property
    def threshold(self) -> float:
        return (
            self.init_threshold * self.diameter *
            ((self.budget - self.used_budget) / self.budget)
            ** self.decay_factor)
