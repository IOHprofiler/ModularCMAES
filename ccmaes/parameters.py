'''
see if we can define dependencies between modules
    we cannot select pairwise selection if mirrored selection is turned off
    This should only effect recombination.
'''
import warnings
from collections import deque
from typing import Generator
import numpy as np

from . import utils, population, sampling
from .utils import AnnotatedStruct
from .population import Population
from .sampling import (
    gaussian_sampling,
    orthogonal_sampling,
    mirrored_sampling,
    sobol_sampling,
    halton_sampling,
)


class SimpleParameters(AnnotatedStruct):
    '''Simple implementation of the parameter.Parameters object.
    This object is required in order to allow correct subclassing of the
    optimizer.Optimizer object'''
    target: float
    budget: int
    fopt: float = float("inf")
    used_budget: int = 0


class Parameters(AnnotatedStruct):
    '''AnnotatedStruct object for holding the parameters for the Configurable CMAES

    Attributes
    ----------
    d: int
        The dimensionality of the problem
    absolute_target: float
        The absolute target of the optimization problem
    rtol: float
        The distance to the absolute target which is an acceptable result
    lambda_: int = None
        The number of offspring in the population
    mu: int = None
        The number of parents in the population
    init_sigma: float = .5
        The initial value of sigma (step size)
    a_tpa: float = .5
        Parameter used in TPA
    b_tpa: float = 0.
        Parameter used in TPA
    c_sigma: float = .3
        Parameter used in TPA, for updating sigma
    seq_cutoff_factor: int = 1
        Used in sequential selection, the number of times mu individuals must be seen
        before a sequential break can be performed
    ub: int = 5
        The upper bound, used for bound correction and threshold convergence
    lb: int = -5
        The lower bound, used for bound correction and threshold convergence
    init_threshold: float = 0.2
        The initial length theshold used in treshold convergence
    decay_factor: float = 0.995
        The decay for the threshold used in threshold covergence
    active: bool = False
        Specifying whether to use active update.
            G. Jastrebski, D. V. Arnold, et al. Improving evolution strategies through
            active covariance matrix adaptation. In Evolutionary Computation (CEC),
            2006 IEEE Congress on, pages 2814–2821. IEEE, 2006
    elitist: bool = False
        Specifying whether to use an elitist approachCMAES
    mirrored: bool = False
        Specifying whether to use mirrored sampling
            D. Brockhoff, A. Auger, N. Hansen, D. V. CMAEST. Hohm.
            Mirrored Sampling and Sequential SelectioCMAESion Strategies.
            In R. Schaefer, C. Cotta, J. Kołodziej, aCMAESh, editors, Parallel
            Problem Solving from Nature, PPSN XI: 11tCMAESnal Conference,
            Kraków, Poland, September 11-15, 2010, PrCMAESart I, pages
            11–21, Berlin, Heidelberg, 2010. SpringerCMAESelberg.
    sequential: bool = False
        Specifying whether to use sequential selection
            D. Brockhoff, A. Auger, N. Hansen, D. V. Arnold, and T. Hohm.
            Mirrored Sampling and Sequential Selection for Evolution Strategies.
            In R. Schaefer, C. Cotta, J. Kołodziej, and G. Rudolph, editors, Parallel
            Problem Solving from Nature, PPSN XI: 11th International Conference,
            Kraków, Poland, September 11-15, 2010, Proceedings, Part I, pages
            11–21, Berlin, Heidelberg, 2010. Springer Berlin Heidelberg.
    threshold_convergence: bool = False
        Specifying whether to use threshold convergence
            A. Piad-Morffis, S. Estevez-Velarde, A. Bolufe-Rohler, J. Montgomery,
            and S. Chen. Evolution strategies with thresheld convergence. In
            Evolutionary Computation (CEC), 2015 IEEE Congress on, pages 2097–
            2104, May 2015.
    bound_correction: bool = False
        Specifying whether to use bound correction to enforce ub and lbs
    orthogonal: bool = False
        Specifying whether to use orthogonal sampling
            H. Wang, M. Emmerich, and T. Bäck. Mirrored Orthogonal Sampling
            with Pairwise Selection in Evolution Strategies. In Proceedings of the
            29th Annual ACM Symposium on Applied Computing, pages 154–156.
            ACM, 2014.
    base_sampler: str = ('gaussian', 'quasi-sobol', 'quasi-halton',)
        Denoting which base sampler to use, 'quasi-sobol', 'quasi-halton' can
        be selected to sample from a quasi random sequence. 
            A. Auger, M. Jebalia, and O. Teytaud. Algorithms (x, sigma, eta):
            quasi-random mutations for evolution strategies. In Artificial Evolution:
            7th International Conference, Revised Selected Papers, pages 296–307.
            Springer, 2006.
    weights_option: str = ('default', '1/mu', '1/2^mu', )
        Denoting the recombination weigths to be used.
            Sander van Rijn, Hao Wang, Matthijs van Leeuwen, and Thomas Bäck. 2016.
            Evolving the Structure of Evolution Strategies. Computer 49, 5 (May 2016), 54–63.
    selection: str = ('best', 'pairwise',)
        Specifying which option should be used for the selection of individuals
        Pairwise selection is introduced as an option to counter the bias
        produced by mirrored selection.
            A. Auger, D. Brockhoff, and N. Hansen. Mirrored sampling in
            evolution strategies with weighted recombination. In Proceedings of
            the 13th Annual Conference Companion on Genetic and Evolutionary
            Computation, GECCO ’11, pages 861–868. ACM, 2011
    step_size_adaptation: str = ('csa', 'tpa', 'msr', )
        Specifying which step size adaptation mechanism should be used. 
        csa:
            Nikolaus Hansen. The CMA evolution strategy: A tutorial.CoRR, abs/1604.00772, 2016
        tpa:
            Nikolaus Hansen. CMA-ES with two-point step-size adaptation.CoRR, abs/0805.0231,2008.
        msr: 
            Ouassim Ait Elhara, Anne Auger, and Nikolaus Hansen.  
            A Median Success Rule for Non-Elitist Evolution Strategies: Study of Feasibility. 
            In Blum et al. Christian, editor,Genetic and Evolutionary Computation Conference, 
            pages 415–422, Amsterdam, Nether-lands, July 2013. ACM, ACM Press.
    local_restart: str = (None, 'IPOP', )
        Specifying which local restart strategy should be used
            IPOP:
                Anne Auger and Nikolaus Hansen. A restart cma evolution strategy  
                with increasing population size. volume 2, pages 1769–1776, 01 2005
    population: Population = None
        The current population of individuals
    old_population: Population = None
        The old population of individuals
    termination_criteria: dict = {}
        A dictionary of termination criteria
    ipop_factor: int = 2
        The factor to increase the population after each resart (IPOP)
    tolx: float = 10e-12
        Use to compute restart condition
    tolup_sigma: float = 10e20
        Use to compute restart condition
    condition_cov: float = 10e14
        Use to compute restart condition
    ps_factor: float = 1.
        Determines the frequence of exploration/expliotation
        1 is neutral, lower is more expliotative, higher is more explorative
    sampler: generator 
        A generator object producing new samples
    used_budget: int
        The number of function evaluations used
    fopt: float
        The fitness of the current best individual
    budget: int 
        The maximum number of objective function evaluations
    target: float
        The target value up until which to optimize        
    t: int
        The number of generations
    sigma_over_time: list
        The value sigma has in each generation
    best_fopts: list
        The value of fopt in each generation
    median_fitnesses: list
        The median fitness value in each generation
    best_fitnesses: list
        The best fitness value observed in each generation
    flat_fitnesses = deque
        A deque containing boolean values denoting if a flat fitness value is observed
        in recent generations
    restarts: list
        A list containing the t values (generations) where a restart has 
        taken place
    seq_cutoff: int
        The number of individuals that must be seen before a sequential break can be performed
    diameter: float
        The diameter of the search space
    max_iter: float
        The maximum number of iterations that can occur between two restarts.        
    nbin: int
        Used to determine a window for equal function values
    n_stagnation: int
        Used to determine a window for stagnation
    flat_fitness_index: int
        Used to determine which ranked individual should be
        the same as the first indivual in order to determine 
        flat fitness values.
    sigma: float
        The step size
    m: np.ndarray
        The mean value of the individuals
    dm: np.ndarray
        The difference in the new mean value of the individuals versus the old mean value. 
    pc: np.ndarray
        The evolution path
    ps: np.ndarray
        The conjugate evolution path
    C: np.ndarray
        The covariance matrix
    B: np.ndarray
        The eigenvectors of the covariance matrix C
    D: np.ndarray
        The eigenvalues of the covariance matrix C
    invC: np.ndarray
        The result of C**-(1/2)
    s: float
        Used for TPA
    rank_tpa: float
        Used for TPA
    weights: np.ndarray
        The recombination weights.
    pweights: np.ndarray
        The positive recombination weights.
    nweights: np.ndarray
        The negative recombination weights, used in active update
    mueff: float
        The variance effective selection mass
    c1: float
        Learning rate for the rank-one update
    cc: float
        Learning rate for the rank-one update
    cmu: float
        Learning rate for the rank-mu update
    cs: float
        Learning rate for the cumulation of the step size control
    damps: float
        Used for adapting sigma with csa
    chiN: np.ndarray
        Value approaching E||N(0,I)||
    ds: float
        Used for msr
    threshold: float
        The length threshold used in threshold convergence
    last_restart: int
        The generation in where the last restart has occored
    '''

    d: int
    absolute_target: float
    rtol: float
    lambda_: int = None
    mu: int = None
    init_sigma: float = .5
    a_tpa: float = .5
    b_tpa: float = 0.
    c_sigma: float = .3
    seq_cutoff_factor: int = 1
    ub: int = 5
    lb: int = -5
    # Threshold convergence TODO: we need to check these values
    init_threshold: float = 0.2
    decay_factor: float = 0.995
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
    local_restart: str = (None, 'IPOP', )  # # TODO: 'BIPOP',)
    population: Population = None
    old_population: Population = None
    termination_criteria: dict = {}
    ipop_factor: int = 2
    tolx: float = pow(10, -12)
    tolup_sigma: float = float(pow(10, 20))
    condition_cov: float = float(pow(10, 14))
    ps_factor: float = 1.

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.init_meta_parameters()
        self.init_selection_parameters()
        self.init_adaptation_parameters()
        self.init_dynamic_parameters()
        self.init_local_restart_parameters()

    def get_sampler(self) -> Generator:
        '''Function to return a sampler generator based on the values 
        of other parameters.

        Returns
        -------
        generator
            a sampler
        '''
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

    def init_meta_parameters(self) -> None:
        '''Initialization function for parameters that hold 
        meta data about other parameters.
        '''

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
        self.restarts = []

    def init_selection_parameters(self) -> None:
        '''Initialization function for parameters that are of influence 
        in selection/population control. 
        '''

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

    def init_local_restart_parameters(self) -> None:
        '''Initialization function for parameters that are used by 
        local restart strategies, i.e. IPOP. 
        '''

        self.restarts.append(self.t)
        self.max_iter = 100 + 50 * (self.d + 3)**2 / np.sqrt(self.lambda_)
        self.nbin = 10 + int(np.ceil(30 * self.d / self.lambda_))
        self.n_stagnation = min(int(120 + (30 * self.d / self.lambda_)), 20000)
        self.flat_fitness_index = int(np.ceil(.1 + self.lambda_ / 4))

    def init_adaptation_parameters(self) -> None:
        '''Initialization function for parameters that are of influence 
        in the self-adaptive processes of the parameters. Examples are 
        recombination weights and learning rates for the covariance 
        matrix adapation.
        '''

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
        mueff_neg = (
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
        amueff_neg = 1 + ((2 * mueff_neg) / (self.mueff + 2))
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
        self.ds = 2 - (2 / self.d)

    def init_dynamic_parameters(self) -> None:
        '''Initialization function of parameters that represent the interal
        state of the CMAES algorithm, and are dynamic. Examples of such parameters
        are the Covariance matrix C and its eigenvectors and the learning rate sigma. 
        '''

        self.sigma = self.init_sigma
        self.m = np.random.rand(self.d, 1)
        self.dm = np.zeros(self.d)
        self.pc = np.zeros((self.d, 1))
        self.ps = np.zeros((self.d, 1))
        self.B = np.eye(self.d)
        self.C = np.eye(self.d)
        self.D = np.ones((self.d, 1))
        self.invC = np.eye(self.d)
        self.s = 0
        self.rank_tpa = None

    def adapt_sigma(self) -> None:
        '''Method to adapt the step size sigma. There are three variants in 
        the methodology, namely:
            ~ Two-Point Stepsize Adaptation (tpa)
            ~ Median Success Rule (msr)
            ~ Cummulative Stepsize Adapatation (csa)
        One of these methods can be selected by setting the step_size_adaptation 
        parameter. 
        '''

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
            self.sigma *= np.exp(
                (self.cs / self.damps) *
                ((np.linalg.norm(self.ps) / self.chiN) - 1)
            )

    def adapt_covariance_matrix(self) -> None:
        '''Method for adapting the covariance matrix. If the option `active` 
        is specified, active update of the covariance matrix is performed, using
        negative weights. '''
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

    def perform_eigendecomposition(self) -> None:
        '''Method to perform eigendecomposition
        If sigma or the coveriance matrix has degenerated, the dynamic parameters
        are reset.
        '''
        if np.isinf(self.C).any() or np.isnan(self.C).any() or (not 1e-16 < self.sigma < 1e6):
            self.init_dynamic_parameters()
        else:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D.astype(complex).reshape(-1, 1)).real
            self.invC = np.dot(self.B, self.D ** -1 * self.B.T)

    def adapt(self) -> None:
        '''Method for adapting the internal state paramters. 
        The conjugate evolution path ps is calculated, in addition to 
        the difference in mean x values dm. Thereafter, sigma is adapated,
        followed by the adapatation of the covariance matrix.  
        '''

        self.dm = (self.m - self.m_old) / self.sigma
        self.ps = ((1 - self.cs) * self.ps + (np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * self.invC @ self.dm) * self.ps_factor)

        self.adapt_sigma()
        self.adapt_covariance_matrix()
        # TODO: eigendecomp is neccesary to be beformed every iteration, says CMAES tut.
        self.perform_eigendecomposition()
        self.record_statistics()
        self.old_population = self.population.copy()
        if any(self.termination_criteria.values()):
            self.perform_local_restart()

    def perform_local_restart(self) -> None:
        '''Method performing local restart, given that a restart
        strategy is specified in the parameters. 
            ~ IPOP: after every restart, `lambda_` is multiplied with a factor. 
        '''

        if self.local_restart:
            if self.local_restart == 'IPOP':
                # TODO, check if mu also needs an increase
                self.mu *= self.ipop_factor
                self.lambda_ *= self.ipop_factor
            elif self.local_restart == 'BIPOP':
                raise NotImplementedError()
            self.init_selection_parameters()
            self.init_adaptation_parameters()
            self.init_dynamic_parameters()
            self.init_local_restart_parameters()
        else:
            warnings.warn("Termination criteria met: {}".format(", ".join(
                name for name, value in self.termination_criteria.items() if value
            )), RuntimeWarning)

    @property
    def threshold(self) -> None:
        '''Calculate threshold for mutation, used in threshold convergence.'''
        return self.init_threshold * self.diameter * (
            (self.budget - self.used_budget) / self.budget
        ) ** self.decay_factor

    @property
    def last_restart(self):
        '''Returns the last index of self.restarts'''
        return self.restarts[-1]

    def record_statistics(self) -> None:
        '''Method for recording metadata.
        If a local restart strategy is specified, stopping criteria
        are calculated. 
        '''

        self.flat_fitnesses.append(
            self.population.f[0] == self.population.f[
                self.flat_fitness_index
            ]
        )
        self.t += 1
        self.sigma_over_time.append(self.sigma)
        self.best_fopts.append(self.fopt)
        self.best_fitnesses.append(np.max(self.population.f))
        self.median_fitnesses.append(np.median(self.population.f))

        # The below computations add a lot of~
        # operations to the entire algorithm
        # which is why they are turned off if there
        # is no local restart strategy
        if self.local_restart:
            _t = (self.t % self.d)
            diag_C = np.diag(self.C.T)
            d_sigma = self.sigma / self.init_sigma

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
                    self.t - self.last_restart > self.n_stagnation and (
                        np.median(best_fopts[-int(.3 * self.t):]) >=
                        np.median(best_fopts[:int(.3 * self.t)]) and
                        np.median(median_fitnesses[-int(.3 * self.t):]) >=
                        np.median(median_fitnesses[:int(.3 * self.t)])
                    )
                )
            }
