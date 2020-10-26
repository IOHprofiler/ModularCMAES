import os
import pickle
import warnings
from collections import deque
from typing import Generator, TypeVar
import numpy as np
from scipy import linalg 

from inverse_covariance import QuicGraphicalLasso

from . import utils, population
from .utils import AnnotatedStruct
from .sampling import (
    gaussian_sampling,
    orthogonal_sampling,
    mirrored_sampling,
    sobol_sampling,
    halton_sampling,
)



def perform_eigendecomp(X):
    X = np.triu(X) + np.triu(X, 1).T
    D, B = linalg.eigh(X)
    if not (D > 1e-9).all():
        D[D < 1e-9] = 1e-9
        X = B.dot(np.diag(D)).dot(B.T)
        warnings.warn('Correcting machine 0 after eigendecomposition', RuntimeWarning)
        
    D = np.sqrt(D.astype(complex).reshape(-1, 1)).real
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("error")
            inv_root = B.dot(np.diag(1 / D.ravel())).dot(B.T)
    except RuntimeWarning:
        'Do not allow runtime warnings here'
        raise
    return X, D, B, inv_root


class BIPOPParameters(AnnotatedStruct):
    'Seperate object which holds BIPOP specific parameters'

    lambda_init: int
    budget: int
    mu_factor: float 
    lambda_large: int = None
    budget_small: int = None
    budget_large: int = None
    used_budget: int = 0


    @property
    def large(self) -> bool:
        'Deternotes where to use a large regime or small regime'
        return (self.budget_large >= self.budget_small) and self.budget_large > 0

    @property
    def remaining_budget(self) -> int:
        'Compute the remaining budget'
        return self.budget - self.used_budget

    @property
    def lambda_(self) -> int:
        'Returns value for lambda, based which regime is active'
        return self.lambda_large if self.large else self.lambda_small

    @property
    def sigma(self) -> float:
        'Return value for sigma, based on which regime is active' 
        return 2 if self.large else 2e-2 * np.random.random()
    
    @property
    def mu(self) -> int:
        'Return value for mu, based on which '
        return np.floor(self.lambda_ * self.mu_factor).astype(int)


    def adapt(self, used_budget: int) -> None:
        'Adapts the parameters for BIPOP on restart'
    
        used_previous_iteration = used_budget - self.used_budget
        self.used_budget += used_previous_iteration

        if self.lambda_large == None:
            self.lambda_large = self.lambda_init * 2 
            self.budget_small = self.remaining_budget // 2
            self.budget_large = self.remaining_budget - self.budget_small
        elif self.large:
            self.budget_large -= used_previous_iteration
            self.lambda_large *= 2
        else:
            self.budget_small -= used_previous_iteration
        
        self.lambda_small = np.floor(self.lambda_init * (
                .5 * self.lambda_large / self.lambda_init
                ) ** (np.random.random() ** 2)
            ).astype(int)


class Parameters(AnnotatedStruct):
    '''AnnotatedStruct object for holding the parameters for the Configurable CMAES

        Attributes
        ----------
        d: int
            The dimensionality of the problem
        target: float = -float("inf")
            The absolute target of the optimization problem
        lambda_: int = None
            The number of offspring in the population
        mu: int = None
            The number of parents in the population
        budget: int = None
            The maximum number of iterations
        init_sigma: float = .5
            The initial value of sigma (step size)
        a_tpa: float = .5
            Parameter used in TPA
        b_tpa: float = 0.
            Parameter used in TPA
        cs: float = None
            Learning rate parameter for sigma
        seq_cutoff_factor: int = 1
            Used in sequential selection, the number of times mu individuals must be seen
            before a sequential break can be performed
        ub: np.array = None 
            The upper bound, used for bound correction and threshold convergence
        lb: np.array = None
            The lower bound, used for bound correction and threshold convergence
        init_threshold: float = 0.2
            The initial length theshold used in treshold convergence
        decay_factor: float = 0.995
            The decay for the threshold used in threshold covergence
        tau: float = .99
            Threshold for applying regularization
        alpha: float = 1 
            Regularization factor    
        active: bool = False
            Specifying whether to use active update.
                G. Jastrebski, D. V. Arnold, et al. Improving evolution strategies through
                active covariance matrix adaptation. In Evolutionary Computation (CEC),
                2006 IEEE Congress on, pages 2814–2821. IEEE, 2006
        elitist: bool = False
            Specifying whether to use an elitist approachCMAES
        mirrored: str = (None, 'mirrored', mirrored pairwise', )
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
        bound_correction: str = (None, 'saturate', 'unif_resample', 'COTN', 'toroidal', 'mirror',)
            Specifying whether to use bound correction to enforce ub and lb
        orthogonal: bool = False
            Specifying whether to use orthogonal sampling
                H. Wang, M. Emmerich, and T. Bäck. Mirrored Orthogonal Sampling
                with Pairwise Selection in Evolution Strategies. In Proceedings of the
                29th Annual ACM Symposium on Applied Computing, pages 154–156.
                ACM, 2014.
        base_sampler: str = ('gaussian', 'sobol', 'halton',)
            Denoting which base sampler to use, 'sobol', 'halton' can
            be selected to sample from a quasi random sequence.
                A. Auger, M. Jebalia, and O. Teytaud. Algorithms (x, sigma, eta):
                random mutations for evolution strategies. In Artificial Evolution:
                7th International Conference, Revised Selected Papers, pages 296–307.
                Springer, 2006.
        weights_option: str = ('default', '1/mu', '1/2^mu', )
            Denoting the recombination weigths to be used.
                Sander van Rijn, Hao Wang, Matthijs van Leeuwen, and Thomas Bäck. 2016.
                Evolving the Structure of Evolution Strategies. Computer 49, 5 (May 2016), 54–63.
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
        regularization: bool = False
            Specifying where regularization should be applied
                Konstantinos Varelas, Anne Auger, and Nikolaus Hansen,
                    Sparse Inverse Covariance Learning for CMA-ES with Graphical Lasso,
                    PPSN 2020                
        population: TypeVar('Population') = None
            The current population of individuals
        old_population: TypeVar('Population') = None
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
        inv_root_C: np.ndarray
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
        c1_init: float
            Learning rate for the rank-one update
        c1_init: float
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
        max_resamples: int
            The maximum amount of resamples which can be done when 'dismiss'-boundary correction is used
        n_out_of_bounds: int
            The number of individals that are sampled out of bounds
    '''

    d: int
    target: float = -float("inf")
    budget: int = None
    lambda_: int = None
    mu: int = None
    init_sigma: float = .5
    a_tpa: float = .5
    b_tpa: float = 0.
    cs: float = None
    cc: float = None
    cmu_init: float = None
    c1_init: float = None
    seq_cutoff_factor: int = 1
    ub: np.ndarray = None
    lb: np.ndarray = None
    init_threshold: float = 0.1
    decay_factor: float = 0.995
    alpha_mu: float = 0.25
    max_resamples: int = 1000
   

    active: bool = False
    elitist: bool = False
    sequential: bool = False
    threshold_convergence: bool = False
    bound_correction: (None, 'saturate', 'unif_resample', 
                        'COTN', 'toroidal', 'mirror',) = None
    orthogonal: bool = False
    local_restart: (None, 'IPOP', 'BIPOP',) = None
    base_sampler: ('gaussian', 'sobol', 'halton',) = 'gaussian'
    mirrored: (None, 'mirrored', 'mirrored pairwise',) = None
    weights_option: ('default', '1/mu', '1/2^mu', ) = 'default'
    step_size_adaptation: ('csa', 'tpa', 'msr', ) = 'csa'

    regularization: bool = False
    tau: float = .99
    alpha: float = 1.
    
    population: TypeVar('Population') = None
    old_population: TypeVar('Population') = None
    termination_criteria: dict = {}
    ipop_factor: int = 2
    tolx: float = pow(10, -12)
    tolup_sigma: float = float(pow(10, 20))
    condition_cov: float = float(pow(10, 14))
    ps_factor: float = 1.
    compute_termination_criteria: bool = False
    __modules__ = (
        "active", 
        "elitist",
        "orthogonal",
        "sequential",
        "threshold_convergence",
        "step_size_adaptation",
        "mirrored",
        "base_sampler",
        "weights_option",
        "local_restart",
        "bound_correction"
     )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.init_selection_parameters()
        self.init_fixed_parameters()
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
            "sobol": sobol_sampling,
            "halton": halton_sampling,
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
 
    def init_fixed_parameters(self) -> None:
        '''Initialization function for parameters that 
        are not to be restarted during a optimization run.
        '''
        self.used_budget = 0
        self.n_out_of_bounds = 0
        self.budget = self.budget or int(1e4) * self.d
        self.max_lambda_ = (self.d * self.lambda_) ** 2
        self.fopt = float("inf")
        self.xopt = None
        self.t = 0
        self.sigma_over_time = []
        self.best_fopts = []
        self.median_fitnesses = []
        self.best_fitnesses = []
        self.flat_fitnesses = deque(maxlen=self.d)
        self.restarts = []
        self.bipop_parameters = BIPOPParameters(
                self.lambda_, 
                self.budget,
                self.mu / self.lambda_
        )
        
        
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

        if self.mirrored == 'mirrored pairwise':
            self.seq_cutoff_factor = max(2, self.seq_cutoff_factor)
        
        self.seq_cutoff = self.mu * self.seq_cutoff_factor
        self.sampler = self.get_sampler() 
        self.set_default('ub', np.ones((self.d, 1)) * 5)
        self.set_default('lb', np.ones((self.d, 1)) * -5)
        self.diameter = linalg.norm(self.ub - (self.lb))
        
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
            ws = np.ones(self.mu) / self.mu
            self.weights = np.append(ws, ws[::-1] * -1)
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
        self.c1_ = self.c1_init or 2 / ((self.d + 1.3)**2 + self.mueff)
        
        # this is the term from the CMA-ES tutorial
        # cmu_term = 2 * ((self.mueff - 2 + (1 / self.mueff)) /
        #         ((self.d + 2)**2 + 2 * self.mueff / 2))
        # this is the term from the PyCMA code, which works better
        cmu_term = 2.0 * ((self.alpha_mu + self.mueff + 1 / self.mueff - 2) /
                            ((self.d + 2)** 2.0 + 2.0 * self.mueff / 2))

        self.cmu_ = self.cmu_init or min(1 - self.c1_, cmu_term) 


        self.pweights = self.pweights / self.pweights.sum()
        amu_neg = 1 + (self.c1_ / self.mu)
        amueff_neg = 1 + ((2 * mueff_neg) / (self.mueff + 2))
        aposdef_neg = (1 - self.c1_ - self.cmu_) / (self.d * self.cmu_)
        self.nweights = (min(amu_neg, amueff_neg, aposdef_neg) /
                         np.abs(self.nweights).sum()) * self.nweights
        self.weights = np.append(self.pweights, self.nweights)

        self.cc = self.cc or (
            (4 + (self.mueff / self.d)) /
            (self.d + 4 + (2 * self.mueff / self.d))
        )
        
        self.cs = self.cs or (
            .3 if self.step_size_adaptation in ("msr", "tpa",) else
            (self.mueff + 2) / (self.d + self.mueff + 5)
        )
        
        self.damps = (
            1. + (2. * max(0., np.sqrt((self.mueff - 1) /
                                       (self.d + 1)) - 1) + self.cs)
        )
        self.chiN = (
            self.d ** .5 * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))
        )
        self.ds = 2 - (2 / self.d)

        # Eigenupdate is not nescessary every generation
        # these weights are taken from pycma
        self.eigeneval_factor = 0.5 * self.d * self.lambda_ * (
            self.c1 + self.cmu)**-1 / self.d**2


    def init_dynamic_parameters(self) -> None:
        '''Initialization function of parameters that represent the internal
        state of the CMAES algorithm, and are dynamic. Examples of such parameters
        are the Covariance matrix C and its eigenvectors and the learning rate sigma. 
        '''
        self.sigma = self.init_sigma
        self.m = np.random.rand(self.d, 1)
        self.dm = np.zeros(self.d)
        self.pc = np.zeros((self.d, 1))
        self.ps = np.zeros((self.d, 1))
        self.B_ = np.eye(self.d)
        self.C = np.eye(self.d)
        self.D_ = np.ones((self.d, 1))
        self.inv_root_C_ = np.eye(self.d)
        self.s = 0
        self.rank_tpa = None
        self.regularization_parameters = RegularizationParameters(self)
            # self.d, self.mueff, self.tau, self.alpha)

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
            self.s = ((1 - self.cs) * self.s) + (self.cs * self.rank_tpa)
            self.sigma *= np.exp(self.s)

        elif self.step_size_adaptation == 'msr' and self.old_population:
            k_succ = (self.population.f < np.median(
                self.old_population.f)).sum()
            z = (2 / self.lambda_) * (k_succ - ((self.lambda_ + 1) / 2))

            self.s = ((1 - self.cs) * self.s) + (self.cs * z)
            self.sigma *= np.exp(self.s / self.ds)
        else:
            self.sigma *= np.exp(
                (self.cs / self.damps) *
                ((linalg.norm(self.ps) / self.chiN) - 1)
            )

    def adapt_covariance_matrix(self) -> None:
        '''Method for adapting the covariance matrix. If the option `active` 
        is specified, active update of the covariance matrix is performed, using
        negative weights. '''
        hs = (
            linalg.norm(self.ps) /
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
                np.power(linalg.norm(
                    self.inv_root_C @  self.population.y[:, weights < 0], axis=0), 2)
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
            warnings.warn('Resetting dynamic parameters', RuntimeWarning)
            self.init_dynamic_parameters()
        else:
            self.C, self.D_, self.B_, self.inv_root_C_ = perform_eigendecomp(self.C)

    def adapt(self) -> None:
        '''Method for adapting the internal state paramters. 
        The conjugate evolution path ps is calculated, in addition to 
        the difference in mean x values dm. Thereafter, sigma is adapated,
        followed by the adapatation of the covariance matrix.  
        '''

        self.dm = (self.m - self.m_old) / self.sigma
       
        self.ps = ((1 - self.cs) * self.ps + (np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * self.inv_root_C @ self.dm) * self.ps_factor)

        self.adapt_sigma()
        self.adapt_covariance_matrix()

        if self.t % self.eigeneval_factor == 0:
            if self.regularization:
                self.regularization_parameters.regularize(self.C)
            else:
                self.perform_eigendecomposition()

        self.record_statistics()
        self.calculate_termination_criteria()
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
                self.mu *= self.ipop_factor
                self.lambda_ *= self.ipop_factor

            elif self.local_restart == 'BIPOP':
                self.bipop_parameters.adapt(self.used_budget)
                self.sigma = self.bipop_parameters.sigma
                self.lambda_ = self.bipop_parameters.lambda_
                self.mu = self.bipop_parameters.mu

            self.init_selection_parameters()
            self.init_adaptation_parameters()
            self.init_dynamic_parameters()
            self.init_local_restart_parameters()
        else:
            warnings.warn("Termination criteria met: {}".format(", ".join(
                name for name, value in self.termination_criteria.items() if value
            )), RuntimeWarning)
    
    @property
    def D(self):
        if self.regularization:
            return self.regularization_parameters.D_
        return self.D_

    @property
    def B(self):
        if self.regularization:
            return self.regularization_parameters.B_
        return self.B_
    
    @property
    def inv_root_C(self):
        if self.regularization:
            return self.regularization_parameters.inv_root_C_
        return self.inv_root_C_

    @property
    def c1(self):
        if self.regularization: 
            return self.c1_init or self.regularization_parameters.c1_
        return self.c1_

    @property
    def cmu(self):
        if self.regularization:
            return self.cmu_init or self.regularization_parameters.cmu_
        return self.cmu_

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

    @staticmethod
    def from_config_array(d:int, config_array: list) -> 'Parameters':
        '''Instantiates a Parameters object from a configuration array 

        Parameters
        ----------
        d: int
            The dimensionality of the problem
    
        config_array: list
            A list of length len(Parameters.__modules__), 
                containing ints from 0 to 2

        Returns
        -------
        A new Parameters instance
        '''
        if not len(config_array) == len(Parameters.__modules__):
            raise AttributeError(
                    "config_array must be of length " +
                    str(len(Parameters.__modules__))
                )
        parameters = dict()
        for name, cidx in zip(Parameters.__modules__, config_array):
            options = getattr(getattr(Parameters, name), 
                    "options", [False, True])
            if not len(options) > cidx:
                raise AttributeError(
                        f"id: {cidx} is invalid for {name} "
                        f"with options {', '.join(map(str, options))}"
                    )
            parameters[name] = options[cidx]
        return Parameters(d, **parameters)

    @staticmethod
    def load(filename: str) -> 'Parameters':
        '''Loads stored  parameter objects from pickle
        
        Parameters
        ----------
        filename: str
            A file path

        Returns
        -------
        A Parameters object
        
        '''
        if not os.path.isfile(filename):
            raise OSError(f"{filename} does not exist")
        
        with open(filename, "rb") as f:
            parameters = pickle.load(f)
            if not isinstance(parameters, Parameters):
                raise AttributeError(
                        f"{filename} does not contain "
                        "a Parameters object"
                    )
        parameters.sampler = parameters.get_sampler()
        return parameters

    def save(self, filename:str='parameters.pkl') -> None:
        '''Saves a parameters object to pickle
        
        Parameters
        ----------
        filename: str
            The name of the file to save to.
        '''
        with open(filename, 'wb') as f:
            self.sampler = None
            pickle.dump(self, f)

    def record_statistics(self) -> None:
        'Method for recording metadata. '
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

    def calculate_termination_criteria(self) -> None:
        '''Methods for computing restart criteria
        Only computes when a local restart
        strategy is specified, or when explicitly told to 
        to so, i.e.: self.compute_termination_criteria = True
        '''
        if self.local_restart or self.compute_termination_criteria:
            _t = (self.t % self.d)
            diag_C = np.diag(self.C.T)
            d_sigma = self.sigma / self.init_sigma
            best_fopts = self.best_fitnesses[self.last_restart:]
            median_fitnesses = self.median_fitnesses[self.last_restart:]

            self.termination_criteria = dict() if self.lambda_ > self.max_lambda_ else {
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
                    d_sigma > self.tolup_sigma * np.sqrt(self.D_.max())
                ),
                "conditioncov": np.linalg.cond(self.C) > self.condition_cov,
                "noeffectaxis": np.all((1 * self.sigma * np.sqrt(
                    self.D_[_t, 0]) * self.B_[:, _t] + self.m) == self.m
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


class RegularizationParameters:
    ''' Holds the parameters for gl-CMA-ES
    
        Attributes
        ----------
        d: int
            The dimensionality of the problem
        mueff: float
            The variance effective selection mass
        tau: float
            Threshold for applying regularization
        alpha: float
            Regularization factor        
    ''' 
    learning_rates_from_paper = False

    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        
        self.B_ = np.eye(self.d)
        self.C = np.eye(self.d)
        self.D_ = np.ones((self.d, 1))
        self.inv_root_C_ = np.eye(self.d)

        # Intialize first values
        self.regularize(self.C)

    @property
    def d(self):
        return self.parameters.d

    @property
    def tau(self):
        return self.parameters.tau

    @property
    def alpha(self):
        return self.parameters.alpha

    @property
    def mueff(self):
        return self.parameters.mueff

    @staticmethod
    def get_correlation_matrix(covariance: np.ndarray) -> np.ndarray:
        """Method to calculate a correlation matrix from a covariance matrix

        Args:
            covariance (np.ndarray): a covariance matrix

        Returns:
            np.ndarray: The correlation matrix
        """        
        v = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(v, v)
        correlation[covariance == 0] = 0
        return correlation

    def regularize(self, C: np.ndarray) -> None: 
        """Method to perform regularization on C, which is a covariance matrix
        Differences from paper:
            1. A2, Covariance matrix is correlation matrix
            2. Custom init method for QuicGraphicalLasso
            3. P_tilde is also a correlation matrix, not diagP^-1/2 P diagP^-1/2
            4. W is absolute, and scaled with alpha
            5. nz is computed differently, also the dimension is added!            
            6. Weights, c1 and cmu are compute differently, 
                I implemented the one from the paper, that looked cleaner

        Args:
            C (np.ndarray): a covariance matrix
        """        
        if self.d == 1:
            return C
            
        C_tilde = self.get_correlation_matrix(C)
        P = linalg.inv(C_tilde)
        P_tilde = self.get_correlation_matrix(P)
        
        W = (np.abs(P_tilde) < self.tau).astype(np.int) * self.alpha
        model = QuicGraphicalLasso(
            lam = W,
            Theta0 = P,                 # Initial guess for the inverse covariance matrix. 
            Sigma0 = C_tilde,           # Initial guess for the covariance matrix.
            init_method=lambda x: (x.copy(), np.max(np.abs(np.triu(x)))) # this was copied from the script from the author
        ).fit(C_tilde)

        diag_root_C = np.diag(np.sqrt(np.diag(C)))

        self.C = diag_root_C.dot(model.covariance_).dot(diag_root_C.T)
        self.C, self.D_, self.B_, self.inv_root_C_ = perform_eigendecomp(self.C)
        
        if self.tau == 0.:
            # Not really nescessary, just a sanity check
            assert np.isclose(self.C, C).all()
            pass # and it is also failing sometime due to float rounding errors


        # this is different from the paper
        self.nz = (np.abs(np.triu(model.precision_, 1)) > 0).sum() + self.d
        
        # This is different in the code from the author, this are from paper
        if self.learning_rates_from_paper:
            self.c1_ = 2 / (
                (self.nz + 1.3) * (self.d + 1.3) + self.mueff
            )
            self.cmu_ = min(1 - self.c1_, (
                2 * ( (self.mueff + 1 / self.mueff - 1.75) / (
                        (self.nz + 2) * (self.d + 2) + self.mueff
                        )
                    )
                )
            )
        else:
            # These are from pycma/scripts 
            rank_one = 2. / (
                2 * (self.nz / self.d + 0.15) * (self.d + 1.3) + self.mueff
            ) / self.parameters.c1_

            self.c1_ = rank_one * (2 / ((self.d + 1.3)**2 + self.mueff))

            t1 = 2. * (self.parameters.alpha_mu + self.mueff + 1. / self.mueff - 2.)
            t2 = 2. * (self.nz / self.d + 0.5) * (self.d + 2.) + self.mueff

            rank_mu = t1 / t2 / self.parameters.cmu_
            cmu_term = t1 / ((self.d + 2)** 2.0 + 2.0 * self.mueff / 2)

            self.cmu_ = min(1 - self.c1_, rank_mu * cmu_term)
