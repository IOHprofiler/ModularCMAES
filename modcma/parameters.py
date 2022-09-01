"""Definition of Parameters objects, which are used by ModularCMA-ES."""
import os
import pickle
import warnings
from collections import deque
from typing import Generator, TypeVar

import numpy as np
from scipy import linalg

from .utils import AnnotatedStruct
from .sampling import (
    gaussian_sampling,
    orthogonal_sampling,
    mirrored_sampling,
    sobol_sampling,
    halton_sampling,
    Sobol, 
    Halton
)


class Parameters(AnnotatedStruct):
    """AnnotatedStruct object for holding the parameters for the ModularCMAES.

    Attributes
    ----------
    d: int
        The dimensionality of the problem
    x0: np.ndarray
        Initial guess of the population center of mass.
    target: float = -float("inf")
        The absolute target of the optimization problem
    budget: int = None
        The maximum number of iterations
    n_generations: int = None
        The number of generations to run the optimizer. If this value is specified
        this will override the default break-conditions, and the optimizer will only
        stop after n_generations. Target-reached and budget will be ignored.
    lambda_: int = None
        The number of offspring in the population
    mu: int = None
        The number of parents in the population
    sigma0: float = .5
        The initial value of sigma (step size)
    a_tpa: float = .5
        Parameter used in TPA
    b_tpa: float = 0.
        Parameter used in TPA
    cs: float = None
        Learning rate for the cumulation of the step size control
    cc: float = None
        Learning rate for the rank-one update
    cmu: float = None
        Learning rate for the rank-mu update
    c1: float = None
        Learning rate for the rank-one update
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
    max_resamples: int
        The maximum amount of resamples which can be done when
        'dismiss'-boundary correction is used
    active: bool = False
        Specifying whether to use active update.
            [1] G. Jastrebski, D. V. Arnold, et al. Improving evolution strategies through
            active covariance matrix adaptation. In Evolutionary Computation (CEC),
            2006 IEEE Congress on, pages 2814–2821. IEEE, 2006
    elitist: bool = False
        Specifying whether to use an elitist approach
    sequential: bool = False
        Specifying whether to use sequential selection
            [3] D. Brockhoff, A. Auger, N. Hansen, D. V. Arnold, and T. Hohm.
            Mirrored Sampling and Sequential Selection for Evolution Strategies.
            In R. Schaefer, C. Cotta, J. Kołodziej, and G. Rudolph, editors, Parallel
            Problem Solving from Nature, PPSN XI: 11th International Conference,
            Kraków, Poland, September 11-15, 2010, Proceedings, Part I, pages
            11–21, Berlin, Heidelberg, 2010. Springer Berlin Heidelberg.
    threshold_convergence: bool = False
        Specifying whether to use threshold convergence
            [4] A. Piad-Morffis, S. Estevez-Velarde, A. Bolufe-Rohler, J. Montgomery,
            and S. Chen. Evolution strategies with thresheld convergence. In
            Evolutionary Computation (CEC), 2015 IEEE Congress on, pages 2097–
            2104, May 2015.
    bound_correction: str = (None, 'saturate', 'unif_resample', 'COTN', 'toroidal', 'mirror',)
        Specifying whether to use bound correction to enforce ub and lb
    orthogonal: bool = False
        Specifying whether to use orthogonal sampling
            [5] H. Wang, M. Emmerich, and T. Bäck. Mirrored Orthogonal Sampling
            with Pairwise Selection in Evolution Strategies. In Proceedings of the
            29th Annual ACM Symposium on Applied Computing, pages 154–156.
    local_restart: str = (None, 'IPOP', )
        Specifying which local restart strategy should be used
            IPOP:
                [11] Anne Auger and Nikolaus Hansen. A restart cma evolution strategy
                with increasing population size. volume 2, pages 1769–1776, 01 2005
    base_sampler: str = ('gaussian', 'sobol', 'halton',)
        Denoting which base sampler to use, 'sobol', 'halton' can
        be selected to sample from a quasi random sequence.
            [6] A. Auger, M. Jebalia, and O. Teytaud. Algorithms (x, sigma, eta):
            random mutations for evolution strategies. In Artificial Evolution:
            7th International Conference, Revised Selected Papers, pages 296–307.
            Springer, 2006.
    mirrored: str = (None, 'mirrored', mirrored pairwise', )
        Specifying whether to use mirrored sampling
            [2] D. Brockhoff, A. Auger, N. Hansen, D. V. CMAEST. Hohm.
            Mirrored Sampling and Sequential SelectioCMAESion Strategies.
            In R. Schaefer, C. Cotta, J. Kołodziej, aCMAESh, editors, Parallel
            Problem Solving from Nature, PPSN XI: 11tCMAESnal Conference,
            Kraków, Poland, September 11-15, 2010, PrCMAESart I, pages
            11–21, Berlin, Heidelberg, 2010. SpringerCMAESelberg.
            ACM, 2014.
    weights_option: str = ("default", "equal", "1/2^lambda", )
        Denoting the recombination weigths to be used.
            [7] Sander van Rijn, Hao Wang, Matthijs van Leeuwen, and Thomas Bäck. 2016.
            Evolving the Structure of Evolution Strategies. Computer 49, 5 (May 2016), 54–63.
    step_size_adaptation: str = ('csa', 'tpa', 'msr', )
        Specifying which step size adaptation mechanism should be used.
        csa:
            [8] Nikolaus Hansen. The CMA evolution strategy: A tutorial.CoRR,
            abs/1604.00772, 2016
        tpa:
            [9] Nikolaus Hansen. CMA-ES with two-point step-size adaptation.CoRR, 
            abs/0805.0231,2008.
        msr:
            [10] Ouassim Ait Elhara, Anne Auger, and Nikolaus Hansen.
            A Median Success Rule for Non-Elitist Evolution Strategies: Study of Feasibility.
            In Blum et al. Christian, editor,Genetic and Evolutionary Computation Conference,
            pages 415–422, Amsterdam, Nether-lands, July 2013. ACM, ACM Press.
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
    sample_sigma: bool = Flase
        Whether to sample sigma for each individual from a lognormal
        distribution.
    sampler: generator
        A generator object producing new samples
    used_budget: int
        The number of function evaluations used
    fopt: float
        The fitness of the current best individual
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
        The maximum amount of resamples which can be done when 
        'dismiss'-boundary correction is used
    n_out_of_bounds: int
        The number of individals that are sampled out of bounds

    """

    d: int
    x0: np.ndarray = None
    target: float = -float("inf")
    budget: int = None
    n_generations: int = None
    lambda_: int = None
    mu: int = None
    sigma0: float = 0.2
    a_tpa: float = 0.5
    b_tpa: float = 0.0
    cs: float = None
    cc: float = None
    cmu: float = None
    c1: float = None
    seq_cutoff_factor: int = 1
    ub: np.ndarray = None
    lb: np.ndarray = None
    init_threshold: float = 0.1
    decay_factor: float = 0.995
    max_resamples: int = 1000
    active: bool = False
    elitist: bool = False
    sequential: bool = False
    threshold_convergence: bool = False
    bound_correction: (
        None, "saturate", "unif_resample", "COTN", "toroidal", "mirror") = None
    orthogonal: bool = False
    local_restart: (None, "IPOP", "BIPOP") = None
    base_sampler: ("gaussian", "sobol", "halton") = "gaussian"
    mirrored: (None, "mirrored", "mirrored pairwise") = None
    weights_option: ("default", "equal", "1/2^lambda") = "default"
    step_size_adaptation: (
        "csa", "tpa", "msr", "xnes", "m-xnes", "lp-xnes", "psr") = "csa"
    population: TypeVar("Population") = None
    old_population: TypeVar("Population") = None
    termination_criteria: dict = {}
    ipop_factor: int = 2
    tolx: float = pow(10, -12)
    tolup_sigma: float = float(pow(10, 20))
    condition_cov: float = float(pow(10, 14))
    ps_factor: float = 1.0
    compute_termination_criteria: bool = False
    sample_sigma: bool = False  # TODO make this a module
    vectorized_fitness: bool = False
    sobol: TypeVar("Sobol") = None
    halton: TypeVar("Halton") = None

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
        "bound_correction",
    )

    def __init__(self, *args, **kwargs) -> None:
        """Intialize parameters. Calls sub constructors for different parameter types."""
        super().__init__(*args, **kwargs)
        self.init_selection_parameters()
        self.init_fixed_parameters()
        self.init_adaptation_parameters()
        self.init_dynamic_parameters()
        self.init_local_restart_parameters()

    def get_sampler(self) -> Generator:
        """Function to return a sampler generator based on the values of other parameters.

        Returns
        -------
        generator
            a sampler

        """
        if self.base_sampler == 'gaussian':
            sampler = gaussian_sampling(self.d)
        elif self.base_sampler == 'sobol':
            self.sobol = self.sobol or Sobol(self.d)
            sampler = sobol_sampling(self.sobol)      
        elif self.base_sampler == 'halton':
            self.halton = self.halton or Halton(self.d)
            sampler = halton_sampling(self.halton)

        if self.orthogonal:
            n_samples = max(
                1,
                (self.lambda_ // (2 - (not self.mirrored)))
                - (2 * self.step_size_adaptation == "tpa"),
            )
            sampler = orthogonal_sampling(sampler, n_samples)

        if self.mirrored:
            sampler = mirrored_sampling(sampler)

        return sampler

    def init_fixed_parameters(self) -> None:
        """Initialization function for parameters that are not restarted during a run."""
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
        self.restarts = [0]
        self.bipop_parameters = BIPOPParameters(
            self.lambda_, self.budget, self.mu / self.lambda_
        )
        self.chiN = self.d ** 0.5 * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))
        self.ds = 2 - (2 / self.d)
        self.beta = np.log(2) / max((np.sqrt(self.d) * np.log(self.d)), 1)
        self.succes_ratio = .25

    def init_selection_parameters(self) -> None:
        """Initialization function for parameters that influence in selection."""
        self.lambda_ = self.lambda_ or (4 + np.floor(3 * np.log(self.d))).astype(int)

        if self.mirrored == "mirrored pairwise":
            self.seq_cutoff_factor = max(2, self.seq_cutoff_factor)
            if self.lambda_ % 2 != 0:
                self.lambda_ += 1

        self.mu = self.mu or self.lambda_ // 2
        if self.mu > self.lambda_:
            warnings.warn(
                "\u03BC ({}) cannot be larger than \u03bb ({}). Modifying \u03bb to ({})".format(
                    self.mu, self.lambda_, self.lambda_ // 2
                ),
                RuntimeWarning,
            )
            self.mu = self.lambda_ // 2

        self.seq_cutoff = self.mu * self.seq_cutoff_factor
        self.sampler = self.get_sampler()
        self.set_default("ub", np.ones((self.d, 1)) * 5)
        self.set_default("lb", np.ones((self.d, 1)) * -5)
        self.diameter = np.linalg.norm(self.ub - (self.lb))

    def init_local_restart_parameters(self) -> None:
        """Initialization function for parameters for local restart strategies, i.e. IPOP.

        TODO: check if we can move this to separate object.
        """
       
        self.max_iter = 100 + 50 * (self.d + 3) ** 2 / np.sqrt(self.lambda_)
        self.nbin = 10 + int(np.ceil(30 * self.d / self.lambda_))
        self.n_stagnation = min(int(120 + (30 * self.d / self.lambda_)), 20000)
        self.flat_fitness_index = int(
            np.round(0.1 + self.lambda_ / 4)
        )

    def init_adaptation_parameters(self) -> None:
        """Initialization function for parameters for self-adaptive processes.

        Examples are recombination weights and learning rates for the covariance
        matrix adapation.
        """
        if self.weights_option == "equal":
            ws = 1 / self.mu
            self.weights = np.append(
                np.ones(self.mu) * ws, np.ones(self.lambda_ - self.mu) * ws * -1
            )
        elif self.weights_option == "1/2^lambda":
            base = np.float64(2)
            positive = self.mu / (base ** np.arange(1, self.mu + 1)) + ( # "self.mu /" should be "1 /"
                (1 / (base ** self.mu)) / self.mu
            )
            n = self.lambda_ - self.mu
            negative = (1 / (base ** np.arange(1, n + 1)) + (
                (1 / (base ** n)) / n
            ))[::-1] * -1
            self.weights = np.append(positive, negative)
        else:
            self.weights = np.log((self.lambda_ + 1) / 2) - np.log(
                np.arange(1, self.lambda_ + 1)
            )

        self.pweights = self.weights[: self.mu]
        self.nweights = self.weights[self.mu:]
        self.mueff = self.pweights.sum() ** 2 / (self.pweights ** 2).sum()
        mueff_neg = self.nweights.sum() ** 2 / (self.nweights ** 2).sum()

        self.pweights = self.pweights / self.pweights.sum()
        self.c1 = self.c1 or 2 / ((self.d + 1.3) ** 2 + self.mueff)
        self.cmu = self.cmu or min(1 - self.c1, (2 * (
            (self.mueff - 2 + (1 / self.mueff))
            / ((self.d + 2) ** 2 + (2 * self.mueff / 2))
        )))

        amu_neg = 1 + (self.c1 / self.mu)
        amueff_neg = 1 + ((2 * mueff_neg) / (self.mueff + 2))
        aposdef_neg = (1 - self.c1 - self.cmu) / (self.d * self.cmu)
        self.nweights = (
            min(amu_neg, amueff_neg, aposdef_neg) / np.abs(self.nweights).sum()
        ) * self.nweights
        self.weights = np.append(self.pweights, self.nweights)

        self.cc = self.cc or (
            (4 + (self.mueff / self.d)) / (self.d + 4 + (2 * self.mueff / self.d))
        )

        self.cs = self.cs or {
            "csa": (self.mueff + 2) / (self.d + self.mueff + 5),
            "msr": .3,
            "tpa": .3,
            "xnes": self.mueff / (2 * np.log(max(2, self.d)) * np.sqrt(self.d)),
            "m-xnes": 1.,
            "lp-xnes": 9 * self.mueff / (10 * np.sqrt(self.d)),
            "psr": .4
        }[self.step_size_adaptation]

        self.damps = 1.0 + (
            2.0 * max(0.0, np.sqrt((self.mueff - 1) / (self.d + 1)) - 1) + self.cs
        )

    def init_dynamic_parameters(self) -> None:
        """Initialization function of parameters that represent the dynamic state of the CMA-ES.

        Examples of such parameters are the Covariance matrix C and its 
        eigenvectors and the learning rate sigma.
        """
        self.sigma = np.float64(self.sigma0) * (self.ub[0,0] - self.lb[0,0])
        if hasattr(self, "m") or self.x0 is None: 
            self.m = np.float64(np.random.uniform(self.lb, self.ub, (self.d, 1)))
        else:
            self.m = np.float64(self.x0.copy())
        self.m_old = np.empty((self.d, 1), dtype=np.float64)
        self.dm = np.zeros(self.d, dtype=np.float64)
        self.pc = np.zeros((self.d, 1), dtype=np.float64)
        self.ps = np.zeros((self.d, 1), dtype=np.float64)
        self.B = np.eye(self.d, dtype=np.float64)
        self.C = np.eye(self.d, dtype=np.float64)
        self.D = np.ones((self.d, 1), dtype=np.float64)
        self.inv_root_C = np.eye(self.d, dtype=np.float64)
        self.s = 0
        self.rank_tpa = None
        self.hs = True

    def adapt(self) -> None:
        """Method for adapting the internal state parameters.

        The conjugate evolution path ps is calculated, in addition to
        the difference in mean x values dm. Thereafter, sigma is adapated,
        followed by the adapatation of the covariance matrix.
        TODO: eigendecomp is not neccesary to be beformed every iteration, says CMAES tut.
        """
        self.adapt_evolution_paths()
        self.adapt_sigma()
        self.adapt_covariance_matrix()
        self.perform_eigendecomposition()
        self.record_statistics()
        self.calculate_termination_criteria()
        self.old_population = self.population.copy()
        if any(self.termination_criteria.values()):
            self.perform_local_restart()

    def adapt_sigma(self) -> None:
        """Method to adapt the step size sigma.

        There are three variants in implemented here, namely:
            ~ Cummulative Stepsize Adaptation (csa)
            ~ Two-Point Stepsize Adaptation (tpa)
            ~ Median Success Rule (msr)
            ~ xNES step size adaptation (xnes)
            ~ median-xNES step size adaptation (m-xnes)
            ~ xNES with Log-normal Prior step size adaptation (lp-xnes)
            ~ Population Step Size rule from lm-cmaes (psr)

        One of these methods can be selected by setting the step_size_adaptation
        parameter.
        """
        if self.step_size_adaptation == "csa":
            self.sigma *= np.exp(
                (self.cs / self.damps) * ((np.linalg.norm(self.ps) / self.chiN) - 1)
            )

        elif self.step_size_adaptation == "tpa" and self.old_population:
            self.s = ((1 - self.cs) * self.s) + (self.cs * self.rank_tpa)
            self.sigma *= np.exp(self.s)

        elif self.step_size_adaptation == "msr" and self.old_population:
            k_succ = (self.population.f < np.median(self.old_population.f)).sum()
            z = (2 / self.lambda_) * (k_succ - ((self.lambda_ + 1) / 2))
            self.s = ((1 - self.cs) * self.s) + (self.cs * z)
            self.sigma *= np.exp(self.s / self.ds)

        elif self.step_size_adaptation == "xnes":
            w = self.weights.clip(0)[:self.population.n]
            z = np.power(
                np.linalg.norm(self.inv_root_C.dot(self.population.y), axis=0), 2
            ) - self.d
            self.sigma *= np.exp((self.cs / np.sqrt(self.d)) * (w * z).sum())

        elif self.step_size_adaptation == "m-xnes" and self.old_population:
            z = (self.mueff * np.power(np.linalg.norm(self.inv_root_C.dot(self.dm)), 2)) - self.d
            self.sigma *= np.exp((self.cs / self.d) * z)

        elif self.step_size_adaptation == "lp-xnes":
            w = self.weights.clip(0)[:self.population.n]
            z = np.exp(self.cs * (w @ np.log(self.population.s)))
            self.sigma = np.power(self.sigma, 1 - self.cs) * z

        elif self.step_size_adaptation == "psr" and self.old_population:
            n = min(self.population.n, self.old_population.n)
            combined = (self.population[:n] + self.old_population[:n]).sort()
            r = np.searchsorted(combined.f, self.population.f[:n])
            r_old = np.searchsorted(combined.f, self.old_population.f[:n])

            zpsr = (r_old - r).sum() / pow(n, 2) - self.succes_ratio
            self.s = (1 - self.cs) * self.s + (self.cs * zpsr)
            self.sigma *= np.exp(self.s / self.ds)

    def adapt_covariance_matrix(self) -> None:
        """Method for adapting the covariance matrix.

        If the option `active` is specified, active update of the covariance
        matrix is performed, using negative weights.
        """
        rank_one = self.c1 * self.pc * self.pc.T

        dhs = (1 - self.hs) * self.cc * (2 - self.cc)
        old_C = (
            1 - (self.c1 * dhs) - self.c1 - (self.cmu * self.pweights.sum())
        ) * self.C

        if self.active:
            weights = self.weights[::].copy()
            weights = weights[: self.population.y.shape[1]]
            weights[weights < 0] = weights[weights < 0] * (
                self.d / np.power(
                    np.linalg.norm(
                        self.inv_root_C @ self.population.y[:, weights < 0], axis=0
                    ), 2
                )
            )

            rank_mu = self.cmu * (weights * self.population.y @ self.population.y.T)
        else:
            rank_mu = self.cmu * (
                self.pweights
                * self.population.y[:, : self.mu]
                @ self.population.y[:, : self.mu].T
            )
        self.C = old_C + rank_one + rank_mu

    def perform_eigendecomposition(self) -> None:
        """Method to perform eigendecomposition.

        If sigma or the coveriance matrix has degenerated, the dynamic parameters
        are reset.
        """
        if (
            np.isinf(self.C).any()
            or np.isnan(self.C).any()
            or (not 1e-16 < self.sigma < 1e6)
        ):
            self.init_dynamic_parameters()
        else:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T

            self.D, self.B = linalg.eigh(self.C)
            if np.all(self.D > 0):
                self.D = np.sqrt(self.D.reshape(-1, 1))
                self.inv_root_C = np.dot(self.B, self.D ** -1 * self.B.T)
            else:
                self.init_dynamic_parameters()

    def adapt_evolution_paths(self) -> None:
        """Method to adapt the evolution paths ps and pc."""
        self.dm = (self.m - self.m_old) / self.sigma
        self.ps = (1 - self.cs) * self.ps + (
            np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.inv_root_C @ self.dm
        ) * self.ps_factor

        self.hs = (
            np.linalg.norm(self.ps)
            / np.sqrt(1 - np.power(1 - self.cs, 2 * (self.used_budget / self.lambda_)))
        ) < (1.4 + (2 / (self.d + 1))) * self.chiN

        self.pc = (1 - self.cc) * self.pc + (
            self.hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff)
        ) * self.dm

    def perform_local_restart(self) -> None:
        """Method performing local restart, if a restart strategy is specified."""
        if self.local_restart:

            if len(self.restarts) == 0:
                self.restarts.append(self.t)

            if self.local_restart == "IPOP" and self.mu < 512: 
                self.mu *= self.ipop_factor
                self.lambda_ *= self.ipop_factor

            elif self.local_restart == "BIPOP":
                self.bipop_parameters.adapt(self.used_budget)
                self.sigma = self.bipop_parameters.sigma
                self.lambda_ = self.bipop_parameters.lambda_
                self.mu = self.bipop_parameters.mu

            self.init_selection_parameters()
            self.init_adaptation_parameters()
            self.init_dynamic_parameters()
            self.init_local_restart_parameters()
            self.restarts.append(self.t)
        else:
            warnings.warn(
                "Termination criteria met: {}".format(
                    ", ".join(
                        name
                        for name, value in self.termination_criteria.items()
                        if value
                    )
                ),
                RuntimeWarning,
            )

    @property
    def threshold(self) -> None:
        """Calculate threshold for mutation, used in threshold convergence."""
        return (
            self.init_threshold
            * self.diameter
            * ((self.budget - self.used_budget) / self.budget) ** self.decay_factor
        )

    @property
    def last_restart(self):
        """Return the last index of self.restarts."""
        return self.restarts[-1]

    @staticmethod
    def from_config_array(d: int, config_array: list) -> "Parameters":
        """Instantiate a Parameters object from a configuration array.

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

        """
        if not len(config_array) == len(Parameters.__modules__):
            raise AttributeError(
                "config_array must be of length " + str(len(Parameters.__modules__))
            )
        parameters = dict()
        for name, cidx in zip(Parameters.__modules__, config_array):
            options = getattr(getattr(Parameters, name), "options", [False, True])
            if not len(options) > cidx:
                raise AttributeError(
                    f"id: {cidx} is invalid for {name} "
                    f"with options {', '.join(map(str, options))}"
                )
            parameters[name] = options[cidx]
        return Parameters(d, **parameters)

    @staticmethod
    def load(filename: str) -> "Parameters":
        """Load stored  parameter objects from pickle.

        Parameters
        ----------
        filename: str
            A file path

        Returns
        -------
        A Parameters object

        """
        if not os.path.isfile(filename):
            raise OSError(f"{filename} does not exist")

        with open(filename, "rb") as f:
            parameters = pickle.load(f)
            if not isinstance(parameters, Parameters):
                raise AttributeError(
                    f"{filename} does not contain " "a Parameters object"
                )
        np.random.set_state(parameters.random_state)
        parameters.sampler = parameters.get_sampler()
        return parameters

    def save(self, filename: str = "parameters.pkl") -> None:
        """Save a parameters object to pickle.

        Parameters
        ----------
        filename: str
            The name of the file to save to.

        """
        sampler = self.sampler 
        with open(filename, "wb") as f:
            self.sampler = None
            self.random_state = np.random.get_state()
            pickle.dump(self, f)
        self.sampler = sampler


    def record_statistics(self) -> None:
        """Method for recording metadata."""
        # if self.local_restart or self.compute_termination_criteria:
        self.flat_fitnesses.append(
            self.population.f[0] == self.population.f[self.flat_fitness_index]
        )
        self.t += 1
        self.sigma_over_time.append(self.sigma)
        self.best_fopts.append(self.fopt)
        self.best_fitnesses.append(np.max(self.population.f))
        self.median_fitnesses.append(np.median(self.population.f))

    def calculate_termination_criteria(self) -> None:
        """Method for computing restart criteria.

        Only computes when a local restart strategy is specified, or when explicitly
        told to do so, i.e.: self.compute_termination_criteria = True
        """
        if self.local_restart or self.compute_termination_criteria:
            _t = self.t % self.d
            diag_C = np.diag(self.C.T)
            d_sigma = self.sigma / self.sigma0
            best_fopts = self.best_fitnesses[self.last_restart:]
            median_fitnesses = self.median_fitnesses[self.last_restart:]
            time_since_restart = self.t - self.last_restart
            self.termination_criteria = (
                dict()
                if self.lambda_ > self.max_lambda_
                else {
                    "max_iter": (time_since_restart > self.max_iter),
                    "equalfunvalues": (
                        len(best_fopts) > self.nbin
                        and np.ptp(best_fopts[-self.nbin:]) == 0
                    ),
                    "flat_fitness": (
                        time_since_restart > self.flat_fitnesses.maxlen
                        and len(self.flat_fitnesses) == self.flat_fitnesses.maxlen
                        and np.sum(self.flat_fitnesses) > (self.d / 3)
                    ),
                    "tolx": np.all(
                        (np.append(self.pc.T, diag_C) * d_sigma)
                        < (self.tolx * self.sigma0)
                    ),
                    "tolupsigma": (d_sigma > self.tolup_sigma * np.sqrt(self.D.max())),
                    "conditioncov": np.linalg.cond(self.C) > self.condition_cov,
                    "noeffectaxis": np.all(
                        (
                            1 * self.sigma * np.sqrt(self.D[_t, 0]) * self.B[:, _t]
                            + self.m
                        )
                        == self.m
                    ),
                    "noeffectcoor": np.any(
                        (0.2 * self.sigma * np.sqrt(diag_C) + self.m) == self.m
                    ),
                    "stagnation": (
                        time_since_restart > self.n_stagnation
                        and (
                            np.median(best_fopts[-int(0.3 * time_since_restart):])
                            >= np.median(best_fopts[: int(0.3 * time_since_restart)])
                            and np.median(median_fitnesses[-int(0.3 * time_since_restart):])
                            >= np.median(median_fitnesses[: int(0.3 * time_since_restart)])
                        )
                    ),
                }
            )

    def update(self, parameters: dict, reset_default_modules=False):
        """Method to update the values of self based on a given dict of new parameters.

        Note that some updated parameters might be overridden by:
            self.init_selection_parameters()
            self.init_adaptation_parameters()
            self.init_local_restart_parameters()
        which are called at the end of this function. Use with caution.


        Parameters
        ----------
        parameters: dict
            A dict with new parameter values

        reset_default_modules: bool = False
            Whether to reset the modules back to their default values.

        """
        if reset_default_modules:
            for name in Parameters.__modules__:
                default_option, *_ = getattr(
                    getattr(Parameters, name), "options", [False, True]
                )
                setattr(self, name, default_option)

        for name, value in parameters.items():
            if not hasattr(self, name):
                raise ValueError(f"The parameter {name} doesn't exist")
            setattr(self, name, value)

        self.init_selection_parameters()
        self.init_adaptation_parameters()
        self.init_local_restart_parameters()
        
    def update_popsize(self, lambda_new):
        """Manually control the population size."""
        if self.local_restart is not None:
            warnings.warn("Modification of population size is disabled when local restart startegies are used")
            return
        self.lambda_ = lambda_new
        self.mu = lambda_new//2
        self.init_selection_parameters()
        self.init_adaptation_parameters()
        self.init_local_restart_parameters()


class BIPOPParameters(AnnotatedStruct):
    """Object which holds BIPOP specific parameters."""

    lambda_init: int
    budget: int
    mu_factor: float
    lambda_large: int = None
    budget_small: int = None
    budget_large: int = None
    used_budget: int = 0

    @property
    def large(self) -> bool:
        """Determine where to use a large regime."""
        if (self.budget_large >= self.budget_small) and self.budget_large > 0:
            return True
        return False

    @property
    def remaining_budget(self) -> int:
        """Compute the remaining budget."""
        return self.budget - self.used_budget

    @property
    def lambda_(self) -> int:
        """Return value for lambda, based which regime is active."""
        return self.lambda_large if self.large else self.lambda_small

    @property
    def sigma(self) -> float:
        """Return value for sigma, based on which regime is active."""
        return 2 if self.large else 2e-2 * np.random.uniform()

    @property
    def mu(self) -> int:
        """Return value for mu."""
        return np.floor(self.lambda_ * self.mu_factor).astype(int)

    def adapt(self, used_budget: int) -> None:
        """Adapt the parameters for BIPOP on restart."""
        used_previous_iteration = used_budget - self.used_budget
        self.used_budget += used_previous_iteration

        if self.lambda_large is None:
            self.lambda_large = self.lambda_init * 2
            self.budget_small = self.remaining_budget // 2
            self.budget_large = self.remaining_budget - self.budget_small
        elif self.large:
            self.budget_large -= used_previous_iteration
            self.lambda_large *= 2
        else:
            self.budget_small -= used_previous_iteration

        self.lambda_small = np.floor(
            self.lambda_init
            * (0.5 * self.lambda_large / self.lambda_init) ** (np.random.uniform() ** 2)
        ).astype(int)

        if self.lambda_small % 2 != 0:
            self.lambda_small += 1
