from typing import Any, Optional

import c_maes.cmaescpp
import c_maes.cmaescpp.mutation
import c_maes.cmaescpp.options
import numpy

class Dynamic:
    def __init__(self, dimension: int) -> None: ...
    def adapt_covariance_matrix(
        self,
        weights: Weights,
        modules: Modules,
        population: c_maes.cmaescpp.Population,
        mu: int,
    ) -> None: ...
    def adapt_evolution_paths(
        self,
        weights: Weights,
        mutation_strategy: c_maes.cmaescpp.mutation.Strategy,
        stats: Stats,
        lamb: int,
    ) -> None: ...
    def perform_eigendecomposition(self, stats: Stats) -> bool: ...
    @property
    def B(self) -> numpy.ndarray[numpy.float64[m, n]]: ...
    @property
    def C(self) -> numpy.ndarray[numpy.float64[m, n]]: ...
    @property
    def chiN(self) -> float: ...
    @property
    def d(self) -> numpy.ndarray[numpy.float64[m, 1]]: ...
    @property
    def dd(self) -> float: ...
    @property
    def dm(self) -> numpy.ndarray[numpy.float64[m, 1]]: ...
    @property
    def hs(self) -> bool: ...
    @property
    def inv_root_C(self) -> numpy.ndarray[numpy.float64[m, n]]: ...
    @property
    def m(self) -> numpy.ndarray[numpy.float64[m, 1]]: ...
    @property
    def m_old(self) -> numpy.ndarray[numpy.float64[m, 1]]: ...
    @property
    def pc(self) -> numpy.ndarray[numpy.float64[m, 1]]: ...
    @property
    def ps(self) -> numpy.ndarray[numpy.float64[m, 1]]: ...

class Modules:
    active: bool
    bound_correction: c_maes.cmaescpp.options.CorrectionMethod
    elitist: bool
    mirrored: c_maes.cmaescpp.options.Mirror
    orthogonal: bool
    restart_strategy: c_maes.cmaescpp.options.RestartStrategy
    sample_sigma: bool
    sampler: c_maes.cmaescpp.options.BaseSampler
    sequential_selection: bool
    ssa: c_maes.cmaescpp.options.StepSizeAdaptation
    threshold_convergence: bool
    weights: c_maes.cmaescpp.options.RecombinationWeights
    def __init__(self) -> None: ...

class Stats:
    budget: int
    evaluations: int
    fopt: float
    max_generations: int
    t: int
    target: float
    xopt: numpy.ndarray[numpy.float64[m, 1]]
    def __init__(self) -> None: ...

class Weights:
    def __init__(
        self, dimension: int, mu0: int, lambda0: int, modules: Modules
    ) -> None: ...
    @property
    def c1(self) -> float: ...
    @property
    def cc(self) -> float: ...
    @property
    def cmu(self) -> float: ...
    @property
    def mueff(self) -> float: ...
    @property
    def mueff_neg(self) -> float: ...
    @property
    def n(self) -> numpy.ndarray[numpy.float64[m, 1]]: ...
    @property
    def p(self) -> numpy.ndarray[numpy.float64[m, 1]]: ...
    @property
    def w(self) -> numpy.ndarray[numpy.float64[m, 1]]: ...
