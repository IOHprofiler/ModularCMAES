from typing import Optional

import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.mutation
import modcma.c_maes.cmaescpp.options
import numpy

class Modules:
    active: bool
    bound_correction: modcma.c_maes.cmaescpp.options.CorrectionMethod
    elitist: bool
    mirrored: modcma.c_maes.cmaescpp.options.Mirror
    orthogonal: bool
    restart_strategy: modcma.c_maes.cmaescpp.options.RestartStrategy
    sample_sigma: bool
    sampler: modcma.c_maes.cmaescpp.options.BaseSampler
    sequential_selection: bool
    ssa: modcma.c_maes.cmaescpp.options.StepSizeAdaptation
    threshold_convergence: bool
    weights: modcma.c_maes.cmaescpp.options.RecombinationWeights
    def __init__(self) -> None: ...

class Adaptation:
    chiN: float
    dd: float
    dm: numpy.ndarray
    m: numpy.ndarray
    m_old: numpy.ndarray
    ps: numpy.ndarray
    def __init__(self, *args, **kwargs) -> None: ...
    def adapt_evolution_paths(self, pop: modcma.c_maes.cmaescpp.Population, weights: Weights, mutation: modcma.c_maes.cmaescpp.mutation.Strategy, stats: Stats, mu: int, lamb: int) -> None: ...
    def adapt_matrix(self, weights: Weights, modules: Modules, population: modcma.c_maes.cmaescpp.Population, mu: int, settings) -> bool: ...
    def restart(self, settings) -> None: ...
    def scale_mutation_steps(self, pop: modcma.c_maes.cmaescpp.Population) -> None: ...

class CovarianceAdaptation(Adaptation):
    B: numpy.ndarray
    C: numpy.ndarray
    d: numpy.ndarray
    hs: bool
    inv_root_C: numpy.ndarray
    pc: numpy.ndarray
    def __init__(self, dimension: int, x0: numpy.ndarray) -> None: ...
    def adapt_covariance_matrix(self, weights: Weights, modules: Modules, population: modcma.c_maes.cmaescpp.Population, mu: int) -> None: ...
    def perform_eigendecomposition(self, stats) -> bool: ...

class MatrixAdaptation(Adaptation):
    M: numpy.ndarray
    def __init__(self, dimension: int, x0: numpy.ndarray) -> None: ...

class Settings:
    budget: int
    c1: Optional[float]
    cc: Optional[float]
    cmu: Optional[float]
    cs: Optional[float]
    dim: int
    lambda0: int
    lb: numpy.ndarray
    max_generations: Optional[int]
    modules: Modules
    mu0: int
    sigma0: float
    target: Optional[float]
    ub: numpy.ndarray
    verbose: bool
    x0: Optional[numpy.ndarray]
    def __init__(
        self,
        dim: int,
        modules: Optional[Modules] = ...,
        target: Optional[float] = ...,
        max_generations: Optional[int] = ...,
        budget: Optional[int] = ...,
        sigma0: Optional[float] = ...,
        lambda0: Optional[int] = ...,
        mu0: Optional[int] = ...,
        x0: Optional[numpy.ndarray] = ...,
        lb: Optional[numpy.ndarray] = ...,
        ub: Optional[numpy.ndarray] = ...,
        cs: Optional[float] = ...,
        cc: Optional[float] = ...,
        cmu: Optional[float] = ...,
        c1: Optional[float] = ...,
        verbose: bool = ...,
    ) -> None: ...

class Stats:
    evaluations: int
    fopt: float
    t: int
    xopt: numpy.ndarray
    def __init__(self) -> None: ...

class Weights:
    c1: float
    cc: float
    cmu: float
    mueff: float
    mueff_neg: float
    negative: numpy.ndarray
    positive: numpy.ndarray
    weights: numpy.ndarray
    def __init__(self, dimension: int, mu0: int, lambda0: int, modules) -> None: ...

class Dynamic:
    B: numpy.ndarray
    C: numpy.ndarray
    chiN: float
    d: numpy.ndarray
    dd: float
    dm: numpy.ndarray
    hs: bool
    inv_root_C: numpy.ndarray
    m: numpy.ndarray
    m_old: numpy.ndarray
    pc: numpy.ndarray
    ps: numpy.ndarray
    def __init__(self, dimension: int, x0: numpy.ndarray) -> None: ...
    def adapt_covariance_matrix(
        self,
        weights: Weights,
        modules: Modules,
        population: modcma.c_maes.cmaescpp.Population,
        mu: int,
    ) -> None: ...
    def adapt_evolution_paths(
        self,
        weights: Weights,
        mutation: modcma.c_maes.cmaescpp.mutation.Strategy,
        stats: Stats,
        lamb: int,
    ) -> None: ...
    def perform_eigendecomposition(self, stats) -> bool: ...
