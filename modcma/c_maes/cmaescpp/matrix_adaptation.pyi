import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.mutation
import numpy

class Adaptation:
    chiN: float
    dd: float
    dm: numpy.ndarray
    m: numpy.ndarray
    m_old: numpy.ndarray
    ps: numpy.ndarray
    def __init__(self, *args, **kwargs) -> None: ...
    def adapt_evolution_paths(
        self,
        pop: modcma.c_maes.cmaescpp.Population,
        weights,
        mutation: modcma.c_maes.cmaescpp.mutation.Strategy,
        stats,
        mu: int,
        lamb: int,
    ) -> None: ...
    def adapt_matrix(
        self,
        weights,
        modules,
        population: modcma.c_maes.cmaescpp.Population,
        mu: int,
        settings,
    ) -> bool: ...
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
    def adapt_covariance_matrix(
        self, weights, modules, population: modcma.c_maes.cmaescpp.Population, mu: int
    ) -> None: ...
    def perform_eigendecomposition(self, stats) -> bool: ...

class MatrixAdaptation(Adaptation):
    M: numpy.ndarray
    def __init__(self, dimension: int, x0: numpy.ndarray) -> None: ...

class NoAdaptation(Adaptation):
    def __init__(self, dimension: int, x0: numpy.ndarray) -> None: ...
