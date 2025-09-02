import modcma.c_maes.cmaescpp
import numpy

class Adaptation:
    dd: float
    dm: numpy.ndarray
    expected_length_z: float
    inv_C: numpy.ndarray
    m: numpy.ndarray
    m_old: numpy.ndarray
    ps: numpy.ndarray
    def __init__(self, *args, **kwargs) -> None: ...
    def adapt_evolution_paths(
            self, 
            pop: modcma.c_maes.cmaescpp.Population, 
            weights: modcma.c_maes.cmaescpp.parameters.Weights, 
            stats: modcma.c_maes.cmaescpp.parameters.Stats,
            settings: modcma.c_maes.cmaescpp.parameters.Settings, 
            mu: int, lamb: int
        ) -> None: ...
    def adapt_matrix(self, weights, modules, population: modcma.c_maes.cmaescpp.Population, mu: int, settings, stats) -> bool: ...
    def compute_y(self, zi: numpy.ndarray) -> numpy.ndarray: ...
    def invert_x(self, xi: numpy.ndarray, sigma: float) -> numpy.ndarray: ...
    def invert_y(self, yi: numpy.ndarray) -> numpy.ndarray: ...
    def restart(self, settings) -> None: ...

class CovarianceAdaptation(Adaptation):
    B: numpy.ndarray
    C: numpy.ndarray
    d: numpy.ndarray
    hs: bool
    inv_root_C: numpy.ndarray
    pc: numpy.ndarray
    def __init__(self, dimension: int, x0: numpy.ndarray, expected_length_z: float) -> None: ...
    def adapt_covariance_matrix(self, weights, modules, population: modcma.c_maes.cmaescpp.Population, mu: int) -> None: ...
    def perform_eigendecomposition(self, stats) -> bool: ...

class MatrixAdaptation(Adaptation):
    M: numpy.ndarray
    M_inv: numpy.ndarray
    def __init__(self, dimension: int, x0: numpy.ndarray, expected_length_z: float) -> None: ...

class NoAdaptation(Adaptation):
    def __init__(self, dimension: int, x0: numpy.ndarray, expected_length_z: float) -> None: ...

class OnePlusOneAdaptation(CovarianceAdaptation):
    def __init__(self, dimension: int, x0: numpy.ndarray, expected_length_z: float) -> None: ...

class SeparableAdaptation(CovarianceAdaptation):
    c: numpy.ndarray
    pc: numpy.ndarray
    d: numpy.ndarray
    def __init__(self, dimension: int, x0: numpy.ndarray, expected_length_z: float) -> None: ...
    
class CovarainceNoEigvAdaptation(CovarainceNoEigvAdaptation):
    def __init__(self, dimension: int, x0: numpy.ndarray, expected_length_z: float) -> None: ...

class CholeskyAdaptation(Adaptation):
    A: numpy.ndarray
    pc: numpy.ndarray
    def __init__(self, dimension: int, x0: numpy.ndarray, expected_length_z: float) -> None: ...

class SelfAdaptation(Adaptation):
    A: numpy.ndarray
    C: numpy.ndarray
    def __init__(self, dimension: int, x0: numpy.ndarray, expected_length_z: float) -> None: ...
    
class NaturalGradientAdaptation(Adaptation):
    A: numpy.ndarray
    G: numpy.ndarray
    A_inv: numpy.ndarray
    def __init__(self, dimension: int, x0: numpy.ndarray, expected_length_z: float) -> None: ...