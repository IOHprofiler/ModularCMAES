import modcma.c_maes.cmaescpp.sampling
import numpy

class BoundCorrection:
    db: numpy.ndarray
    diameter: float
    lb: numpy.ndarray
    ub: numpy.ndarray
    @property
    def n_out_of_bounds(self) -> int: ...
    def correct(
        self, X: numpy.ndarray, Y: numpy.ndarray, s: numpy.ndarray, m: numpy.ndarray
    ) -> None: ...

class COTN(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...
    @property
    def sampler(self) -> modcma.c_maes.cmaescpp.sampling.Gaussian: ...

class CountOutOfBounds(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...

class Mirror(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...

class NoCorrection(BoundCorrection):
    def __init__(
        self, dimension: int, lb: numpy.ndarray, ub: numpy.ndarray
    ) -> None: ...

class Saturate(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...

class Toroidal(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...

class UniformResample(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...
