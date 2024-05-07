import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.sampling
import numpy

class BoundCorrection:
    db: numpy.ndarray
    diameter: float
    lb: numpy.ndarray
    ub: numpy.ndarray
    def __init__(self, *args, **kwargs) -> None: ...
    def correct(
        self, population: modcma.c_maes.cmaescpp.Population, m: numpy.ndarray
    ) -> None: ...
    @property
    def n_out_of_bounds(self) -> int: ...

class COTN(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...
    @property
    def sampler(self) -> modcma.c_maes.cmaescpp.sampling.Gaussian: ...

class Mirror(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...

class NoCorrection(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...

class Saturate(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...

class Toroidal(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...

class UniformResample(BoundCorrection):
    def __init__(self, lb: numpy.ndarray, ub: numpy.ndarray) -> None: ...
