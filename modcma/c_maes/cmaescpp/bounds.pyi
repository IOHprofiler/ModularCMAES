import c_maes.cmaescpp.sampling
import numpy

class BoundCorrection:
    db: numpy.ndarray[numpy.float64[m, 1]]
    diameter: float
    lb: numpy.ndarray[numpy.float64[m, 1]]
    ub: numpy.ndarray[numpy.float64[m, 1]]
    @property
    def n_out_of_bounds(self) -> int: ...

class COTN(BoundCorrection):
    def __init__(self, dimension: int) -> None: ...
    def correct(
        self,
        X: numpy.ndarray[numpy.float64[m, n]],
        Y: numpy.ndarray[numpy.float64[m, n]],
        s: numpy.ndarray[numpy.float64[m, 1]],
        m: numpy.ndarray[numpy.float64[m, 1]],
    ) -> None: ...
    @property
    def sampler(self) -> c_maes.cmaescpp.sampling.Gaussian: ...

class CountOutOfBounds(BoundCorrection):
    def __init__(self, dimension: int) -> None: ...
    def correct(
        self,
        X: numpy.ndarray[numpy.float64[m, n]],
        Y: numpy.ndarray[numpy.float64[m, n]],
        s: numpy.ndarray[numpy.float64[m, 1]],
        m: numpy.ndarray[numpy.float64[m, 1]],
    ) -> None: ...

class Mirror(BoundCorrection):
    def __init__(self, dimension: int) -> None: ...
    def correct(
        self,
        X: numpy.ndarray[numpy.float64[m, n]],
        Y: numpy.ndarray[numpy.float64[m, n]],
        s: numpy.ndarray[numpy.float64[m, 1]],
        m: numpy.ndarray[numpy.float64[m, 1]],
    ) -> None: ...

class NoCorrection(BoundCorrection):
    def __init__(self, dimension: int) -> None: ...
    def correct(
        self,
        X: numpy.ndarray[numpy.float64[m, n]],
        Y: numpy.ndarray[numpy.float64[m, n]],
        s: numpy.ndarray[numpy.float64[m, 1]],
        m: numpy.ndarray[numpy.float64[m, 1]],
    ) -> None: ...

class Saturate(BoundCorrection):
    def __init__(self, dimension: int) -> None: ...
    def correct(
        self,
        X: numpy.ndarray[numpy.float64[m, n]],
        Y: numpy.ndarray[numpy.float64[m, n]],
        s: numpy.ndarray[numpy.float64[m, 1]],
        m: numpy.ndarray[numpy.float64[m, 1]],
    ) -> None: ...

class Toroidal(BoundCorrection):
    def __init__(self, dimension: int) -> None: ...
    def correct(
        self,
        X: numpy.ndarray[numpy.float64[m, n]],
        Y: numpy.ndarray[numpy.float64[m, n]],
        s: numpy.ndarray[numpy.float64[m, 1]],
        m: numpy.ndarray[numpy.float64[m, 1]],
    ) -> None: ...

class UniformResample(BoundCorrection):
    def __init__(self, dimension: int) -> None: ...
    def correct(
        self,
        X: numpy.ndarray[numpy.float64[m, n]],
        Y: numpy.ndarray[numpy.float64[m, n]],
        s: numpy.ndarray[numpy.float64[m, 1]],
        m: numpy.ndarray[numpy.float64[m, 1]],
    ) -> None: ...
