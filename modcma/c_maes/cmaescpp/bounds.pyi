from typing import Annotated

import numpy
from numpy.typing import NDArray

import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.parameters
import modcma.c_maes.cmaescpp.sampling


class BoundCorrection:
    @property
    def n_out_of_bounds(self) -> int: ...

    def correct(self, index: int, parameters: modcma.c_maes.cmaescpp.Parameters) -> None: ...

    def correct_x(self, xi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], oob: Annotated[NDArray[numpy.bool_], dict(shape=(None,), order='C')], sigma: float, settings: modcma.c_maes.cmaescpp.parameters.Settings) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    def delta_out_of_bounds(self, xi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], oob: Annotated[NDArray[numpy.bool_], dict(shape=(None,), order='C')], settings: modcma.c_maes.cmaescpp.parameters.Settings) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    def is_out_of_bounds(self, xi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], settings: modcma.c_maes.cmaescpp.parameters.Settings) -> Annotated[NDArray[numpy.bool_], dict(shape=(None,), order='C')]: ...

class Resample(BoundCorrection):
    def __init__(self) -> None: ...

class NoCorrection(BoundCorrection):
    def __init__(self) -> None: ...

class COTN(BoundCorrection):
    def __init__(self, dim: int) -> None: ...

    @property
    def sampler(self) -> modcma.c_maes.cmaescpp.sampling.Gaussian: ...

class Mirror(BoundCorrection):
    def __init__(self) -> None: ...

class UniformResample(BoundCorrection):
    def __init__(self, dim: int) -> None: ...

class Saturate(BoundCorrection):
    def __init__(self) -> None: ...

class Toroidal(BoundCorrection):
    def __init__(self) -> None: ...
