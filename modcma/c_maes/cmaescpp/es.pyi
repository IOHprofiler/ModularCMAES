import modcma.c_maes.cmaescpp.bounds
import modcma.c_maes.cmaescpp.parameters
import modcma.c_maes.cmaescpp.sampling
import numpy
from typing import Callable

class MuCommaLambdaES:
    S: numpy.ndarray[numpy.float64[m, n]]
    X: numpy.ndarray[numpy.float64[m, n]]
    budget: int
    corrector: modcma.c_maes.cmaescpp.bounds.BoundCorrection
    d: int
    e: int
    f: numpy.ndarray
    f_min: float
    lamb: int
    m: numpy.ndarray
    mu: int
    mu_inv: float
    rejection_sampling: bool
    sampler: modcma.c_maes.cmaescpp.sampling.Sampler
    sigma: numpy.ndarray
    sigma_sampler: modcma.c_maes.cmaescpp.sampling.Sampler
    t: int
    target: float
    tau: float
    tau_i: float
    x_min: numpy.ndarray
    def __init__(self, d: int, x0: numpy.ndarray, sigma0: float = ..., budget: int = ..., target: float = ..., modules: modcma.c_maes.cmaescpp.parameters.Modules = ...) -> None: ...
    def sample(self, arg0: numpy.ndarray) -> numpy.ndarray: ...
    def step(self, arg0: Callable[[numpy.ndarray], float]) -> None: ...
    def __call__(self, arg0: Callable[[numpy.ndarray], float]) -> None: ...

class OnePlusOneES:
    budget: int
    corrector: modcma.c_maes.cmaescpp.bounds.BoundCorrection
    d: int
    decay: float
    f: float
    rejection_sampling: bool
    sampler: modcma.c_maes.cmaescpp.sampling.Sampler
    sigma: float
    t: int
    target: float
    x: numpy.ndarray
    def __init__(self, d: int, x0: numpy.ndarray, f0: float, sigma0: float = ..., budget: int = ..., target: float = ..., modules: modcma.c_maes.cmaescpp.parameters.Modules = ...) -> None: ...
    def sample(self) -> numpy.ndarray: ...
    def step(self, arg0: Callable[[numpy.ndarray], float]) -> None: ...
    def __call__(self, arg0: Callable[[numpy.ndarray], float]) -> None: ...
