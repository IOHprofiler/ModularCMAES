from typing import Any, Callable, List, Optional, Union

from typing import overload
import numpy
from . import matrix_adaptation, sampling, parameters, mutation, restart, repelling

class Solution:
    x: numpy.ndarray
    y: float
    t: int
    e: int

class ModularCMAES:
    def __init__(self, parameters: Parameters) -> None: ...
    def adapt(self) -> None: ...
    def break_conditions(self) -> bool: ...
    def mutate(self, arg0: Callable[[numpy.ndarray], float]) -> None: ...
    def recombine(self) -> None: ...
    def run(self, objective: Callable[[numpy.ndarray], float]) -> None: ...
    def select(self) -> None: ...
    def step(self, objective: Callable[[numpy.ndarray], float]) -> bool: ...
    def __call__(self, objective: Callable[[numpy.ndarray], float]) -> None: ...
    @property
    def p(self) -> Parameters: ...

class Parameters:
    adaptation: Union[
        matrix_adaptation.MatrixAdaptation,
        matrix_adaptation.CovarianceAdaptation,
        matrix_adaptation.NoAdaptation,
    ]
    bounds: Any
    lamb: int
    mu: int
    mutation: mutation.Strategy
    old_pop: Population
    pop: Population
    restart: restart.Strategy
    repelling: repelling.Repelling
    sampler: sampling.Sampler
    selection: Any
    settings: parameters.Settings
    stats: parameters.Stats
    weights: parameters.Weights
    @overload
    def __init__(self, dimension: int) -> None: ...
    @overload
    def __init__(self, settings: parameters.Settings) -> None: ...
    def adapt(self) -> None: ...
    def perform_restart(self, sigma: Optional[float] = ...) -> None: ...

class Population:
    X: numpy.ndarray
    Y: numpy.ndarray
    Z: numpy.ndarray
    d: int
    f: numpy.ndarray
    n: int
    s: numpy.ndarray
    @overload
    def __init__(self, dimension: int, n: int) -> None: ...
    @overload
    def __init__(
        self,
        X: numpy.ndarray,
        Z: numpy.ndarray,
        Y: numpy.ndarray,
        f: numpy.ndarray,
        s: numpy.ndarray,
    ) -> None: ...
    def keep_only(self, idx: List[int]) -> None: ...
    def resize_cols(self, size: int) -> None: ...
    def sort(self) -> None: ...
    def __add__(self, other: Population) -> Population: ...
    @property
    def n_finite(self) -> int: ...
