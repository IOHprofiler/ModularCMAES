from typing import Any, Callable, List, overload, ClassVar

import numpy
from . import (
    matrix_adaptation,
    sampling,
    parameters,
    mutation,
    restart,
    repelling,
    center,
)

class Solution:
    x: numpy.ndarray
    y: float
    t: int
    e: int

class constants:
    cache_max_doubles: ClassVar[int] = ...
    cache_min_samples: ClassVar[int] = ...
    cache_samples: ClassVar[bool] = ...
    clip_sigma: ClassVar[bool] = ...
    def __init__(self, *args, **kwargs) -> None: ...



class Population:
    X: numpy.ndarray
    Y: numpy.ndarray
    Z: numpy.ndarray
    d: int
    f: numpy.ndarray
    n: int
    s: numpy.ndarray
    t: numpy.ndarray
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
    def __add__(self, other: "Population") -> "Population": ...
    @property
    def n_finite(self) -> int: ...

class Parameters:
    adaptation: (
        matrix_adaptation.MatrixAdaptation
        | matrix_adaptation.CovarianceAdaptation
        | matrix_adaptation.SeparableAdaptation
        | matrix_adaptation.OnePlusOneAdaptation
        | matrix_adaptation.NoAdaptation
    )
    bounds: Any
    center_placement: center.Placement
    criteria: restart.Criteria
    lamb: int
    mu: int
    mutation: mutation.Strategy
    old_pop: Population
    pop: Population
    repelling: repelling.Repelling
    restart_strategy: restart.Strategy
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
    def perform_restart(
        self,
        objective: Callable[[numpy.ndarray[numpy.float64[m, 1]]], float],
        sigma: float | None = ...,
    ) -> None: ...
    def start(
        self, objective: Callable[[numpy.ndarray[numpy.float64[m, 1]]], float]
    ) -> None: ...
    
class ModularCMAES:
    @overload
    def __init__(self, parameters: Parameters) -> None: ...
    @overload
    def __init__(self, dimension: int) -> None: ...
    @overload
    def __init__(self, settings: parameters.Settings) -> None: ...
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
