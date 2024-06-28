import numpy
from typing import Callable
from .parameters import Solution

class NoRepelling(Repelling):
    def __init__(self) -> None: ...

class Repelling:
    archive: list[TabooPoint]
    attempts: int
    coverage: float
    def __init__(self) -> None: ...
    def is_rejected(self, xi: numpy.ndarray, p) -> bool: ...
    def prepare_sampling(self, p) -> None: ...
    def update_archive(
        self, objective: Callable[[numpy.ndarray], float], p
    ) -> None: ...

class TabooPoint:
    criticality: float
    n_rep: int
    radius: float
    shrinkage: float
    solution: Solution
    def __init__(self, solution: Solution, radius: float) -> None: ...
    def calculate_criticality(self, p) -> None: ...
    def rejects(
        self, xi: numpy.ndarray, p, attempts: int
    ) -> bool: ...
    def shares_basin(
        self,
        objective: Callable[[numpy.ndarray], float],
        xi: Solution,
        p,
    ) -> bool: ...

def euclidian(
    u: numpy.ndarray, v: numpy.ndarray
) -> float: ...
def hill_valley_test(
    u: Solution,
    v: Solution,
    f: Callable[[numpy.ndarray], float],
    n_evals: int,
) -> bool: ...
def mahanolobis(
    u: numpy.ndarray,
    v: numpy.ndarray,
    C_inv: numpy.ndarray[numpy.float64[m, n]],
) -> float: ...
def manhattan(
    u: numpy.ndarray, v: numpy.ndarray
) -> float: ...
