import numpy
from typing import overload

class CachedShuffleSequence:
    def __init__(self, dim: int) -> None: ...
    def fill(self, arg0: list[float]) -> None: ...
    def get_index(self, index: int) -> numpy.ndarray: ...
    def next(self) -> numpy.ndarray: ...

class Shuffler:
    found: int
    modulus: int
    multiplier: int
    n: int
    offset: int
    seed: int
    start: int
    stop: int
    @overload
    def __init__(self, start: int, stop: int) -> None: ...
    @overload
    def __init__(self, stop: int) -> None: ...
    def next(self) -> int: ...

def cdf(x: float) -> float: ...
def compute_ert(running_times: list[int], budget: int) -> tuple[float, int]: ...
def i8_sobol(dim_num: int, seed: int, quasi: float) -> None: ...
def ppf(x: float) -> float: ...
def random_normal() -> float: ...
def random_uniform() -> float: ...
def set_seed(seed: int) -> None: ...
