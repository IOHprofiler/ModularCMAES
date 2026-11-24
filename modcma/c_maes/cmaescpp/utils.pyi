from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


def build_info() -> dict[str, str]:
    """How the extension was built (compiler)"""

def cdf(x: float) -> float: ...

def ppf(x: float) -> float: ...

def i8_sobol(dim_num: int, seed: int, quasi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> int:
    """
    Write the next point of the Sobol sequence into quasi and return the next seed
    """

def compute_ert(running_times: Sequence[int], budget: int) -> tuple[float, int]: ...

def set_seed(seed: int) -> None:
    """Set the random seed"""

def random_uniform() -> float:
    """Generate a uniform random number in (0, 1)"""

def random_normal() -> float:
    """Generate a standard normal random number"""

class Shuffler:
    @overload
    def __init__(self, start: int, stop: int) -> None: ...

    @overload
    def __init__(self, stop: int) -> None: ...

    def next(self) -> int: ...

    @property
    def start(self) -> int: ...

    @start.setter
    def start(self, arg: int, /) -> None: ...

    @property
    def stop(self) -> int: ...

    @stop.setter
    def stop(self, arg: int, /) -> None: ...

    @property
    def n(self) -> int: ...

    @n.setter
    def n(self, arg: int, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

    @property
    def offset(self) -> int: ...

    @offset.setter
    def offset(self, arg: int, /) -> None: ...

    @property
    def multiplier(self) -> int: ...

    @multiplier.setter
    def multiplier(self, arg: int, /) -> None: ...

    @property
    def modulus(self) -> int: ...

    @modulus.setter
    def modulus(self, arg: int, /) -> None: ...

    @property
    def found(self) -> int: ...

    @found.setter
    def found(self, arg: int, /) -> None: ...

class CachedShuffleSequence:
    def __init__(self, dim: int) -> None: ...

    def fill(self, c: Sequence[float]) -> None: ...

    def get_index(self, index: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    def next(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...
