from collections.abc import Iterable, Iterator, Sequence
from typing import Annotated, ClassVar, overload

import numpy
from numpy.typing import NDArray

import modcma.c_maes.cmaescpp


class Criterion:
    def __init__(self, name: str) -> None: ...

    def on_reset(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None: ...

    def update(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None: ...

    def reset(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None: ...

    @property
    def met(self) -> bool: ...

    @met.setter
    def met(self, arg: bool, /) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def last_restart(self) -> int: ...

    @last_restart.setter
    def last_restart(self, arg: int, /) -> None: ...

    def __repr__(self) -> str: ...

class ExceededMaxIter(Criterion):
    def __init__(self) -> None: ...

    @property
    def max_iter(self) -> int: ...

    @max_iter.setter
    def max_iter(self, arg: int, /) -> None: ...

class NoImprovement(Criterion):
    def __init__(self) -> None: ...

    @property
    def n_bin(self) -> int: ...

    @n_bin.setter
    def n_bin(self, arg: int, /) -> None: ...

    @property
    def best_fitnesses(self) -> list[float]: ...

    @best_fitnesses.setter
    def best_fitnesses(self, arg: Sequence[float], /) -> None: ...

class MaxSigma(Criterion):
    def __init__(self) -> None: ...

    tolerance: ClassVar[float] = ...
    """(arg: object, /) -> float"""

class MinSigma(Criterion):
    def __init__(self) -> None: ...

    tolerance: ClassVar[float] = ...
    """(arg: object, /) -> float"""

class UnableToAdapt(Criterion):
    def __init__(self) -> None: ...

class FlatFitness(Criterion):
    def __init__(self) -> None: ...

    @property
    def max_flat_fitness(self) -> int: ...

    @max_flat_fitness.setter
    def max_flat_fitness(self, arg: int, /) -> None: ...

    @property
    def flat_fitness_index(self) -> int: ...

    @flat_fitness_index.setter
    def flat_fitness_index(self, arg: int, /) -> None: ...

    @property
    def flat_fitnesses(self) -> Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')]: ...

    @flat_fitnesses.setter
    def flat_fitnesses(self, arg: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')], /) -> None: ...

class TolX(Criterion):
    def __init__(self) -> None: ...

    @property
    def tolx_vector(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    @tolx_vector.setter
    def tolx_vector(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    tolerance: ClassVar[float] = ...
    """(arg: object, /) -> float"""

class MaxDSigma(Criterion):
    def __init__(self) -> None: ...

    tolerance: ClassVar[float] = ...
    """(arg: object, /) -> float"""

class MinDSigma(Criterion):
    def __init__(self) -> None: ...

    tolerance: ClassVar[float] = ...
    """(arg: object, /) -> float"""

class ConditionC(Criterion):
    def __init__(self) -> None: ...

    tolerance: ClassVar[float] = ...
    """(arg: object, /) -> float"""

class NoEffectAxis(Criterion):
    def __init__(self) -> None: ...

    tolerance: ClassVar[float] = ...
    """(arg: object, /) -> float"""

class NoEffectCoord(Criterion):
    def __init__(self) -> None: ...

    tolerance: ClassVar[float] = ...
    """(arg: object, /) -> float"""

class Stagnation(Criterion):
    def __init__(self) -> None: ...

    @property
    def n_stagnation(self) -> int: ...

    @n_stagnation.setter
    def n_stagnation(self, arg: int, /) -> None: ...

    @property
    def median_fitnesses(self) -> list[float]: ...

    @median_fitnesses.setter
    def median_fitnesses(self, arg: Sequence[float], /) -> None: ...

    @property
    def best_fitnesses(self) -> list[float]: ...

    @best_fitnesses.setter
    def best_fitnesses(self, arg: Sequence[float], /) -> None: ...

    tolerance: ClassVar[float] = ...
    """(arg: object, /) -> float"""

class TooMuchRepelling(Criterion):
    def __init__(self) -> None: ...

    tolerance: ClassVar[float] = ...
    """(arg: object, /) -> float"""

class CriteriaVector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: CriteriaVector, /) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[Criterion], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    @overload
    def __repr__(self) -> str: ...

    @overload
    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[Criterion]: ...

    @overload
    def __getitem__(self, arg: int, /) -> Criterion: ...

    @overload
    def __getitem__(self, arg: slice, /) -> CriteriaVector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: Criterion, /) -> None:
        """Append ``arg`` to the end of the list."""

    def insert(self, arg0: int, arg1: Criterion, /) -> None:
        """Insert object ``arg1`` before index ``arg0``."""

    def pop(self, index: int = -1) -> Criterion:
        """Remove and return item at ``index`` (default last)."""

    def extend(self, arg: CriteriaVector, /) -> None:
        """Extend ``self`` by appending elements from ``arg``."""

    @overload
    def __setitem__(self, arg0: int, arg1: Criterion, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: CriteriaVector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: Criterion, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: Criterion, /) -> int:
        """Return number of occurrences of ``arg``."""

    def remove(self, arg: Criterion, /) -> None:
        """Remove first occurrence of ``arg``."""

class Criteria:
    @property
    def items(self) -> CriteriaVector: ...

    @items.setter
    def items(self, arg: CriteriaVector, /) -> None: ...

    def reset(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None: ...

    def update(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None: ...

    def reason(self) -> str: ...

    def any(self) -> bool: ...

class Strategy:
    def update(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> float: ...

class IPOP(Strategy):
    @property
    def ipop_factor(self) -> float: ...

    @ipop_factor.setter
    def ipop_factor(self, arg: float, /) -> None: ...

class BIPOP(Strategy):
    def large(self) -> bool: ...

    @property
    def mu_factor(self) -> float: ...

    @mu_factor.setter
    def mu_factor(self, arg: float, /) -> None: ...

    @property
    def lambda_init(self) -> int: ...

    @lambda_init.setter
    def lambda_init(self, arg: int, /) -> None: ...

    @property
    def budget(self) -> int: ...

    @budget.setter
    def budget(self, arg: int, /) -> None: ...

    @property
    def lambda_large(self) -> int: ...

    @lambda_large.setter
    def lambda_large(self, arg: int, /) -> None: ...

    @property
    def lambda_small(self) -> int: ...

    @lambda_small.setter
    def lambda_small(self, arg: int, /) -> None: ...

    @property
    def budget_small(self) -> int: ...

    @budget_small.setter
    def budget_small(self, arg: int, /) -> None: ...

    @property
    def budget_large(self) -> int: ...

    @budget_large.setter
    def budget_large(self, arg: int, /) -> None: ...

    @property
    def used_budget(self) -> int: ...
