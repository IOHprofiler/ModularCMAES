from collections.abc import Callable, Sequence
from typing import Annotated, Any, ClassVar, overload

import numpy
import numpy.typing
from numpy.typing import NDArray

from modcma.c_maes.cmaescpp import (
    bounds as bounds,
    center as center,
    es as es,
    matrix_adaptation as matrix_adaptation,
    mutation as mutation,
    options as options,
    parameters as parameters,
    repelling as repelling,
    restart as restart,
    sampling as sampling,
    selection as selection,
    utils as utils
)


class constants:
    cache_max_doubles: ClassVar[int] = ...
    """(arg: object, /) -> int"""

    cache_min_samples: ClassVar[int] = ...
    """(arg: object, /) -> int"""

    cache_samples: ClassVar[bool] = ...
    """(arg: object, /) -> bool"""

    clip_sigma: ClassVar[bool] = ...
    """(arg: object, /) -> bool"""

    use_box_muller: ClassVar[bool] = ...
    """(arg: object, /) -> bool"""

    eigen_solver: ClassVar[int] = ...
    """(arg: object, /) -> int"""

    cholesky_solver: ClassVar[int] = ...
    """(arg: object, /) -> int"""

    blas: ClassVar[int] = ...
    """(arg: object, /) -> int"""

    blas_min_dim: ClassVar[int] = ...
    """(arg: object, /) -> int"""

    lapack_min_dim: ClassVar[int] = ...
    """(arg: object, /) -> int"""

class Population:
    @overload
    def __init__(self, dimension: int, n: int) -> None: ...

    @overload
    def __init__(self, X: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], Z: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], Y: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], f: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], S: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]) -> None: ...

    def sort(self) -> None: ...

    def resize_cols(self, size: int) -> None: ...

    def keep_only(self, idx: Sequence[int]) -> None: ...

    @property
    def n_finite(self) -> int: ...

    def __add__(self, other: Population) -> Population: ...

    @property
    def X(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @X.setter
    def X(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def X_transformed(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @X_transformed.setter
    def X_transformed(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def Z(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @Z.setter
    def Z(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def Y(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @Y.setter
    def Y(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def S(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @S.setter
    def S(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def f(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    @f.setter
    def f(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def t(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    @t.setter
    def t(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def d(self) -> int: ...

    @d.setter
    def d(self, arg: int, /) -> None: ...

    @property
    def n(self) -> int: ...

    @n.setter
    def n(self, arg: int, /) -> None: ...

    @property
    def capacity(self) -> int: ...

class Parameters:
    @overload
    def __init__(self, dimension: int) -> None: ...

    @overload
    def __init__(self, settings: parameters.Settings) -> None: ...

    def adapt(self) -> None: ...

    def start(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], float]) -> None: ...

    def perform_restart(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], float], sigma: float | None = None) -> None: ...

    def resize_population(self, new_mu: int, new_lambda: int) -> None: ...

    @property
    def settings(self) -> parameters.Settings: ...

    @settings.setter
    def settings(self, arg: parameters.Settings, /) -> None: ...

    @property
    def mu(self) -> int: ...

    @mu.setter
    def mu(self, arg: int, /) -> None: ...

    @property
    def lamb(self) -> int: ...

    @lamb.setter
    def lamb(self, arg: int, /) -> None: ...

    @property
    def adaptation(self) -> matrix_adaptation.Adaptation: ...

    @adaptation.setter
    def adaptation(self, arg: matrix_adaptation.Adaptation, /) -> None: ...

    @property
    def criteria(self) -> restart.Criteria: ...

    @criteria.setter
    def criteria(self, arg: restart.Criteria, /) -> None: ...

    @property
    def stats(self) -> parameters.Stats: ...

    @stats.setter
    def stats(self, arg: parameters.Stats, /) -> None: ...

    @property
    def weights(self) -> parameters.Weights: ...

    @weights.setter
    def weights(self, arg: parameters.Weights, /) -> None: ...

    @property
    def pop(self) -> Population: ...

    @pop.setter
    def pop(self, arg: Population, /) -> None: ...

    @property
    def old_pop(self) -> Population: ...

    @old_pop.setter
    def old_pop(self, arg: Population, /) -> None: ...

    @property
    def sampler(self) -> sampling.Sampler: ...

    @sampler.setter
    def sampler(self, arg: sampling.Sampler, /) -> None: ...

    @property
    def mutation(self) -> mutation.Strategy: ...

    @mutation.setter
    def mutation(self, arg: mutation.Strategy, /) -> None: ...

    @property
    def selection(self) -> selection.Strategy: ...

    @selection.setter
    def selection(self, arg: selection.Strategy, /) -> None: ...

    @property
    def restart_strategy(self) -> restart.Strategy: ...

    @restart_strategy.setter
    def restart_strategy(self, arg: restart.Strategy, /) -> None: ...

    @property
    def repelling(self) -> repelling.Repelling: ...

    @repelling.setter
    def repelling(self, arg: repelling.Repelling, /) -> None: ...

    @property
    def bounds(self) -> bounds.BoundCorrection: ...

    @bounds.setter
    def bounds(self, arg: bounds.BoundCorrection, /) -> None: ...

    @property
    def center_placement(self) -> center.Placement: ...

    @center_placement.setter
    def center_placement(self, arg: center.Placement, /) -> None: ...

    @property
    def coordinate_mapping(self) -> parameters.CoordinateMapping: ...

    @coordinate_mapping.setter
    def coordinate_mapping(self, arg: parameters.CoordinateMapping, /) -> None: ...

class ModularCMAES:
    @overload
    def __init__(self, parameters: Parameters) -> None: ...

    @overload
    def __init__(self, dimension: int) -> None: ...

    @overload
    def __init__(self, settings: parameters.Settings) -> None: ...

    def recombine(self) -> None: ...

    def mutate(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], float]) -> None: ...

    def select(self) -> None: ...

    def adapt(self) -> None: ...

    def step(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], Any], *, vectorized: bool = False) -> bool:
        """
        One generation. Returns False when a break condition holds. A vectorized objective is called with every batch of points as a (k, d) array and returns k values.
        """

    def __call__(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], Any], *, vectorized: bool = False) -> None: ...

    def run(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], Any], *, vectorized: bool = False) -> None:
        """
        Run until a break condition holds. A vectorized objective is called with every batch of points as a (k, d) array and returns k values.
        """

    def ask(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')]:
        """
        The points to evaluate next, a new (k, d) array (row i is point i). Starts a generation when none is in progress; until tell(), it returns the same points.
        """

    def tell(self, f: numpy.typing.ArrayLike, X: numpy.typing.ArrayLike | None = None) -> bool:
        """
        The objective values of the points of ask(), in order. X (k, d): the points that were evaluated, when they differ from the asked ones (repaired or injected). Returns True when this completed a generation.
        """

    @property
    def has_pending(self) -> bool:
        """Whether ask() has points that wait for tell()"""

    def break_conditions(self) -> bool: ...

    @property
    def p(self) -> Parameters: ...
