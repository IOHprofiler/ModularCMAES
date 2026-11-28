"""
Restart criteria and restart strategies.

After every generation, the optimizer updates the criteria in `Parameters.criteria`; when one of them is met, the next generation starts a restart (or, with RestartStrategy.STOP, the run ends). The restart strategy (`Parameters.restart_strategy`, chosen by `Modules.restart_strategy`) sets the population size and the step size of the new run. The criteria can be changed and extended: remove items, change the tolerances (static, shared by all instances of a class), or subclass `Criterion` in Python.
"""

from collections.abc import Iterable, Iterator, Sequence
from typing import Annotated, ClassVar, overload

import numpy
from numpy.typing import NDArray

import modcma.c_maes.cmaescpp


class Criterion:
    """
    A restart criterion: a condition on the state of the optimizer, checked after every generation.

    Subclass it in Python to define your own: call `super().__init__(name)`, override `update` (required) to set `self.met`, and optionally `on_reset`. Then add an instance to `Parameters.criteria.items`:

        class MyCriterion(restart.Criterion):
            def __init__(self):
                super().__init__("MyCriterion")

            def update(self, parameters):
                self.met = parameters.stats.t - self.last_restart > 100

        cma.p.criteria.items.append(MyCriterion())
    """

    def __init__(self, name: str) -> None:
        """A criterion called name (shown by `Criteria.reason`), not met."""

    def on_reset(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Called by `reset`, at the start and at every restart; override it to reset the state of a criterion (the default does nothing).
        """

    def update(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Called after every generation with the `Parameters` of the optimizer; sets `met`. Python subclasses must override it.
        """

    def reset(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Start over: sets `last_restart` to the current generation, `met` to False and calls `on_reset`.
        """

    @property
    def met(self) -> bool:
        """Whether the criterion is met (set by `update`)."""

    @met.setter
    def met(self, arg: bool, /) -> None: ...

    @property
    def name(self) -> str:
        """The name of the criterion."""

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def last_restart(self) -> int:
        """
        The generation (`Stats.t`) of the last reset, at the start or a restart.
        """

    @last_restart.setter
    def last_restart(self, arg: int, /) -> None: ...

    def __repr__(self) -> str:
        """<name met: 0 or 1>."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy of a criterion defined in Python: (name, met, last_restart, was_reset, the instance's attributes). The attributes must be picklable.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore a criterion defined in Python from a state of __getstate__ (pickle and copy).
        """

class ExceededMaxIter(Criterion):
    """
    Met when the run has lasted `max_iter` generations, so a run takes at most `max_iter` generations (at least one).

    As Hansen (2009), "MaxIter ... the maximal number of iterations in each run", and pycma's maxiter (countiter >= maxiter). `update` counts the generation it is called at: the optimizer updates the criteria at the end of a generation, before `Stats.t` counts it. Before 2.0 a run took max_iter + 2 generations (#139).
    """

    def __init__(self) -> None:
        """
        The criterion; `max_iter` is set by the first reset (or, added to a running optimizer, before its first update).
        """

    @property
    def max_iter(self) -> int:
        """
        The most generations of one run, set at every reset to 100 + 50 (d + 3)^2 / sqrt(lambda) (rounded down; lambda of the run), or to the smaller `Strategy.max_iter` of the restart strategy (BIPOP's small runs). pycma's default maxiter, 100 + 150 (d + 3)^2 / sqrt(lambda), has three times the variable term.
        """

    @max_iter.setter
    def max_iter(self, arg: int, /) -> None: ...

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class NoImprovement(Criterion):
    """
    Met when the best fitness of a generation has not changed over the last `n_bin` generations.

    Checked from n_bin generations after a restart on: met when the range (max - min) of the best fitness of the last n_bin generations is at most `NoImprovement.tolerance`. EqualFunValues of Hansen (2016, App. B.3), which stops when that range is zero.
    """

    def __init__(self) -> None:
        """
        The criterion; `n_bin` is set by the first reset (or, added to a running optimizer, before its first update).
        """

    @property
    def n_bin(self) -> int:
        """
        The number of generations compared, set at every reset to 10 + ceil(30 d / lambda).
        """

    @n_bin.setter
    def n_bin(self, arg: int, /) -> None: ...

    @property
    def best_fitnesses(self) -> list[float]:
        """The best fitness of every generation since the last reset."""

    @best_fitnesses.setter
    def best_fitnesses(self, arg: Sequence[float], /) -> None: ...

    tolerance: ClassVar[float] = ...
    """
    The largest range of the best fitness that counts as no improvement (default 1e-20).
    """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class MaxSigma(Criterion):
    """
    Met when sigma max(D) has grown by more than `MaxSigma.tolerance` since the start of the run: TolXUp of Hansen (2016, App. B.3), "usually indicates a far too small initial sigma, or divergent behavior".

    D are the axis lengths of the distribution, the square roots of the eigenvalues of C, from `Adaptation.metric()`, for COVARIANCE, SEPARABLE and NONE. The others, which would decompose their matrix every generation, use sqrt(max C_ii) (`Adaptation.coordinate_wise_variances`) instead of max(D): for them, when C starts at the identity, the test is pycma's tolfacupx (sigma sqrt(C_ii) > tolfacupx sigma0 for some i, pycma's default 1e3) with 1e4. max C_ii is at most the largest eigenvalue, so it fires later than with the eigenvalues. Before 2.0 the criterion compared sigma with 1e4, whatever sigma0 and C.
    """

    def __init__(self) -> None:
        """The criterion; `start` is set by the first reset."""

    @property
    def start(self) -> float:
        """sigma max(D) at the start of the run, set at every reset."""

    @start.setter
    def start(self, arg: float, /) -> None: ...

    tolerance: ClassVar[float] = ...
    """
    The largest growth of sigma max(D) (default 1e4). Also the upper limit of `constants.clip_sigma`, there as a step size.
    """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class MinSigma(Criterion):
    """Met when the step size sigma is smaller than `MinSigma.tolerance`."""

    def __init__(self) -> None:
        """The criterion."""

    tolerance: ClassVar[float] = ...
    """
    The smallest step size (default 1e-20); also the lower limit of `constants.clip_sigma`.
    """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class UnableToAdapt(Criterion):
    """
    Met when the matrix adaptation failed (e.g. its eigendecomposition or Cholesky factorization) or sigma is not finite.

    Always among the criteria, also with RestartStrategy.NONE, which then restarts too: a matrix that could not be updated cannot be used further.
    """

    def __init__(self) -> None:
        """The criterion."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class FlatFitness(Criterion):
    """
    Met when the fitness landscape looks flat: too often, the best individual of a generation has the same fitness as the one at rank `flat_fitness_index`.

    Of the last d generations (from d generations after a restart on), counts those in which the two fitness values are equal; met when more than `max_flat_fitness` are. Only used when lambda > 1.
    """

    def __init__(self) -> None:
        """
        The criterion; its parameters are set by the first reset (or, added to a running optimizer, before its first update).
        """

    @property
    def max_flat_fitness(self) -> int:
        """
        The most flat generations among the last d, set at every reset to ceil(d / 3).
        """

    @max_flat_fitness.setter
    def max_flat_fitness(self, arg: int, /) -> None: ...

    @property
    def flat_fitness_index(self) -> int:
        """
        The index (0-based) in the sorted population of the value compared with the best, set at every reset to round(0.1 + lambda / 4); at most the last evaluated individual (sequential selection).
        """

    @flat_fitness_index.setter
    def flat_fitness_index(self, arg: int, /) -> None: ...

    @property
    def flat_fitnesses(self) -> Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')]:
        """
        For each of the last d generations (generation t at t mod d), 1 when it was flat.
        """

    @flat_fitnesses.setter
    def flat_fitnesses(self, arg: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')], /) -> None: ...

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class TolX(Criterion):
    """
    Met when the search distribution has become very small in every coordinate.

    Met when every entry of `tolx_vector` (sigma times the standard deviations sqrt(C_ii) and the absolute values of the evolution path p_c) is smaller than tolerance times the step size at the start of the run (Hansen 2016, App. B.3, TolX: 1e-12 times the initial sigma). pycma's tolx is the same test with an absolute tolerance, 1e-11.

    For every matrix adaptation: C_ii from `Adaptation.coordinate_wise_variances`, and p_c where the adaptation has one (COVARIANCE, COVARIANCE_NO_EIGV, SEPARABLE, CHOLESKY).
    """

    def __init__(self) -> None:
        """The criterion; `sigma_start` is set by the first reset."""

    @property
    def tolx_vector(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The values compared: sigma sqrt(C_ii) (d values), then sigma |p_c| (d more, for the adaptations with p_c).
        """

    @tolx_vector.setter
    def tolx_vector(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def sigma_start(self) -> float:
        """
        The step size at the start of the run, set at every reset; the threshold is tolerance * sigma_start.
        """

    @sigma_start.setter
    def sigma_start(self, arg: float, /) -> None: ...

    tolerance: ClassVar[float] = ...
    """The tolerance (default 1e-11)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class MaxDSigma(Criterion):
    """
    Met when sigma has grown too much relative to the step size at the start of the run: sigma / sigma_start > tolerance * max(D), with D the square roots of the eigenvalues of C (TolUpSigma of Hansen 2009; pycma's tolupsigma).

    sigma_start is the sigma0 of the run, as in Hansen (2009) and pycma (es.sigma0, which its fmin sets for each run): Settings.sigma0, or the step size of the restart, smaller in the small runs of BIPOP. Before 2.0 it was Settings.sigma0 in every run (#139).

    D from `Adaptation.metric()` for COVARIANCE, SEPARABLE and NONE. The others use sqrt(max C_ii) (`Adaptation.coordinate_wise_variances`) instead of max(D): free, and it follows the scale of C. max C_ii is at most the largest eigenvalue, so the criterion fires earlier than with the eigenvalues; no source does this.
    """

    def __init__(self) -> None:
        """The criterion; `sigma_start` is set by the first reset."""

    @property
    def sigma_start(self) -> float:
        """The step size at the start of the run, set at every reset."""

    @sigma_start.setter
    def sigma_start(self, arg: float, /) -> None: ...

    tolerance: ClassVar[float] = ...
    """The factor (default 1e20)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class MinDSigma(Criterion):
    """
    Met when sigma has shrunk too much relative to the step size at the start of the run: sigma / sigma_start < tolerance * min(D), with D the square roots of the eigenvalues of C. modcma's own counterpart of `MaxDSigma`; sigma_start as there (#139).

    D from `Adaptation.metric()` for COVARIANCE, SEPARABLE and NONE. The others use sqrt(min C_ii) (`Adaptation.coordinate_wise_variances`) instead of min(D). min C_ii is at least the smallest eigenvalue, so the criterion fires earlier than with the eigenvalues; no source does this.
    """

    def __init__(self) -> None:
        """The criterion; `sigma_start` is set by the first reset."""

    @property
    def sigma_start(self) -> float:
        """The step size at the start of the run, set at every reset."""

    @sigma_start.setter
    def sigma_start(self, arg: float, /) -> None: ...

    tolerance: ClassVar[float] = ...
    """The factor (default 1e-20)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class ConditionC(Criterion):
    """
    Met when the condition number of the covariance matrix, max(D)^2 / min(D)^2, exceeds the tolerance (pycma's tolconditioncov).

    D from `Adaptation.metric()`. Created by default for COVARIANCE, SEPARABLE and NONE, which keep the metric; added to the others, it decomposes their matrix once per update of it, O(d^3).
    """

    def __init__(self) -> None:
        """The criterion."""

    tolerance: ClassVar[float] = ...
    """The largest condition number (default 1e14)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class NoEffectAxis(Criterion):
    """
    Met when a step of 0.1 standard deviations along a principal axis of C does not change the mean (pycma's noeffectaxis; Hansen 2016, App. B.3).

    In generation t, axis i = t mod d: met when m + 0.1 sigma D_i b_i == m in floating point (b_i the eigenvector, D_i the square root of its eigenvalue, from `Adaptation.metric()`). Created by default for COVARIANCE, SEPARABLE and NONE, which keep the metric; added to the others, it decomposes their matrix once per update of it, O(d^3).
    """

    def __init__(self) -> None:
        """The criterion."""

    tolerance: ClassVar[float] = ...
    """
    The step along the axis, in standard deviations (default 0.1, pycma). Before 2.0, an absolute threshold of the step (10 times the machine epsilon).
    """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class NoEffectCoord(Criterion):
    """
    Met when a step of 0.2 standard deviations in some coordinate does not change the mean: m_i + 0.2 sigma sqrt(C_ii) == m_i in floating point (pycma's noeffectcoord; Hansen 2016, App. B.3).

    For every matrix adaptation: C_ii from `Adaptation.coordinate_wise_variances`.
    """

    def __init__(self) -> None:
        """The criterion."""

    tolerance: ClassVar[float] = ...
    """
    The step along the coordinate, in standard deviations (default 0.2, pycma). Before 2.0, an absolute threshold of the step (10 times the machine epsilon).
    """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Stagnation(Criterion):
    """
    Met when neither the best nor the median fitness of the generations improves any more.

    From `n_stagnation` generations after a restart on, the history of each since the reset is split at tolerance (0.3) of its length; met when the median of the later part is not lower than the median of the earlier part, for both the best and the median fitness. Hansen's Stagnation (2016, App. B.3) compares other parts: the most recent 30 % of a window of the history (the last 20 % of the generations, at least 120 + 30 d / lambda and at most 20000) with its first 30 %.
    """

    def __init__(self) -> None:
        """
        The criterion; `n_stagnation` is set by the first reset (or, added to a running optimizer, before its first update).
        """

    @property
    def n_stagnation(self) -> int:
        """
        The generations before the criterion is checked, set at every reset to 100 + 100 d^1.5 / lambda (rounded down), the formula of pycma's default tolstagnation.
        """

    @n_stagnation.setter
    def n_stagnation(self, arg: int, /) -> None: ...

    @property
    def median_fitnesses(self) -> list[float]:
        """
        The median fitness of every generation since the last reset (a copy; assign a list to replace it).
        """

    @median_fitnesses.setter
    def median_fitnesses(self, arg: Sequence[float], /) -> None: ...

    @property
    def best_fitnesses(self) -> list[float]:
        """
        The best fitness of every generation since the last reset (a copy; assign a list to replace it).
        """

    @best_fitnesses.setter
    def best_fitnesses(self, arg: Sequence[float], /) -> None: ...

    tolerance: ClassVar[float] = ...
    """The fraction of the history that forms the earlier part (default 0.3)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class TooMuchRepelling(Criterion):
    """
    Met when repelling rejected too many samples: on average at least tolerance rejections per individual in a generation.

    Used when `Modules.repelling_type` is not NONE (and the restart strategy is not NONE).
    """

    def __init__(self) -> None:
        """The criterion."""

    tolerance: ClassVar[float] = ...
    """The rejections per individual (default 50)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class CriteriaVector:
    """
    The list of restart criteria of `Criteria.items` (a mutable sequence of `Criterion` objects).
    """

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
    """
    The restart criteria of an optimizer (`Parameters.criteria`).

    The defaults: `UnableToAdapt` always; with a restart strategy other than NONE also MinSigma, MaxSigma, ExceededMaxIter, NoImprovement, Stagnation, MinDSigma, MaxDSigma, FlatFitness (for lambda > 1), TolX and NoEffectCoord; ConditionC and NoEffectAxis with COVARIANCE, SEPARABLE and NONE (the matrix adaptations that keep their metric); with repelling also TooMuchRepelling.
    """

    @property
    def items(self) -> CriteriaVector:
        """
        The criteria, a list-like `CriteriaVector`; append, remove or assign a list of `Criterion` objects. An empty list means no restarts.
        """

    @items.setter
    def items(self, arg: CriteriaVector, /) -> None: ...

    def reset(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """Reset every criterion (`Criterion.reset`)."""

    def update(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """Update every criterion (`Criterion.update`)."""

    def reason(self) -> str:
        """
        The names of the criteria that are met, each followed by "; " (empty when none is).
        """

    def any(self) -> bool:
        """Whether any criterion is met."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Strategy:
    """
    A restart strategy: sets mu, lambda and sigma of the next run (`Parameters.restart_strategy`).

    This base class, used for RestartStrategy.NONE, STOP and RESTART, keeps mu and lambda and restarts with sigma0.
    """

    @property
    def max_iter(self) -> int:
        """
        The most generations of the current run: `ExceededMaxIter` uses the smaller of this and its own limit, at its reset. 2^64 - 1 (no limit) except in BIPOP's small runs.
        """

    @max_iter.setter
    def max_iter(self, arg: int, /) -> None: ...

    def update(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> float:
        """
        Prepare a restart: may change `Parameters.mu` and `Parameters.lamb`, and returns the step size of the next run (here sigma0).
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class IPOP(Strategy):
    """
    IPOP (Auger & Hansen 2005): every restart multiplies lambda by `ipop_factor`, up to `max_lambda_factor` * lambda0 (2^9 lambda0, the framework paper), and restarts with sigma0.

    mu is the default of the new lambda, lambda // 2 (lambda // 4 with ssa SA or matrix_adaptation CMSA), as pycma's fmin (popsize // 2) and the framework paper (Alg. 1); with a Settings.mu0 that was given, mu0 / lambda0 of the new lambda, rounded half up (at least 1). Before 2.0 mu kept the ratio mu / lambda of the previous run (#150).
    """

    @property
    def ipop_factor(self) -> float:
        """
        The factor of the population size per restart (default 2; lambda * ipop_factor is rounded).
        """

    @ipop_factor.setter
    def ipop_factor(self, arg: float, /) -> None: ...

    @property
    def max_lambda_factor(self) -> int:
        """
        The largest population size as a multiple of lambda0 (default 512 = 2^9, the same for BIPOP's large regime).
        """

    @max_lambda_factor.setter
    def max_lambda_factor(self, arg: int, /) -> None: ...

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class BIPOP(Strategy):
    """
    BIPOP (Hansen 2009): restarts alternate between a large and a small population regime.

    Each regime counts the evaluations of its runs; the first run counts as small (pycma's fmin; Hansen 2009 disregards it). A restart uses the regime that has used fewer evaluations, the large one on a tie, so the first restart is large. A large run doubles lambda_large, the population of the latest large run (lambda_init before the first), up to `max_lambda_factor` * lambda_init (2^9 lambda_init, as IPOP), and restarts with sigma0. A small run draws lambda_small = floor(lambda_init (lambda_large / lambda_init)^(u^2)) (rounded up to even for PAIRWISE, at least 2), between lambda_init and the latest large population, and restarts with sigma0 * 10^(-2u'), u and u' uniform in (0, 1), drawn from the raw output of the generator (the same on every platform, #164). A small run takes about half the evaluations of the large regime so far: `Strategy.max_iter` = ceil(0.5 * used_large / lambda), which `ExceededMaxIter` enforces, as pycma's fmin (maxiter = min(maxiter0, 0.5 * sum(large_i) / popsize), cma/evolution_strategy.py:5372, not rounded, with the stop countiter >= maxiter; before 2.0 rounded down). mu follows lambda as in `IPOP`: lambda // 2 (pycma's fmin, Hansen 2009), or the ratio of a given mu0 (#150; before 2.0, lambda mu0 / lambda0, truncated).

    Up to the limit, the population sizes are those of pycma 4.6.0's `fmin` (cma/evolution_strategy.py:5362-5381) given an integer popsize (its default, 4 + 3 ln d, is not rounded there) and without PAIRWISE. Hansen (2009) writes the same formula with lambda_l / 2 in place of lambda_l, lambda_l the latest large population: half of pycma's range. pycma has no limit on the population, but its `fmin` stops after `restarts` large runs; with 9 (restarts=True), as Hansen (2009), its largest population is 2^9 lambda_def, this limit. modcma keeps restarting at the limit until the budget is used.
    """

    def large(self) -> bool:
        """
        Whether the next run is large: the large regime has used no more evaluations than the small one.
        """

    @property
    def lambda_init(self) -> int:
        """The initial lambda."""

    @lambda_init.setter
    def lambda_init(self, arg: int, /) -> None: ...

    @property
    def lambda_large(self) -> int:
        """
        lambda of the latest large run (0 before the first restart, then 2 lambda_init, doubled at each large run up to max_lambda_factor * lambda_init).
        """

    @lambda_large.setter
    def lambda_large(self, arg: int, /) -> None: ...

    @property
    def lambda_small(self) -> int:
        """
        lambda of the last small-regime draw (one at every restart), in [lambda_init, lambda_large]: up to the latest large population.
        """

    @lambda_small.setter
    def lambda_small(self, arg: int, /) -> None: ...

    @property
    def max_lambda_factor(self) -> int:
        """
        The largest population of the large regime as a multiple of lambda_init (default 512 = 2^9, as `IPOP.max_lambda_factor`).
        """

    @max_lambda_factor.setter
    def max_lambda_factor(self, arg: int, /) -> None: ...

    @property
    def used_small(self) -> int:
        """The evaluations used by the small regime, the first run included."""

    @used_small.setter
    def used_small(self, arg: int, /) -> None: ...

    @property
    def used_large(self) -> int:
        """The evaluations used by the large regime."""

    @used_large.setter
    def used_large(self, arg: int, /) -> None: ...

    @property
    def large_run(self) -> bool:
        """
        Whether the current run belongs to the large regime (False for the first run).
        """

    @large_run.setter
    def large_run(self, arg: bool, /) -> None: ...

    @property
    def used_budget(self) -> int:
        """The evaluations used up to the last restart."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """
