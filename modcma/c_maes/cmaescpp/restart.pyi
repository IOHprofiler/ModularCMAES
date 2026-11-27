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

class ExceededMaxIter(Criterion):
    """
    Met when a run has lasted more than `max_iter` generations since the last restart.
    """

    def __init__(self) -> None:
        """
        The criterion; `max_iter` is set by the first reset (or, added to a running optimizer, before its first update).
        """

    @property
    def max_iter(self) -> int:
        """
        The most generations of one run, set at every reset to 100 + 50 (d + 3)^2 / sqrt(lambda), or to the smaller `Strategy.max_iter` of the restart strategy (BIPOP's small runs).
        """

    @max_iter.setter
    def max_iter(self, arg: int, /) -> None: ...

class NoImprovement(Criterion):
    """
    Met when the best fitness of a generation has not changed over the last `n_bin` generations.

    Checked from n_bin generations after a restart on: met when the range (max - min) of the best fitness of the last n_bin generations is at most `NoImprovement.tolerance`.
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

class MaxSigma(Criterion):
    """
    Met when sigma max(D) has grown by more than `MaxSigma.tolerance` since the start of the run: TolXUp of Hansen (2016, App. B.3), "usually indicates a far too small initial sigma, or divergent behavior".

    D are the axis lengths of the distribution, the square roots of the eigenvalues of C, from `Adaptation.metric()`, for COVARIANCE, SEPARABLE and NONE. The others, which would decompose their matrix every generation, use sqrt(max C_ii) (`Adaptation.coordinate_wise_variances`) instead of max(D): for them the test is pycma's tolfacupx, sigma sqrt(C_ii) > 1e3 sigma0 for some i, with 1e4. max C_ii is at most the largest eigenvalue, so it fires later than with the eigenvalues. Before 2.0 the criterion compared sigma with 1e4, whatever sigma0 and C.
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

class MinSigma(Criterion):
    """Met when the step size sigma is smaller than `MinSigma.tolerance`."""

    def __init__(self) -> None:
        """The criterion."""

    tolerance: ClassVar[float] = ...
    """
    The smallest step size (default 1e-20); also the lower limit of `constants.clip_sigma`.
    """

class UnableToAdapt(Criterion):
    """
    Met when the matrix adaptation failed (e.g. its eigendecomposition or Cholesky factorization) or sigma is not finite.

    Always among the criteria, also with RestartStrategy.NONE.
    """

    def __init__(self) -> None:
        """The criterion."""

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
        The rank compared with the best, set at every reset to round(0.1 + lambda / 4).
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

class TolX(Criterion):
    """
    Met when the search distribution has become very small in every coordinate.

    Met when every entry of `tolx_vector` (sigma times the standard deviations sqrt(C_ii) and the absolute values of the evolution path p_c) is smaller than tolerance times the step size at the start of the run (Hansen 2016, App. B.3; pycma's tolx).

    For every matrix adaptation: C_ii from `Adaptation.coordinate_wise_variances`, and p_c where the adaptation has one (COVARIANCE, COVARIANCE_NO_EIGV, SEPARABLE, CHOLESKY).
    """

    def __init__(self) -> None:
        """The criterion."""

    @property
    def tolx_vector(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The values compared: sigma sqrt(C_ii) (d values), then sigma |p_c| (d more, for the adaptations with p_c).
        """

    @tolx_vector.setter
    def tolx_vector(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    tolerance: ClassVar[float] = ...
    """The tolerance (default 1e-11)."""

class MaxDSigma(Criterion):
    """
    Met when sigma has grown too much relative to sigma0: sigma / sigma0 > tolerance * max(D), with D the square roots of the eigenvalues of C (pycma's tolupsigma).

    D from `Adaptation.metric()` for COVARIANCE, SEPARABLE and NONE. The others use sqrt(max C_ii) (`Adaptation.coordinate_wise_variances`) instead of max(D): free, and it follows the scale of C. max C_ii is at most the largest eigenvalue, so the criterion fires earlier than with the eigenvalues; no source does this.
    """

    def __init__(self) -> None:
        """The criterion."""

    tolerance: ClassVar[float] = ...
    """The factor (default 1e20)."""

class MinDSigma(Criterion):
    """
    Met when sigma has shrunk too much relative to sigma0: sigma / sigma0 < tolerance * min(D), with D the square roots of the eigenvalues of C.

    D from `Adaptation.metric()` for COVARIANCE, SEPARABLE and NONE. The others use sqrt(min C_ii) (`Adaptation.coordinate_wise_variances`) instead of min(D). min C_ii is at least the smallest eigenvalue, so the criterion fires earlier than with the eigenvalues; no source does this.
    """

    def __init__(self) -> None:
        """The criterion."""

    tolerance: ClassVar[float] = ...
    """The factor (default 1e-20)."""

class ConditionC(Criterion):
    """
    Met when the condition number of the covariance matrix, max(D)^2 / min(D)^2, exceeds the tolerance (pycma's tolconditioncov).

    D from `Adaptation.metric()`. Created by default for COVARIANCE, SEPARABLE and NONE, which keep the metric; added to the others, it decomposes their matrix once per update of it, O(d^3).
    """

    def __init__(self) -> None:
        """The criterion."""

    tolerance: ClassVar[float] = ...
    """The largest condition number (default 1e14)."""

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

class Stagnation(Criterion):
    """
    Met when neither the best nor the median fitness of the generations improves any more.

    From `n_stagnation` generations after a restart on, the history of each is split at tolerance (0.3) of its length; met when the median of the later part is not lower than the median of the earlier part, for both the best and the median fitness.
    """

    def __init__(self) -> None:
        """
        The criterion; `n_stagnation` is set by the first reset (or, added to a running optimizer, before its first update).
        """

    @property
    def n_stagnation(self) -> int:
        """
        The generations before the criterion is checked, set at every reset to 100 + 100 d^1.5 / lambda.
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

class TooMuchRepelling(Criterion):
    """
    Met when repelling rejected too many samples: on average at least tolerance rejections per individual in a generation.

    Used when `Modules.repelling_type` is not NONE.
    """

    def __init__(self) -> None:
        """The criterion."""

    tolerance: ClassVar[float] = ...
    """The rejections per individual (default 50)."""

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

class IPOP(Strategy):
    """
    IPOP (Auger & Hansen 2005): every restart multiplies lambda by `ipop_factor` (mu keeps its ratio to lambda), up to `max_lambda_factor` * lambda0 (2^9 lambda0, the framework paper).
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
        The largest population size as a multiple of lambda0 (default 512 = 2^9).
        """

    @max_lambda_factor.setter
    def max_lambda_factor(self, arg: int, /) -> None: ...

class BIPOP(Strategy):
    """
    BIPOP (Hansen 2009): restarts alternate between a large and a small population regime.

    Each regime counts the evaluations of its runs; the first run counts as small (pycma's fmin; Hansen 2009 disregards it). A restart uses the regime that has used fewer evaluations, the large one on a tie, so the first restart is large. The large regime doubles lambda_large each time and restarts with sigma0. The small regime draws lambda_small = floor(lambda_init (lambda_large / (2 lambda_init))^(u^2)) (rounded up to even for PAIRWISE, at least 2) and restarts with sigma0 * 10^(-2u'), u and u' uniform in [0, 1); a small run takes at most half the evaluations of the large regime so far (`Strategy.max_iter`, which `ExceededMaxIter` enforces). mu is lambda * mu_factor (at least 1).
    """

    def large(self) -> bool:
        """
        Whether the next run is large: the large regime has used no more evaluations than the small one.
        """

    @property
    def mu_factor(self) -> float:
        """mu / lambda of the initial population."""

    @mu_factor.setter
    def mu_factor(self, arg: float, /) -> None: ...

    @property
    def lambda_init(self) -> int:
        """The initial lambda."""

    @lambda_init.setter
    def lambda_init(self, arg: int, /) -> None: ...

    @property
    def lambda_large(self) -> int:
        """
        lambda of the next large run (0 before the first restart, then 2 lambda_init, doubled after each large run).
        """

    @lambda_large.setter
    def lambda_large(self, arg: int, /) -> None: ...

    @property
    def lambda_small(self) -> int:
        """lambda of the last small-regime draw."""

    @lambda_small.setter
    def lambda_small(self, arg: int, /) -> None: ...

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
