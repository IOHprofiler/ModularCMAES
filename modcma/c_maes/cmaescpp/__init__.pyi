"""
The C++ ModularCMAES library: a modular CMA-ES in which every module (matrix adaptation, step-size adaptation, sampling, selection, restarts, bound correction, ...) is an option that combines freely with the others.

Use it through `modcma.c_maes`, which re-exports these names. The classes you need most are `ModularCMAES`, `parameters.Settings` and `parameters.Modules`, with the options in `options`. Submodules: `options` (the module options), `parameters` (settings, modules, weights, statistics), `sampling`, `mutation` (step-size adaptation), `matrix_adaptation`, `selection`, `bounds`, `center`, `repelling`, `restart` (criteria and strategies), `es` (a (1+1)-ES and a (mu, lambda)-ES) and `constants` (global switches); `set_seed`, `build_info` and `compute_ert`.
"""

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
    selection as selection
)


class constants:
    """
    Global switches of the library, as static properties (`constants.eigen_solver = 0`).

    They are process-wide: a change applies to every optimizer, also to ones that exist already, except for the cache settings, which are read when a sampler is created. Set them before creating optimizers, and not while optimizers run in other threads.
    """

    cache_max_doubles: ClassVar[int] = ...
    """
    The number of floats a sample cache may hold (default 2000000).

    A cache (the Sobol sampler, and `cache_samples`) holds max(cache_min_samples, the largest power of 2 <= cache_max_doubles / d) points of dimension d. Read when a sampler is created.
    """

    cache_min_samples: ClassVar[int] = ...
    """
    The least number of points a sample cache holds (default 128); see `cache_max_doubles`.
    """

    cache_samples: ClassVar[bool] = ...
    """
    Whether new optimizers cache their samples (default False).

    When True, the sampler of an optimizer created afterwards is wrapped in a `sampling.CachedSampler`: the first n samples (n from `cache_max_doubles` and `cache_min_samples`) are drawn and then repeated in the same cyclic order. Faster sampling, at the cost of a finite set of samples. Not applied to the SOBOL sampler, which caches its points itself.
    """

    clip_sigma: ClassVar[bool] = ...
    """
    Whether the step size is clipped after every adaptation (default False).

    When True, sigma is kept in [restart.MinSigma.tolerance, restart.MaxSigma.tolerance] (default [1e-20, 1e4]), as step sizes (the criterion MaxSigma reads its tolerance as a growth factor).
    """

    use_box_muller: ClassVar[bool] = ...
    """
    Whether the GAUSSIAN sample transformation uses the Box-Muller transform (default False).

    False: every uniform number is mapped to a normal one with the inverse normal CDF. True: pairs of uniform numbers are mapped with Box-Muller, a different random stream.
    """

    eigen_solver: ClassVar[int] = ...
    """
    The eigendecomposition of the covariance matrix (COVARIANCE) (default -1).

    -1: automatic, LAPACK dsyevd from `lapack_min_dim` on when LAPACK is available, Eigen below; 0: Eigen's SelfAdjointEigenSolver; 1: LAPACK dsyevd (divide and conquer); 2: LAPACK dsyevr (MRRR). 1 and 2 raise ValueError without LAPACK (`build_info()['lapack'] == 'none'`). LAPACK is several times faster in high dimension (d = 1000: 5.5 ms against 18.6 ms per generation, docs/performance.md); the results are not bit-identical to Eigen's.
    """

    cholesky_solver: ClassVar[int] = ...
    """
    The Cholesky factorization of CMSA and COVARIANCE_NO_EIGV (default -1).

    -1: automatic, LAPACK dpotrf from `lapack_min_dim` on when LAPACK is available, Eigen below; 0: Eigen's LLT; 1: LAPACK dpotrf (raises ValueError without LAPACK).
    """

    blas: ClassVar[int] = ...
    """
    The matrix products and the inverse of the matrix adaptations (default -1).

    Covers sampling a batch (Y = A Z), the updates of MATRIX (MA-ES) and NATURAL_GRADIENT (xNES), and the LU inverse of their matrix (dgetrf/dgetri). -1: automatic, BLAS/LAPACK from `blas_min_dim` on when available, Eigen below; 0: always Eigen; 1: always BLAS/LAPACK when available (Eigen otherwise).
    """

    blas_min_dim: ClassVar[int] = ...
    """
    The dimension from which `blas` = -1 uses BLAS and LAPACK (default 100).

    Below it, Eigen's products are faster (docs/performance.md, "LAPACK and BLAS in high dimension").
    """

    lapack_min_dim: ClassVar[int] = ...
    """
    The dimension from which `eigen_solver` = -1 and `cholesky_solver` = -1 use LAPACK (default 120).

    Below it, Eigen's decompositions are faster (docs/performance.md, "LAPACK and BLAS in high dimension").
    """

def build_info() -> dict[str, str]:
    """
    How the extension was built.

    Returns
    -------
    dict[str, str]
        'compiler': the compiler and its version, e.g. 'gcc 13.3.0';
        'fma': 'yes' when the code may use fused multiply-add instructions (which change
        the last bits of results), 'no' otherwise;
        'lapack': where LAPACK and BLAS come from: 'linked' (built with them, Accelerate
        on macOS), 'scipy' (scipy's, handed over at import) or 'none' (Eigen only).
    """

def set_seed(seed: int) -> None:
    """
    Seed the global random generator (mt19937).

    An optimizer without `Settings.seed` copies the global generator when it is created and advances it, so that after set_seed(s) the optimizers created in the same order draw the same numbers. Samplers and bound corrections that are used on their own draw from the global generator directly.
    """

def compute_ert(running_times: Sequence[int], budget: int) -> tuple[float, int]:
    """
    The expected running time (ERT) of a set of runs.

    Parameters
    ----------
    running_times : list[int]
        The evaluations each run used; a run succeeded when it used fewer than budget.
    budget : int
        The budget of each run.

    Returns
    -------
    tuple[float, int]
        The sum of all running times divided by the number of successful runs (inf when no run succeeded), and the number of successful runs.
    """

class Population:
    """
    The individuals of one generation, as the columns of (d, capacity) buffers.

    The first n columns are the active individuals. Reading an array returns a view of the active columns that shares memory with the population: X, X_transformed, Z, Y and S are (d, n) matrices, f and t are (n,) vectors. Assigning an array replaces the whole buffer and leaves n unchanged. The buffers can hold more than n columns, e.g. after sequential selection stopped a generation early; those columns hold no population data.

    A view is valid until the next generation: the optimizer then swaps pop and old_pop, and a restart or a larger population replaces the buffers (the view would read freed memory). Copy an array (`.copy()`) to keep its values.
    """

    @overload
    def __init__(self, dimension: int, n: int) -> None:
        """
        A population of n individuals in the given dimension.

        f is +inf for every individual; the other arrays are not initialised.
        """

    @overload
    def __init__(self, X: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], Z: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], Y: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], f: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], S: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]) -> None:
        """
        A population from its arrays.

        Parameters
        ----------
        X : numpy.ndarray
            (d, n) the points; X_transformed is a copy of X.
        Z : numpy.ndarray
            (d, n) the samples of the sampler.
        Y : numpy.ndarray
            (d, n) the samples transformed by the matrix adaptation.
        f : numpy.ndarray
            (n,) the objective values.
        S : numpy.ndarray
            (d, n) the step sizes.

        t (the generation of each individual) is not initialised.
        """

    def sort(self) -> None:
        """
        Sort the active individuals by f, ascending, in every array.

        Ties keep their order and NaN values go last. Every individual is then ranked (n_ranked = n).
        """

    def resize_cols(self, size: int) -> None:
        """
        Reduce the number of active individuals to min(size, n).

        The buffers keep their capacity; this never adds individuals.
        """

    def keep_only(self, idx: Sequence[int]) -> None:
        """
        Keep only the given columns, in the given order.

        Every buffer becomes its columns idx (indices into the buffers, 0..capacity-1) and n becomes len(idx).

        Parameters
        ----------
        idx : list of int
            The columns to keep.
        """

    @property
    def n_finite(self) -> int:
        """
        The number of active individuals whose f is not +inf.

        Unevaluated individuals have f = +inf. (Before 2.0, so had the worse individual of each pair under pairwise selection; it keeps its value now, see n_ranked.)
        """

    @property
    def n_ranked(self) -> int:
        """
        The number of ranked individuals, the first ones; at most n.

        n, except after pairwise selection (`Mirror.PAIRWISE`): the winners of the mirrored pairs (and the parents, with elitism) come first, sorted by f, then the losers, sorted by f, with their real values. The losers get no negative weights in the active update, and MSR and PSR compare the ranked individuals only. Set by the selection; `sort`, `keep_only` and adding populations make every individual ranked again.
        """

    @n_ranked.setter
    def n_ranked(self, arg: int, /) -> None: ...

    def __add__(self, other: Population) -> Population:
        """
        A new population with the active individuals of this one, then those of other.
        """

    def __iadd__(self, other: Population) -> Population:
        """Append the active individuals of other to this population (in place)."""

    @property
    def X(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        (d, n) array: the sampled points, X = m + S * Y (elementwise), after bound correction.

        Points of the search space before the integer variables are rounded (see X_transformed).
        """

    @X.setter
    def X(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def X_transformed(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        (d, n) array: the points that are evaluated.

        X with the integer variables rounded (CoordinateMapping); equal to X without integer variables.
        """

    @X_transformed.setter
    def X_transformed(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def Z(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        (d, n) array: the samples drawn from the sampler (standard normal by default), lengthened by threshold convergence when it is on.
        """

    @Z.setter
    def Z(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def Y(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        (d, n) array: the samples Z transformed by the matrix adaptation (the shape of the search distribution), before scaling by the step size.
        """

    @Y.setter
    def Y(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def S(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        (d, n) array: the step size of every individual and coordinate.

        Every entry is sigma, unless sample_sigma is on (each individual draws its own step sizes around sigma) or the lower bound of an integer variable raises it.
        """

    @S.setter
    def S(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def f(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (n,) array: the objective values; +inf before an individual is evaluated.
        """

    @f.setter
    def f(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def t(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (n,) array: the generation (Stats.t) in which each individual was sampled.
        """

    @t.setter
    def t(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def d(self) -> int:
        """The dimension: the number of rows of the arrays."""

    @d.setter
    def d(self, arg: int, /) -> None: ...

    @property
    def n(self) -> int:
        """
        The number of active individuals: the columns 0..n-1 of the buffers (at most capacity; otherwise ValueError).
        """

    @n.setter
    def n(self, arg: int, /) -> None: ...

    @property
    def capacity(self) -> int:
        """The number of columns of the buffers (at least n)."""

class Parameters:
    """
    The state of an optimizer: settings, module objects, populations and statistics.

    ModularCMAES works on a Parameters object (ModularCMAES.p). Every module choice of Settings.modules is an object here (adaptation, mutation, selection, sampler, bounds, ...), which can be inspected and, for experiments, changed. Every Parameters object has its own random generator: seeded with Settings.seed, or a copy of the global generator (set_seed) at creation.
    """

    @overload
    def __init__(self, dimension: int) -> None:
        """
        Parameters with the default settings for this dimension (Settings(dimension)).
        """

    @overload
    def __init__(self, settings: parameters.Settings) -> None:
        """Parameters for the given settings (copied)."""

    def adapt(self) -> None:
        """
        The adaptation at the end of a generation.

        Updates the evolution paths, the step size (clipped to the tolerances of the MinSigma and MaxSigma restart criteria when constants.clip_sigma is set), the matrix of the matrix adaptation and the restart criteria, then increments stats.t. ModularCMAES.adapt calls it.
        """

    def resize_population(self, new_mu: int, new_lambda: int) -> None:
        """
        Change the population size now.

        Sets mu and lamb and resets the sampler and the weights. The populations keep their contents: the next generation compares with this one (elitism, MSR, PSR, sequential selection) and reserves the columns it needs. The restart strategies do not call it: they change mu and lamb, which the reset of the restart then uses.

        Parameters
        ----------
        new_mu : int
            The number of parents.
        new_lambda : int
            The number of offspring per generation.
        """

    @property
    def settings(self) -> parameters.Settings:
        """The Settings of this optimizer."""

    @settings.setter
    def settings(self, arg: parameters.Settings, /) -> None: ...

    @property
    def mu(self) -> int:
        """
        The current number of parents: settings.mu0 at first, changed by IPOP and BIPOP restarts.
        """

    @mu.setter
    def mu(self, arg: int, /) -> None: ...

    @property
    def lamb(self) -> int:
        """
        The current number of offspring per generation (lambda): settings.lambda0 at first, changed by IPOP and BIPOP restarts.
        """

    @lamb.setter
    def lamb(self, arg: int, /) -> None: ...

    @property
    def adaptation(self) -> matrix_adaptation.Adaptation:
        """
        The matrix adaptation, a matrix_adaptation.Adaptation of the type chosen by modules.matrix_adaptation.

        Holds the mean m, the evolution paths and the matrix of the search distribution.
        """

    @adaptation.setter
    def adaptation(self, arg: matrix_adaptation.Adaptation, /) -> None: ...

    @property
    def criteria(self) -> restart.Criteria:
        """
        The restart criteria (restart.Criteria); its list items can be changed from Python.
        """

    @criteria.setter
    def criteria(self, arg: restart.Criteria, /) -> None: ...

    @property
    def stats(self) -> parameters.Stats:
        """The statistics of the run (Stats)."""

    @stats.setter
    def stats(self, arg: parameters.Stats, /) -> None: ...

    @property
    def weights(self) -> parameters.Weights:
        """The recombination weights and learning rates in use (Weights)."""

    @weights.setter
    def weights(self, arg: parameters.Weights, /) -> None: ...

    @property
    def pop(self) -> Population:
        """The population of the current generation."""

    @pop.setter
    def pop(self, arg: Population, /) -> None: ...

    @property
    def old_pop(self) -> Population:
        """The population of the previous generation (the parents for elitism)."""

    @old_pop.setter
    def old_pop(self, arg: Population, /) -> None: ...

    @property
    def sampler(self) -> sampling.Sampler:
        """
        The sampler of the samples z (sampling.Sampler), built from modules.sampler, sample_transformation, mirrored and orthogonal.
        """

    @sampler.setter
    def sampler(self, arg: sampling.Sampler, /) -> None: ...

    @property
    def mutation(self) -> mutation.Strategy:
        """
        The mutation (mutation.Strategy, by modules.ssa): the step size sigma and its adaptation, threshold convergence, sequential selection and sampled step sizes.
        """

    @mutation.setter
    def mutation(self, arg: mutation.Strategy, /) -> None: ...

    @property
    def selection(self) -> selection.Strategy:
        """The selection (selection.Strategy): pairwise selection and elitism."""

    @selection.setter
    def selection(self, arg: selection.Strategy, /) -> None: ...

    @property
    def restart_strategy(self) -> restart.Strategy:
        """
        The restart strategy (restart.Strategy, by modules.restart_strategy): the population size and step size of the next run.
        """

    @restart_strategy.setter
    def restart_strategy(self, arg: restart.Strategy, /) -> None: ...

    @property
    def repelling(self) -> repelling.Repelling:
        """
        The repelling restarts (repelling.Repelling, by modules.repelling_type): the archive of taboo points.
        """

    @repelling.setter
    def repelling(self, arg: repelling.Repelling, /) -> None: ...

    @property
    def bounds(self) -> bounds.BoundCorrection:
        """
        The bound correction (bounds.BoundCorrection, by modules.bound_correction).
        """

    @bounds.setter
    def bounds(self, arg: bounds.BoundCorrection, /) -> None: ...

    @property
    def center_placement(self) -> center.Placement:
        """
        The placement of the mean after a restart (center.Placement, by modules.center_placement).
        """

    @center_placement.setter
    def center_placement(self, arg: center.Placement, /) -> None: ...

    @property
    def coordinate_mapping(self) -> parameters.CoordinateMapping:
        """The rounding of the integer variables (CoordinateMapping)."""

    @coordinate_mapping.setter
    def coordinate_mapping(self, arg: parameters.CoordinateMapping, /) -> None: ...

class ModularCMAES:
    """
    The modular CMA-ES.

    Create it from Settings (or from Parameters), then run it in one of two ways:

    - with an objective function: run() (or calling the object) until a break condition holds, step() for one generation, or mutate(), select(), recombine() and adapt() for the steps of a generation;
    - ask-tell, for an objective that is evaluated outside the optimizer (in parallel, on a cluster): ask() returns the points to evaluate, tell() takes their objective values.

    step() and run() are loops over ask() and tell(), so both ways give the same results, bit for bit. The state, including the statistics and the best solution (p.stats.global_best), is in p.
    """

    @overload
    def __init__(self, parameters: Parameters) -> None:
        """
        An optimizer that works on the given Parameters (shared, not copied: p is parameters).
        """

    @overload
    def __init__(self, dimension: int) -> None:
        """
        An optimizer with the default settings for this dimension: the same as ModularCMAES(Settings(dimension)).
        """

    @overload
    def __init__(self, settings: parameters.Settings) -> None:
        """An optimizer for the given Settings (copied)."""

    @overload
    def __init__(self, fitness_func: Callable, *args, **kwargs) -> None:
        """
        The signature of modcma 1.x (the pure-Python implementation): raises TypeError.
        """

    @property
    def center_of_mass(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The center of mass, (d,) array: the mean of the search distribution as a point of the objective (integer variables rounded; inside the bounds when a bound correction is used), pycma's `xfavorite`. The current mean: during a run it moves with every generation.

        Every run ends with its evaluation, recorded in `p.stats.centers` (#111). With a noisy objective the value of `p.stats.global_best` is biased low (it was selected for being low); the mean averages over the population and, with a large population, is often closer to the optimum.
        """

    def recombine(self) -> None:
        """
        Move the mean to the weighted mean of the mu best individuals.

        m_old = m, then m = m_old + sum_k positive[k] (X[:, k] - m_old) over the first mu columns of the (sorted) population; with the utility weights of NES/xNES, over every individual. The third step of a generation, after select(). Raises RuntimeError while the points of a generation wait for their values (after ask(), or an interrupted mutate() or step()).
        """

    def mutate(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], float]) -> None:
        """
        Sample and evaluate a generation: the first step of a generation.

        Restarts first when a restart criterion holds (the run that ends evaluates its center of mass with objective). Then select(), recombine() and adapt() complete the generation. Every call samples a new generation, as in 1.x; only a generation that ask() started is completed instead (its points evaluated with objective). When the optimizer stops (stop_conditions()), it evaluates the final center of mass instead of a generation, and select(), recombine() and adapt() then do nothing: the 1.x loop `while not es.break_conditions(): es.mutate(f); es.select(); ...` ends as run() does.

        Parameters
        ----------
        objective : callable
            The objective function, called with a (d,) array, returns a float.
        """

    def select(self) -> None:
        """
        Select the individuals of the evaluated generation: the second step.

        Applies pairwise selection (mirrored PAIRWISE) and elitism (the parents join the offspring), sorts the population by f, keeps lambda individuals and updates the statistics (current_avg, current_best, global_best). Raises RuntimeError while the points of a generation wait for their values (after ask(), or an interrupted mutate() or step()).
        """

    def adapt(self) -> None:
        """
        Adapt the search distribution: the last step of a generation.

        Updates the evolution paths, the step size, the matrix and the restart criteria, and increments p.stats.t (Parameters.adapt). Raises RuntimeError while the points of a generation wait for their values (after ask(), or an interrupted mutate() or step()).
        """

    def step(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], Any], *, vectorized: bool = False) -> bool:
        """
        Run one generation (with a restart first, when one is due).

        When the optimizer stops (stop_conditions()), the call evaluates the final center of mass instead, one evaluation, and returns False.

        Parameters
        ----------
        objective : callable
            The objective function. Called with a (d,) array, it returns a float; with vectorized=True it is called with every batch of points as a (k, d) array and returns k values.
        vectorized : bool, default False
            Whether objective evaluates a batch of points in one call.

        Returns
        -------
        bool
            False when the optimizer has stopped (see break_conditions), True otherwise.
        """

    def __call__(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], Any], *, vectorized: bool = False) -> None:
        """Run until a break condition holds: the same as run()."""

    def run(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], Any], *, vectorized: bool = False) -> None:
        """
        Run generations while no break condition holds.

        The break conditions are checked before every generation, the first one too: the run never exceeds the budget, and an optimizer that has stopped does not run again. Every run ends with the evaluation of its center of mass, the last one too (#111). The result is in p.stats (global_best, evaluations, centers). A vectorized objective saves a Python call per evaluation.

        Parameters
        ----------
        objective : callable
            The objective function. Called with a (d,) array, it returns a float; with vectorized=True it is called with every batch of points as a (k, d) array and returns k values.
        vectorized : bool, default False
            Whether objective evaluates a batch of points in one call.
        """

    def ask(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C')]:
        """
        The points to evaluate next, a new (k, d) array (row i is point i).

        Starts a generation when none is in progress; until tell(), it returns the same points. k >= 1 is the largest batch that can be evaluated in any order: usually lambda (lambda + 2 with TPA step-size adaptation). With sequential selection, the points up to the first one after which the generation may stop, then one point at a time (a pair with pairwise mirroring). Every run ends with its center of mass, one point: a restart asks for it first and, with repelling restarts, for the points of the hill-valley tests, one at a time; when the optimizer stops (stop_conditions()), the last ask() returns it, and break_conditions() holds after its tell().
        """

    def tell(self, f: numpy.typing.ArrayLike, X: numpy.typing.ArrayLike | None = None) -> bool:
        """
        Give the objective values of the points of ask().

        Completing a generation runs select(), recombine() and adapt(); then break_conditions() tells whether to stop.

        Parameters
        ----------
        f : array_like
            (k,) the objective values of the k points of ask(), in the same order.
        X : array_like, optional
            (k, d), or (d,) for one point: the points that were evaluated, when they differ from the asked ones (repaired or injected solutions). A row that differs from the asked point replaces it; for an individual of the population, its y and z are derived from the row with its step size, as for elitism.

        Returns
        -------
        bool
            True when this completed a generation (or the final center of mass, when the optimizer stops), False when ask() has more points of this generation.

        Raises
        ------
        RuntimeError
            No points are pending (call ask() first).
        ValueError
            f or X does not match the points of ask().
        """

    @property
    def has_pending(self) -> bool:
        """Whether ask() has points that wait for tell()."""

    def break_conditions(self) -> bool:
        """
        Whether the optimizer has stopped; meaningful between generations (after tell() returned True).

        True when stop_conditions() hold and the final center of mass has been evaluated (the last ask() and tell(), or step()). False again when the stop conditions no longer hold, after the budget was raised: the run continues.
        """

    def stop_conditions(self) -> bool:
        """
        Whether the optimizer should stop: then the next ask() (or step()) evaluates the final center of mass, and break_conditions() holds after it.

        True when one of these holds: p.stats.t reached settings.max_generations; the best objective value is at most settings.target; the next generation (lambda points, and the 2 extra points of TPA) would leave no evaluation of settings.budget for the final center of mass; a restart is due, but its center, the smallest first generation of the new run and its center of mass would exceed settings.budget; or restart_strategy is STOP and a restart criterion holds.
        """

    @property
    def p(self) -> Parameters:
        """
        The Parameters: the state of the optimizer (settings, modules, population, statistics).
        """
