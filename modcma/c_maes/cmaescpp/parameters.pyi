"""
The configuration and the bookkeeping of the optimizer.

Modules (the module choices) and Settings (dimension, budget, bounds, strategy parameters) configure a run; Weights holds the recombination weights and learning rates, Stats and Solution the progress of a run, CoordinateMapping the rounding of integer variables. The Parameters class itself is modcma.c_maes.Parameters.
"""

from collections.abc import Sequence
from typing import Annotated

import numpy
import numpy.typing
from numpy.typing import NDArray

import modcma.c_maes.cmaescpp.options


class Modules:
    """
    The module configuration: one choice per module.

    A bool attribute turns a module on or off; an enum attribute selects one option of an enum in modcma.c_maes.options. The defaults, Modules(), give the standard CMA-ES with cumulative step-size adaptation (CSA) and no restarts. Settings may adjust some choices so that they fit together (see Settings).
    """

    def __init__(self) -> None:
        """The default configuration: CMA-ES with CSA, every bool module off."""

    @property
    def elitist(self) -> bool:
        """
        bool, default False: (mu + lambda) selection.

        The parents of the previous generation compete with the offspring. Can speed up convergence on unimodal functions, at the cost of diversity.
        """

    @elitist.setter
    def elitist(self, arg: object, /) -> None: ...

    @property
    def active(self) -> bool:
        """
        bool, default False: active covariance matrix update.

        The worst individuals enter the update of the matrix with negative weights (Jastrebski and Arnold). Settings turns it off for matrix_adaptation NONE and for a (1+1)-ES.
        """

    @active.setter
    def active(self, arg: object, /) -> None: ...

    @property
    def orthogonal(self) -> bool:
        """
        bool, default False: orthogonal sampling.

        The samples of a generation are made orthogonal to each other (by a QR decomposition), keeping their lengths (Wang et al.).
        """

    @orthogonal.setter
    def orthogonal(self, arg: object, /) -> None: ...

    @property
    def sequential_selection(self) -> bool:
        """
        bool, default False: sequential selection (Brockhoff et al.).

        The evaluation of a generation stops at the first individual, from index mu on, that improves on the best of the previous generation; the rest of the generation is not evaluated.
        """

    @sequential_selection.setter
    def sequential_selection(self, arg: object, /) -> None: ...

    @property
    def threshold_convergence(self) -> bool:
        """
        bool, default False: threshold convergence (Piad-Morffis et al.).

        A sample z shorter than the threshold t is lengthened to 2 t - ||z||. t = init_threshold E||z|| ((budget - evaluations) / budget)^decay_factor (`ThresholdConvergence`), computed once per generation: it decreases as the budget is used, which prolongs the exploration phase.
        """

    @threshold_convergence.setter
    def threshold_convergence(self, arg: object, /) -> None: ...

    @property
    def sample_sigma(self) -> bool:
        """
        bool, default False: every individual samples its own step sizes.

        The step sizes are drawn from a log-normal distribution around sigma, with learning rate Weights.beta (self-adaptation).
        """

    @sample_sigma.setter
    def sample_sigma(self, arg: object, /) -> None: ...

    @property
    def weights(self) -> modcma.c_maes.cmaescpp.options.RecombinationWeights:
        """
        options.RecombinationWeights, default DEFAULT: the recombination weights.

        See modcma.c_maes.options.RecombinationWeights for the choices.
        """

    @weights.setter
    def weights(self, arg: modcma.c_maes.cmaescpp.options.RecombinationWeights, /) -> None: ...

    @property
    def sampler(self) -> modcma.c_maes.cmaescpp.options.BaseSampler:
        """
        options.BaseSampler, default UNIFORM: the generator of the base points in [0, 1)^d.

        Pseudo-random or a low-discrepancy sequence; sample_transformation turns the points into samples of the search distribution. See modcma.c_maes.options.BaseSampler.
        """

    @sampler.setter
    def sampler(self, arg: modcma.c_maes.cmaescpp.options.BaseSampler, /) -> None: ...

    @property
    def mirrored(self) -> modcma.c_maes.cmaescpp.options.Mirror:
        """
        options.Mirror, default NONE: mirrored sampling.

        Every second sample is the mirror image of the one before; PAIRWISE also selects only the better point of each pair. See modcma.c_maes.options.Mirror.
        """

    @mirrored.setter
    def mirrored(self, arg: modcma.c_maes.cmaescpp.options.Mirror, /) -> None: ...

    @property
    def ssa(self) -> modcma.c_maes.cmaescpp.options.StepSizeAdaptation:
        """
        options.StepSizeAdaptation, default CSA: the step-size adaptation.

        See modcma.c_maes.options.StepSizeAdaptation for the choices.
        """

    @ssa.setter
    def ssa(self, arg: modcma.c_maes.cmaescpp.options.StepSizeAdaptation, /) -> None: ...

    @property
    def bound_correction(self) -> modcma.c_maes.cmaescpp.options.CorrectionMethod:
        """
        options.CorrectionMethod, default NONE: the correction of points outside the bounds lb and ub.

        See modcma.c_maes.options.CorrectionMethod for the choices.
        """

    @bound_correction.setter
    def bound_correction(self, arg: modcma.c_maes.cmaescpp.options.CorrectionMethod, /) -> None: ...

    @property
    def restart_strategy(self) -> modcma.c_maes.cmaescpp.options.RestartStrategy:
        """
        options.RestartStrategy, default NONE: what happens when a restart criterion holds.

        Restart (with the same, an increasing or an alternating population size) or stop. See modcma.c_maes.options.RestartStrategy.
        """

    @restart_strategy.setter
    def restart_strategy(self, arg: modcma.c_maes.cmaescpp.options.RestartStrategy, /) -> None: ...

    @property
    def repelling_type(self) -> modcma.c_maes.cmaescpp.options.RepellingType:
        """
        options.RepellingType, default NONE: repelling restarts.

        Restarts avoid the regions of the optima found by earlier runs (taboo points). See modcma.c_maes.options.RepellingType.
        """

    @repelling_type.setter
    def repelling_type(self, arg: modcma.c_maes.cmaescpp.options.RepellingType, /) -> None: ...

    @property
    def repelling_metric(self) -> modcma.c_maes.cmaescpp.options.RepellingMetric:
        """
        options.RepellingMetric, default EUCLIDEAN: the distance between a sample and a taboo point of repelling restarts.

        See modcma.c_maes.options.RepellingMetric for the choices.
        """

    @repelling_metric.setter
    def repelling_metric(self, arg: modcma.c_maes.cmaescpp.options.RepellingMetric, /) -> None: ...

    @property
    def matrix_adaptation(self) -> modcma.c_maes.cmaescpp.options.MatrixAdaptationType:
        """
        options.MatrixAdaptationType, default COVARIANCE: the adaptation of the shape of the search distribution.

        Settings replaces COVARIANCE by COVARIANCE_NO_EIGV when ssa is not CSA (unless always_compute_eigv). See modcma.c_maes.options.MatrixAdaptationType.
        """

    @matrix_adaptation.setter
    def matrix_adaptation(self, arg: modcma.c_maes.cmaescpp.options.MatrixAdaptationType, /) -> None: ...

    @property
    def center_placement(self) -> modcma.c_maes.cmaescpp.options.CenterPlacement:
        """
        options.CenterPlacement, default X0: where the mean of the search distribution starts after a restart.

        See modcma.c_maes.options.CenterPlacement for the choices.
        """

    @center_placement.setter
    def center_placement(self, arg: modcma.c_maes.cmaescpp.options.CenterPlacement, /) -> None: ...

    @property
    def sample_transformation(self) -> modcma.c_maes.cmaescpp.options.SampleTranformerType:
        """
        options.SampleTranformerType, default GAUSSIAN: the distribution of the samples.

        Transforms the base points of the sampler into samples of this distribution. See modcma.c_maes.options.SampleTranformerType.
        """

    @sample_transformation.setter
    def sample_transformation(self, arg: modcma.c_maes.cmaescpp.options.SampleTranformerType, /) -> None: ...

    def __repr__(self) -> str:
        """The module choices as text."""

class Solution:
    """A point with its objective value and when it was found."""

    def __init__(self) -> None:
        """An empty solution: x is empty, y is +inf, t and e are 0."""

    @property
    def x(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """(d,) array: the point, as evaluated (integer variables rounded)."""

    @x.setter
    def x(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def y(self) -> float:
        """The objective value of x; +inf for an empty solution."""

    @y.setter
    def y(self, arg: float, /) -> None: ...

    @property
    def t(self) -> int:
        """The generation in which x was found."""

    @t.setter
    def t(self, arg: int, /) -> None: ...

    @property
    def e(self) -> int:
        """The number of evaluations when x was found."""

    @e.setter
    def e(self, arg: int, /) -> None: ...

    def __repr__(self) -> str:
        """x and y as text."""

class Stats:
    """The progress of a run: counters and the best solutions."""

    def __init__(self) -> None:
        """Empty statistics: no generation, no evaluation, no solution."""

    @property
    def t(self) -> int:
        """
        The generation counter, incremented at the end of every generation; restarts do not reset it.
        """

    @t.setter
    def t(self, arg: int, /) -> None: ...

    @property
    def t_run(self) -> int:
        """
        The generations of the current run: reset at a restart. h_sigma's bias correction counts these.
        """

    @t_run.setter
    def t_run(self, arg: int, /) -> None: ...

    @property
    def evaluations(self) -> int:
        """
        The number of objective function evaluations so far.

        Includes the extra points of TPA, the centers of restarts and the hill-valley tests of repelling restarts.
        """

    @evaluations.setter
    def evaluations(self, arg: int, /) -> None: ...

    @property
    def current_avg(self) -> float:
        """
        The mean objective value of the population after the last selection (+inf before the first).
        """

    @current_avg.setter
    def current_avg(self, arg: float, /) -> None: ...

    @property
    def solutions(self) -> list[Solution]:
        """
        list of Solution: the best solution of every finished run, appended at each restart (a copy).
        """

    @solutions.setter
    def solutions(self, arg: Sequence[Solution], /) -> None: ...

    @property
    def centers(self) -> list[Solution]:
        """list of Solution: the center evaluated at each restart (a copy)."""

    @centers.setter
    def centers(self, arg: Sequence[Solution], /) -> None: ...

    @property
    def current_best(self) -> Solution:
        """The best solution of the current run (since the last restart)."""

    @current_best.setter
    def current_best(self, arg: Solution, /) -> None: ...

    @property
    def global_best(self) -> Solution:
        """The best solution found so far, over all runs."""

    @global_best.setter
    def global_best(self, arg: Solution, /) -> None: ...

    @property
    def has_improved(self) -> bool:
        """Whether the last selection improved current_best."""

    @has_improved.setter
    def has_improved(self, arg: bool, /) -> None: ...

    @property
    def success_ratio(self) -> float:
        """
        The smoothed rate of successful generations.

        A generation is a success when its best individual is better than the best of the run so far (with elitism, the parent); counted once per generation. An exponential moving average with weight 1/12, starting at 2/11. Used by the success rule step-size adaptation (SR) and by the update of the evolution path in the (1+1) configuration (lambda0 = 1).
        """

    @success_ratio.setter
    def success_ratio(self, arg: float, /) -> None: ...

    @property
    def last_update(self) -> int:
        """
        The generation of the last update of the matrix of the matrix adaptation.
        """

    @last_update.setter
    def last_update(self, arg: int, /) -> None: ...

    @property
    def n_updates(self) -> int:
        """The number of updates of the matrix of the matrix adaptation so far."""

    @n_updates.setter
    def n_updates(self, arg: int, /) -> None: ...

    def __repr__(self) -> str:
        """
        The generation, the evaluations, the best solution and has_improved as text.
        """

class Weights:
    """
    The recombination weights and the learning rates of a run.

    Computed when the Parameters are created, and again at every restart with the current mu and lambda. A learning rate given in the Settings is used as is; the defaults below use d for the dimension.
    """

    def __init__(self, Settings: Settings, expected_length_z: float) -> None:
        """
        The weights and learning rates for settings.mu0 and settings.lambda0.

        Parameters
        ----------
        Settings : Settings
            The settings: dimension, modules and the learning rates given there.
        expected_length_z : float
            The expected length of a sample of the sampler, E||z|| (Parameters pass the sampler's expected_length()).
        """

    @property
    def mueff(self) -> float:
        """
        The variance effective selection mass of the positive weights, 1 / sum(positive**2).
        """

    @mueff.setter
    def mueff(self, arg: float, /) -> None: ...

    @property
    def mueff_mean(self) -> float:
        """
        The variance effective selection mass of the weights of the mean, 1 / sum(w**2): `mueff`, except for UTILITY, where the mean uses all lambda utilities. Normalizes the evolution paths.
        """

    @mueff_mean.setter
    def mueff_mean(self, arg: float, /) -> None: ...

    @property
    def utility(self) -> bool:
        """
        Whether the weights are the NES utilities (`RecombinationWeights.UTILITY`).
        """

    @utility.setter
    def utility(self, arg: bool, /) -> None: ...

    @property
    def pairwise(self) -> bool:
        """
        Whether the selection is pairwise (`Mirror.PAIRWISE`): the losers of the mirrored pairs (f = +inf, sorted last) then get no negative weights in the active update, as in pycma.
        """

    @pairwise.setter
    def pairwise(self, arg: bool, /) -> None: ...

    @property
    def mueff_neg(self) -> float:
        """
        The variance effective selection mass of the negative weights, sum(negative)**2 / sum(negative**2).
        """

    @mueff_neg.setter
    def mueff_neg(self, arg: float, /) -> None: ...

    @property
    def c1(self) -> float:
        """
        The learning rate of the rank-one update of the matrix.

        Default acov / ((d + 1.3)**2 + mueff); 2 / (d**2 + 6) for a (1+1)-ES (lambda0 = 1).
        """

    @c1.setter
    def c1(self, arg: float, /) -> None: ...

    @property
    def cmu(self) -> float:
        """
        The learning rate of the rank-mu update of the matrix.

        Default min(1 - c1, acov (mueff - 2 + 1 / mueff) / ((d + 2)**2 + acov mueff / 2)), multiplied by (d + 2) / 3 for SEPARABLE; 2 / (d**2 + 6) for lambda0 = 1.
        """

    @cmu.setter
    def cmu(self, arg: float, /) -> None: ...

    @property
    def cc(self) -> float:
        """
        The learning rate of the evolution path pc.

        Default (4 + mueff / d) / (d + 4 + 2 mueff / d); 2 / (d + 2) for a (1+1)-ES; with NATURAL_GRADIENT cs / 2, or 1 / (4 d**1.5) for a (1+1)-ES.
        """

    @cc.setter
    def cc(self, arg: float, /) -> None: ...

    @property
    def cs(self) -> float:
        """
        The learning rate of the step-size adaptation.

        Default by ssa: CSA (mueff + 2) / (d + mueff + 3); SR 2 / (d + 2); XNES 0.01; MXNES 1; LPXNES 9 mueff / (10 sqrt(d)); otherwise (TPA, MSR, PSR, SA) 0.3. With matrix_adaptation NATURAL_GRADIENT (any ssa): (9 + 3 ln(d)) / (5 d sqrt(d)).
        """

    @cs.setter
    def cs(self, arg: float, /) -> None: ...

    @property
    def damps(self) -> float:
        """
        The damping of the step-size update.

        Default for CSA 1 + 2 max(0, sqrt((mueff - 1) / (d + 1)) - 1) + cs; for SR 1 + d / 2; for MSR max(1, 2 - 2 / d); otherwise (PSR among them) 1.
        """

    @damps.setter
    def damps(self, arg: float, /) -> None: ...

    @property
    def sqrt_cc_mueff(self) -> float:
        """
        sqrt(cc (2 - cc) mueff): the normalisation of the update of the path pc.
        """

    @sqrt_cc_mueff.setter
    def sqrt_cc_mueff(self, arg: float, /) -> None: ...

    @property
    def sqrt_cs_mueff(self) -> float:
        """
        sqrt(cs (2 - cs) mueff): the normalisation of the update of the path ps.
        """

    @sqrt_cs_mueff.setter
    def sqrt_cs_mueff(self, arg: float, /) -> None: ...

    @property
    def lazy_update_interval(self) -> float:
        """
        The number of generations between two eigendecompositions of the covariance matrix (COVARIANCE), 1 / (c1 + cmu) / d / 10; the matrix itself is updated every generation.

        Below 1, the matrix is decomposed every generation.
        """

    @lazy_update_interval.setter
    def lazy_update_interval(self, arg: float, /) -> None: ...

    @property
    def expected_length_z(self) -> float:
        """The expected length of a sample of the sampler, E||z||."""

    @expected_length_z.setter
    def expected_length_z(self, arg: float, /) -> None: ...

    @property
    def expected_length_ps(self) -> float:
        """
        (1.4 + 2 / (d + 1)) expected_length_z: the threshold on the length of ps above which the update of the path pc stalls (h_sigma).
        """

    @expected_length_ps.setter
    def expected_length_ps(self, arg: float, /) -> None: ...

    @property
    def beta(self) -> float:
        """
        The learning rate of the sampled step sizes (sample_sigma).

        1 / sqrt(2 mueff); log(2) / (sqrt(d) log(d)) with LPXNES.
        """

    @beta.setter
    def beta(self, arg: float, /) -> None: ...

    @property
    def weights(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """(lambda,) array: all weights by rank, positive followed by negative."""

    @weights.setter
    def weights(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def positive(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (mu,) array: the weights of the mu best individuals, normalised to sum 1.
        """

    @positive.setter
    def positive(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def negative(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (lambda - mu,) array: the weights of the other individuals, for the active update.

        Scaled so that their absolute values sum to min(1 + c1 / cmu, 1 + 2 mueff_neg / (mueff + 2), (1 - c1 - cmu) / (d cmu)) (Hansen 2016).
        """

    @negative.setter
    def negative(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    def for_population(self, n: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The weights for a population of n evaluated individuals.

        After sequential selection stopped a generation early (mu < n < lambda), the positive weights are unchanged and the negative weights of ranks mu..n-1 are scaled so that their sum equals the sum of all lambda - mu negative weights. For n >= lambda, these are the weights themselves.

        Parameters
        ----------
        n : int
            The number of evaluated individuals.

        Returns
        -------
        numpy.ndarray
            (min(n, lambda),) the weights.
        """

    def __repr__(self) -> str:
        """The selection masses, learning rates and weights as text."""

class Settings:
    """
    The settings of a run: dimension, modules, stopping conditions, search space and strategy parameters.

    A strategy parameter left None gets its default when the Parameters are created (see Weights). The constructor computes the derived values (db, center, diameter, volume, has_bounds) and adjusts the modules; setting lb or ub afterwards updates the derived values, changing the modules afterwards does not adjust them.
    """

    def __init__(self, dim: int, modules: Modules | None = None, *, target: float | None = None, max_generations: int | None = None, budget: int | None = None, sigma0: float | None = None, mu0: int | None = None, lambda0: int | None = None, x0: numpy.typing.ArrayLike | None = None, lb: numpy.typing.ArrayLike | None = None, ub: numpy.typing.ArrayLike | None = None, integer_variables: numpy.typing.ArrayLike | None = None, cs: float | None = None, cc: float | None = None, cmu: float | None = None, c1: float | None = None, damps: float | None = None, acov: float | None = None, verbose: bool = False, always_compute_eigv: bool = False, seed: int | None = None, fast_sampling: bool = False) -> None:
        """
        Settings for a problem of dimension dim.

        dim and modules are positional, every other argument is keyword-only. None means a default computed from the dimension and the modules.

        The constructor makes the configuration consistent:

        - with mirrored PAIRWISE, an odd lambda0 is increased by one;
        - a mu0 larger than lambda0 becomes lambda0 // 2;
        - with ssa SA or matrix_adaptation CMSA, mu0 is the given value or lambda0 // 4 (this replaces the previous rule);
        - when ssa is not CSA, matrix_adaptation COVARIANCE becomes COVARIANCE_NO_EIGV (unless always_compute_eigv);
        - with matrix_adaptation NONE, active is turned off;
        - lambda0 = 1 gives a (1+1)-ES (one_plus_one): mu0 = 1, elitist, ssa SR, EQUAL weights, no active update, no sequential selection, and IPOP or BIPOP become RESTART;
        - when the box lb..ub is not finite, has_bounds is False, diameter 10, volume 10**dim and center the zero vector.

        Parameters
        ----------
        dim : int
            The dimension of the search space.
        modules : Modules, optional
            The module configuration (copied). Default Modules(): CMA-ES with CSA.
        target : float, optional
            Stop when the best objective value is at most target. Default None: no target.
        max_generations : int, optional
            Stop after this many generations. Default None: no limit.
        budget : int, optional
            The maximum number of evaluations, default 10000 * dim. A generation runs only when it fits: the run stops when the evaluations so far plus lambda (and TPA's 2 points) would exceed it.
        sigma0 : float, optional
            The initial step size, default 2.0; also the step size after a restart (the small-population runs of BIPOP use sigma0 * 10**(-2 u), u uniform in [0, 1)).
        mu0 : int, optional
            The number of parents, default lambda0 // 2 (lambda0 // 4 with ssa SA or matrix_adaptation CMSA).
        lambda0 : int, optional
            The number of offspring per generation, default 4 + floor(3 ln(dim)).
        x0 : array_like, optional
            (dim,) the initial mean of the search distribution. Default None: the zero vector; after a restart, center_placement X0 uses x0 or, without x0, the center of the bounds.
        lb : array_like, optional
            (dim,) the lower bounds, default -5 in every coordinate.
        ub : array_like, optional
            (dim,) the upper bounds, default 5 in every coordinate.
        integer_variables : sequence of int, optional
            The indices of the variables that take integer values: they are rounded before evaluation and their step size has a lower bound. Default: none.
        cs : float, optional
            The learning rate of the step-size adaptation. Default depends on ssa (see Weights.cs); for CSA (mueff + 2) / (dim + mueff + 3).
        cc : float, optional
            The learning rate of the evolution path pc. Default (4 + mueff / dim) / (dim + 4 + 2 mueff / dim) (see Weights.cc).
        cmu : float, optional
            The learning rate of the rank-mu update. Default min(1 - c1, acov (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + acov mueff / 2)) (see Weights.cmu).
        c1 : float, optional
            The learning rate of the rank-one update. Default acov / ((dim + 1.3)**2 + mueff) (see Weights.c1).
        damps : float, optional
            The damping of the step-size update. Default for CSA 1 + 2 max(0, sqrt((mueff - 1) / (dim + 1)) - 1) + cs (see Weights.damps).
        acov : float, optional
            The factor alpha_cov in the defaults of c1 and cmu, default 2.
        verbose : bool, default False
            Print a message at every restart and when a decomposition of the matrix fails.
        always_compute_eigv : bool, default False
            Keep matrix_adaptation COVARIANCE, with its eigendecomposition, when ssa is not CSA.
        seed : int, optional
            The seed of the optimizer's own random generator. Default None: the optimizer starts from a copy of the global generator (set_seed) when it is created.
        fast_sampling : bool, default False
            Sample with xoshiro256++ and a ziggurat instead of mt19937 and the inverse normal CDF (the default UNIFORM + GAUSSIAN sampler only): faster, but a different random stream.
        """

    @property
    def dim(self) -> int:
        """The dimension of the search space (read-only)."""

    @property
    def seed(self) -> int | None:
        """
        The seed of the optimizer's own random generator, or None.

        None: the optimizer starts from a copy of the global generator (set_seed) when it is created. Set it before the optimizer is created.
        """

    @seed.setter
    def seed(self, arg: int | None, /) -> None: ...

    @property
    def fast_sampling(self) -> bool:
        """
        Opt-in: sample with xoshiro256++ and a ziggurat, default False.

        Instead of mt19937 and the inverse normal CDF (the default UNIFORM + GAUSSIAN sampler only). Faster, but a different random stream. Set it before the optimizer is created.
        """

    @fast_sampling.setter
    def fast_sampling(self, arg: bool, /) -> None: ...

    @property
    def modules(self) -> Modules:
        """
        The module configuration, after the adjustments of the constructor.

        The attribute is read-only. Its fields can still be changed in place, but the adjustments of the constructor are not applied again, and an optimizer that exists keeps the modules it was created with.
        """

    @property
    def target(self) -> float | None:
        """
        Stop when the best objective value is at most target; None (default): no target.
        """

    @target.setter
    def target(self, arg: float | None, /) -> None: ...

    @property
    def max_generations(self) -> int | None:
        """Stop after this many generations; None (default): no limit."""

    @max_generations.setter
    def max_generations(self, arg: int | None, /) -> None: ...

    @property
    def budget(self) -> int:
        """
        The maximum number of evaluations, default 10000 * dim.

        A generation runs only when it fits: the run stops when the evaluations so far plus lambda (and TPA's 2 points) would exceed it.
        """

    @budget.setter
    def budget(self, arg: int, /) -> None: ...

    @property
    def sigma0(self) -> float:
        """
        The initial step size, default 2.0; also the step size after a restart (smaller for the small-population runs of BIPOP).
        """

    @sigma0.setter
    def sigma0(self, arg: float, /) -> None: ...

    @property
    def mu0(self) -> int:
        """
        The initial number of parents, default lambda0 // 2.

        Parameters.mu is the current value, which IPOP and BIPOP restarts change.
        """

    @mu0.setter
    def mu0(self, arg: int, /) -> None: ...

    @property
    def lambda0(self) -> int:
        """
        The initial number of offspring per generation, default 4 + floor(3 ln(dim)).

        Parameters.lamb is the current value, which IPOP and BIPOP restarts change.
        """

    @lambda0.setter
    def lambda0(self, arg: int, /) -> None: ...

    @property
    def x0(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None:
        """
        (dim,) array: the initial mean of the search distribution, or None (default) for the zero vector.
        """

    @x0.setter
    def x0(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None, /) -> None: ...

    @property
    def lb(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (dim,) array: the lower bounds, default -5. Setting it updates `db`, `center`, `diameter`, `volume` and `has_bounds`.
        """

    @lb.setter
    def lb(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def ub(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (dim,) array: the upper bounds, default 5. Setting it updates `db`, `center`, `diameter`, `volume` and `has_bounds`.
        """

    @ub.setter
    def ub(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def db(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """(dim,) array: the widths of the box, ub - lb."""

    @db.setter
    def db(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def center(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (dim,) array: the center of the box, lb + (ub - lb) / 2; the zero vector when the box is not finite.
        """

    @center.setter
    def center(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def diameter(self) -> float:
        """
        The length of the diagonal of the box, ||ub - lb||; 10 when the box is not finite (then volume is 10^dim).
        """

    @diameter.setter
    def diameter(self, arg: float, /) -> None: ...

    @property
    def volume(self) -> float:
        """
        The volume of the box, prod(ub - lb); 10**dim when the box is not finite.
        """

    @property
    def has_bounds(self) -> bool:
        """
        Whether the box lb..ub is finite; bound correction only acts when it is.
        """

    @has_bounds.setter
    def has_bounds(self, arg: object, /) -> None: ...

    @property
    def integer_variables(self) -> Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')]:
        """(k,) int array: the indices of the integer variables (read-only)."""

    @property
    def cs(self) -> float | None:
        """
        The learning rate cs given to the constructor, or None for the default.

        The value in use is Parameters.weights.cs.
        """

    @cs.setter
    def cs(self, arg: float | None, /) -> None: ...

    @property
    def cc(self) -> float | None:
        """
        The learning rate cc given to the constructor, or None for the default.

        The value in use is Parameters.weights.cc.
        """

    @cc.setter
    def cc(self, arg: float | None, /) -> None: ...

    @property
    def cmu(self) -> float | None:
        """
        The learning rate cmu given to the constructor, or None for the default.

        The value in use is Parameters.weights.cmu.
        """

    @cmu.setter
    def cmu(self, arg: float | None, /) -> None: ...

    @property
    def c1(self) -> float | None:
        """
        The learning rate c1 given to the constructor, or None for the default.

        The value in use is Parameters.weights.c1.
        """

    @c1.setter
    def c1(self, arg: float | None, /) -> None: ...

    @property
    def damps(self) -> float | None:
        """
        The damping given to the constructor, or None for the default.

        The value in use is Parameters.weights.damps.
        """

    @damps.setter
    def damps(self, arg: float | None, /) -> None: ...

    @property
    def acov(self) -> float | None:
        """
        The factor alpha_cov of the defaults of c1 and cmu, or None for the default 2.
        """

    @acov.setter
    def acov(self, arg: float | None, /) -> None: ...

    @property
    def verbose(self) -> bool:
        """
        Print a message at every restart and when a decomposition of the matrix fails.
        """

    @verbose.setter
    def verbose(self, arg: object, /) -> None: ...

    @property
    def one_plus_one(self) -> bool:
        """Whether lambda0 is 1: a (1+1)-ES (read-only; see the constructor)."""

    def __repr__(self) -> str:
        """The settings as text."""

class CoordinateMapping:
    """
    Maps a sample to the point that is evaluated: the integer variables are rounded.

    Rounding is to the nearest integer, halfway cases away from zero.
    """

    def __init__(self, integer_variables: Sequence[int], d: int) -> None:
        """
        A mapping that rounds the given variables.

        Parameters
        ----------
        integer_variables : list of int
            The indices of the integer variables, each in 0..d-1 (otherwise ValueError).
        d : int
            The dimension.
        """

    def transform(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """A copy of the point x, (d,), with the integer variables rounded."""
