"""
The configuration and the bookkeeping of the optimizer.

Modules (the module choices) and Settings (dimension, budget, bounds, strategy parameters) configure a run; Weights holds the recombination weights and learning rates, Stats and Solution the progress of a run, Center the record of a finished run, CoordinateMapping the rounding of integer variables. The Parameters class itself is modcma.c_maes.Parameters.
"""

from collections.abc import Sequence
from typing import Annotated

import numpy
import numpy.typing
from numpy.typing import NDArray

from modcma.c_maes import preset as preset
import modcma.c_maes.cmaescpp.matrix_adaptation
import modcma.c_maes.cmaescpp.options


class Modules:
    """
    The module configuration: one choice per module.

    A bool attribute turns a module on or off; an enum attribute selects one option of an enum in modcma.c_maes.options. The defaults, Modules(), give the standard CMA-ES with cumulative step-size adaptation (CSA) and no restart strategy. Settings may adjust some choices so that they fit together (see Settings).
    """

    def __init__(self) -> None:
        """The default configuration: CMA-ES with CSA, every bool module off."""

    @property
    def elitist(self) -> bool:
        """
        bool, default False: (mu + lambda) selection.

        The parents of the previous generation compete with the offspring. Can speed up convergence on unimodal functions, at the cost of diversity. Settings turns it on for a (1+1)-ES (lambda0 = 1).
        """

    @elitist.setter
    def elitist(self, arg: object, /) -> None: ...

    @property
    def active(self) -> bool:
        """
        bool, default False: active covariance matrix update.

        The worst individuals enter the update of the matrix with negative weights (Jastrebski and Arnold), rescaled by d / ||A^-1 y||^2 as in Hansen (2016), Eq. (46). With PAIRWISE the losers of the pairs get no negative weights; with elitism they go to the worst points of the (mu + lambda) pool (Population.n_pool).

        Settings turns it on for weights UTILITY, and off for matrix_adaptation NONE and CMSA (which has no active variant) and for a (1+1)-ES; these win. With NATURAL_GRADIENT, active turns weights DEFAULT into UTILITY, xNES's active form.

        With StepSizeAdaptation.XNES (and weights other than UTILITY) a generation with h_sigma = 0 has no negative weights (COVARIANCE, COVARIANCE_NO_EIGV, SEPARABLE, CHOLESKY, MATRIX): xNES SA keeps sigma too small, and there the negative weights undid the growth of C towards the optimum (#175; modcma's rule, not the tutorial's or pycma's).
        """

    @active.setter
    def active(self, arg: object, /) -> None: ...

    @property
    def orthogonal(self) -> bool:
        """
        bool, default False: orthogonal sampling.

        The samples of a generation are made orthogonal to each other (by a QR decomposition), keeping their lengths (Wang et al.): in blocks of at most d, the first block starting with the generation. With mirrored sampling the first sample of each pair. See `sampling.Orthogonal`.
        """

    @orthogonal.setter
    def orthogonal(self, arg: object, /) -> None: ...

    @property
    def sequential_selection(self) -> bool:
        """
        bool, default False: sequential selection (Brockhoff et al.).

        The evaluation of a generation stops at the first individual, from index mu on (0-based), that improves on the best of the previous generation; the rest of the generation is not evaluated. With PAIRWISE from index 2 mu on, and only after a complete pair. Not in the first generation of a run, which has no previous one. Settings turns it off for a (1+1)-ES.
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
        bool, default False: every individual samples its own step size.

        One step size per individual, the same in every coordinate, drawn from a log-normal distribution around sigma, sigma exp(beta n) with n ~ N(0, 1) and the learning rate beta = Weights.beta (self-adaptation). Always on with ssa SA and LPXNES, also when this attribute is False.
        """

    @sample_sigma.setter
    def sample_sigma(self, arg: object, /) -> None: ...

    @property
    def weights(self) -> modcma.c_maes.cmaescpp.options.RecombinationWeights:
        """
        options.RecombinationWeights, default DEFAULT: the recombination weights.

        See modcma.c_maes.options.RecombinationWeights for the choices. Settings sets EQUAL for a (1+1)-ES, and UTILITY for NATURAL_GRADIENT with active and DEFAULT weights.
        """

    @weights.setter
    def weights(self, arg: modcma.c_maes.cmaescpp.options.RecombinationWeights, /) -> None: ...

    @property
    def sampler(self) -> modcma.c_maes.cmaescpp.options.BaseSampler:
        """
        options.BaseSampler, default UNIFORM: the generator of the base points in (0, 1)^d.

        Pseudo-random or a low-discrepancy sequence; sample_transformation turns the points into samples of the search distribution. See modcma.c_maes.options.BaseSampler.
        """

    @sampler.setter
    def sampler(self, arg: modcma.c_maes.cmaescpp.options.BaseSampler, /) -> None: ...

    @property
    def mirrored(self) -> modcma.c_maes.cmaescpp.options.Mirror:
        """
        options.Mirror, default NONE: mirrored sampling.

        Individuals 2i and 2i + 1 of a generation are mirror images, z and -z (an odd last individual is unpaired); PAIRWISE also ranks only the better point of each pair. See modcma.c_maes.options.Mirror. With PAIRWISE, Settings increases an odd lambda0 by one; for a (1+1)-ES it sets PAIRWISE to NONE.
        """

    @mirrored.setter
    def mirrored(self, arg: modcma.c_maes.cmaescpp.options.Mirror, /) -> None: ...

    @property
    def ssa(self) -> modcma.c_maes.cmaescpp.options.StepSizeAdaptation:
        """
        options.StepSizeAdaptation, default CSA: the step-size adaptation.

        See modcma.c_maes.options.StepSizeAdaptation for the choices. Settings sets SR for a (1+1)-ES.
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

        Restart (with the same, an increasing or an alternating population size) or stop. See modcma.c_maes.options.RestartStrategy. With NONE the only criterion is UnableToAdapt (the update of the matrix failed, or sigma is not finite), and a run restarts when it holds; the other criteria need one of the other options. Settings sets IPOP and BIPOP to RESTART for a (1+1)-ES.
        """

    @restart_strategy.setter
    def restart_strategy(self, arg: modcma.c_maes.cmaescpp.options.RestartStrategy, /) -> None: ...

    @property
    def repelling_type(self) -> modcma.c_maes.cmaescpp.options.RepellingType:
        """
        options.RepellingType, default NONE: repelling restarts.

        Restarts avoid the regions of the optima found by earlier runs (taboo points). No effect without restarts (restart_strategy STOP, or NONE unless UnableToAdapt holds). See modcma.c_maes.options.RepellingType.
        """

    @repelling_type.setter
    def repelling_type(self, arg: modcma.c_maes.cmaescpp.options.RepellingType, /) -> None: ...

    @property
    def repelling_metric(self) -> modcma.c_maes.cmaescpp.options.RepellingMetric:
        """
        options.RepellingMetric, default CURRENT: the distance between a sample and a taboo point of repelling restarts.

        See modcma.c_maes.options.RepellingMetric for the choices (#158).
        """

    @repelling_metric.setter
    def repelling_metric(self, arg: modcma.c_maes.cmaescpp.options.RepellingMetric, /) -> None: ...

    @property
    def matrix_adaptation(self) -> modcma.c_maes.cmaescpp.options.MatrixAdaptationType:
        """
        options.MatrixAdaptationType, default COVARIANCE: the adaptation of the shape of the search distribution.

        Settings replaces COVARIANCE by COVARIANCE_NO_EIGV when ssa is not CSA, also for lambda0 = 1, which runs SR (unless always_compute_eigv). See modcma.c_maes.options.MatrixAdaptationType.
        """

    @matrix_adaptation.setter
    def matrix_adaptation(self, arg: modcma.c_maes.cmaescpp.options.MatrixAdaptationType, /) -> None: ...

    @property
    def center_placement(self) -> modcma.c_maes.cmaescpp.options.CenterPlacement:
        """
        options.CenterPlacement, default X0: where the mean of the search distribution starts after a restart.

        The first run always starts at Settings.x0, or Settings.center without x0. See modcma.c_maes.options.CenterPlacement for the choices.
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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Solution:
    """A point with its objective value and when it was found."""

    def __init__(self) -> None:
        """An empty solution: x is empty, y is +inf, t and e are 0."""

    @property
    def x(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (d,) array: the point, as evaluated (integer variables rounded). A copy: it keeps its values when the optimizer finds a better point, and changing it does not change the optimizer.
        """

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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Center(Solution):
    """
    The record of a finished run: its evaluated center of mass, with the distribution the run ended with, and why it ended (#111).

    Every run ends with an evaluation of its center of mass (`ModularCMAES.center_of_mass`), at a restart and when the optimizer stops: x and y are that point and its value, t the last generation of the run and e the evaluations after it. The run ended with the distribution N(m, sigma^2 C), C = B diag(d)^2 B^T, with d and B in `metric`; x is its mean m, clipped to the box when a bound correction is used and with the integer variables rounded. Every run is recorded, converged or not: `std` and `reasons` tell how it ended.
    """

    def __init__(self) -> None:
        """An empty record: x is empty, y is +inf, no metric, no reasons."""

    @property
    def sigma(self) -> float:
        """The step size at the end of the run."""

    @property
    def metric(self) -> modcma.c_maes.cmaescpp.matrix_adaptation.Metric:
        """
        The shape of the distribution at the end of the run, `matrix_adaptation.Metric` (the run's last `Adaptation.metric()`).
        """

    @property
    def reasons(self) -> list[str]:
        """
        list of str: why the run ended: the names of the restart criteria that were met, and "budget", "target" or "max_generations" when one of those ended it.
        """

    @property
    def std(self) -> float:
        """
        sigma * max(metric.d): the longest semi-axis of the run's one-sigma ellipsoid, the radius of the smallest ball that contains it. How far the run converged (NaN for an empty record).
        """

    def __repr__(self) -> str:
        """x, y and std as text."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Stats:
    """
    The progress of an optimizer: counters, the best solutions and the records of the finished runs.
    """

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

        Includes the extra points of TPA, the center of mass that ends every run and the hill-valley tests of repelling restarts.
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
        list of Solution: the best solution of every finished run, appended when it ends, at a restart and when the optimizer stops (a copy of current_best, so with RESAMPLE possibly outside the box). Its center of mass, evaluated after it, can be better: see centers.

        Reading returns a copy of the list and its elements, which stays valid and does not follow the optimizer; assign a list to change it.
        """

    @solutions.setter
    def solutions(self, arg: Sequence[Solution], /) -> None: ...

    @property
    def centers(self) -> list[Center]:
        """
        list of Center: the record of every finished run, appended when it ends, at a restart and when the optimizer stops: its evaluated center of mass, with sigma, the metric and why it ended. One per run, as solutions.

        Reading returns a copy of the list and its elements, which stays valid and does not follow the optimizer; assign a list to change it.
        """

    @centers.setter
    def centers(self, arg: Sequence[Center], /) -> None: ...

    @property
    def current_best(self) -> Solution:
        """
        The best solution of the current run (since the last restart), of every evaluated point; the success rules and the restarts read it. With RESAMPLE it can lie outside the box (see global_best).

        Reading returns the optimizer's own record, not a copy: an object held across a generation follows the optimizer (its x is a copy when read). copy.copy() keeps its values.
        """

    @current_best.setter
    def current_best(self, arg: Solution, /) -> None: ...

    @property
    def global_best(self) -> Solution:
        """
        The best feasible solution found so far, over all runs: the result of a run (`run()` returns None and leaves it here), and what `fmin` returns.

        With a bound correction (not NONE) only a point inside [lb, ub] becomes global_best: RESAMPLE evaluates a point outside the box after 5 d rejected tries, and such a point is not returned (#168). Without a bound correction the box does not constrain the points, and every point counts. Before 2.0 it could lie outside the box.

        Reading returns the optimizer's own record, as current_best: copy.copy() keeps its values.
        """

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
        The generation of the last update of the sampling factor of the matrix adaptation.

        For COVARIANCE and COVARIANCE_NO_EIGV the last decomposition of C (every Weights.lazy_update_interval generations; C itself is updated every generation); for SEPARABLE, MATRIX, CHOLESKY, CMSA and NATURAL_GRADIENT the last update of the matrix; NONE does not set it.
        """

    @last_update.setter
    def last_update(self, arg: int, /) -> None: ...

    @property
    def n_updates(self) -> int:
        """
        The number of updates of the sampling factor so far, counted as last_update.
        """

    @n_updates.setter
    def n_updates(self, arg: int, /) -> None: ...

    def __repr__(self) -> str:
        """
        The generation, the evaluations, the best solution and has_improved as text.
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Weights:
    """
    The recombination weights and the learning rates of a run.

    Computed when the Parameters are created, and again with the current mu and lambda at every restart and by Parameters.resize_population. A learning rate given in the Settings is used as is; the defaults below use d for the dimension.
    """

    def __init__(self, settings: Settings, expected_length_z: float) -> None:
        """
        The weights and learning rates for settings.mu0 and settings.lambda0.

        Parameters
        ----------
        settings : Settings
            The settings: dimension, modules and the learning rates given there.
        expected_length_z : float
            The expected length of a sample of the sampler, E||z|| (Parameters pass the sampler's expected_length()).
        """

    @property
    def mueff(self) -> float:
        """
        The variance effective selection mass of the positive weights, 1 / sum(positive^2).
        """

    @mueff.setter
    def mueff(self, arg: float, /) -> None: ...

    @property
    def mueff_mean(self) -> float:
        """
        The variance effective selection mass of the weights of the mean, 1 / sum(w^2): `mueff`, except for UTILITY, where the mean uses all lambda utilities. Normalizes the evolution paths.
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
        Whether the selection is pairwise (`Mirror.PAIRWISE`): the losers of the mirrored pairs (the individuals after `Population.n_ranked`) then get no negative weights in the active update, as in pycma.
        """

    @pairwise.setter
    def pairwise(self, arg: bool, /) -> None: ...

    @property
    def mueff_neg(self) -> float:
        """
        The variance effective selection mass of the negative weights, sum(negative)^2 / sum(negative^2) (for DEFAULT with mu > lambda / 2: of the mu negative weights of a population of 2 mu, see `negative`).
        """

    @mueff_neg.setter
    def mueff_neg(self, arg: float, /) -> None: ...

    @property
    def c1(self) -> float:
        """
        The learning rate of the rank-one update of the matrix.

        Default acov / ((d + 1.3)^2 + mueff) (Hansen, arXiv:1604.00772v2, Table 1, Eq. (57)); for SEPARABLE pycma's separable rate 1 / (d + 2 sqrt(d) + mueff / d) (c1_sep of CMA_diagonal, pycma 4.6.0); 2 / (d^2 + 6) for a (1+1)-ES (lambda0 = 1).
        """

    @c1.setter
    def c1(self, arg: float, /) -> None: ...

    @property
    def cmu(self) -> float:
        """
        The learning rate of the rank-mu update of the matrix.

        Default min(1 - c1, acov (mueff - 2 + 1 / mueff) / ((d + 2)^2 + acov mueff / 2)), as `cmaes` 0.13 (Hansen, arXiv:1604.00772v2, Table 1, Eq. (58), and pycma add 1/4 to mueff - 2 + 1 / mueff); for SEPARABLE pycma's separable rate min(1 - c1, (0.25 + mueff + 1 / mueff - 2) / (d + 4 sqrt(d) + mueff / 2)) (cmu_sep of CMA_diagonal, pycma 4.6.0), so that c1 + cmu <= 1; 2 / (d^2 + 6) for lambda0 = 1.
        """

    @cmu.setter
    def cmu(self, arg: float, /) -> None: ...

    @property
    def cc(self) -> float:
        """
        The learning rate of the evolution path pc.

        Default (4 + mueff / d) / (d + 4 + 2 mueff / d) (Hansen, arXiv:1604.00772v2, Table 1, Eq. (56)); 2 / (d + 2) for a (1+1)-ES. With NATURAL_GRADIENT the learning rate of A <- A expm(cc G): xNES's eta_B / 2 = (9 + 3 ln(d)) / (10 d sqrt(d)) (Glasmachers et al. 2010) with every ssa, or 1 / (4 d^1.5) for a (1+1)-ES.
        """

    @cc.setter
    def cc(self, arg: float, /) -> None: ...

    @property
    def cs(self) -> float:
        """
        The learning rate of the step-size adaptation.

        Default by ssa: CSA (mueff + 2) / (d + mueff + 5) (Hansen, The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772v2, Table 1, Eq. (55); pycma uses d + mueff + 3 and caps cs at 1/2); SR 2 / (d + 2); XNES mueff / (2 ln(max(2, d)) sqrt(d)), or with NATURAL_GRADIENT xNES's eta_sigma = (9 + 3 ln(d)) / (5 d sqrt(d)) (Glasmachers et al. 2010); MXNES 1; LPXNES min(1, 9 mueff / (10 sqrt(d))); otherwise (TPA, TPA_INPLACE, MSR, PSR, SA) 0.3. The rates of XNES, MXNES and LPXNES are those of Krause, Glasmachers & Igel (2017), Table 3; the cap of LPXNES's at 1 is modcma's (above 1 its update overshoots). cs is also the rate of the path ps, at most 1 (see `cs_path`). SR and SA do not use cs in their update of sigma: for them it is only the rate of ps.
        """

    @cs.setter
    def cs(self, arg: float, /) -> None: ...

    @property
    def cs_path(self) -> float:
        """
        The learning rate of the evolution path ps: min(cs, 1).

        The default cs of XNES, mueff / (2 ln(max(2, d)) sqrt(d)), exceeds 1 for large populations (13.8 at d = 2, lambda = 100), and LPXNES's exceeded 2 before it was capped; as the rate of ps a cs above 2 made ps NaN. At 1 ps is the current step, sqrt(mueff_mean) dz, as with MXNES (cs = 1). ps is read by CSA, by MATRIX (its rank-one update), and by h_sigma of COVARIANCE, COVARIANCE_NO_EIGV and SEPARABLE, and of MATRIX and CHOLESKY with a step-size rule other than CSA (#160).
        """

    @cs_path.setter
    def cs_path(self, arg: float, /) -> None: ...

    @property
    def damps(self) -> float:
        """
        The damping of the step-size update.

        Default for CSA 1 + 2 max(0, sqrt((mueff - 1) / (d + 1)) - 1) + cs (Hansen, arXiv:1604.00772v2, Table 1, Eq. (55)); for SR 1 + d / 2; for MSR max(1, 2 - 2 / d); for TPA_INPLACE 0.7 + 2 ln(d) + 2 ln(max(1, lambda - d)) (pycma 4.6.0, CMAAdaptSigmaTPA); otherwise (PSR among them) 1. Used by CSA, MSR, PSR, SR and TPA_INPLACE; the other step-size rules have no damping.
        """

    @damps.setter
    def damps(self, arg: float, /) -> None: ...

    @property
    def sqrt_cc_mueff(self) -> float:
        """
        sqrt(cc (2 - cc) mueff_mean): the normalisation of the update of the path pc (mueff_mean = mueff, or 1 / sum u_i^2 for the utilities of UTILITY).
        """

    @sqrt_cc_mueff.setter
    def sqrt_cc_mueff(self, arg: float, /) -> None: ...

    @property
    def sqrt_cs_mueff(self) -> float:
        """
        sqrt(cs_path (2 - cs_path) mueff_mean): the normalisation of the update of the path ps (mueff_mean = mueff, or 1 / sum u_i^2 for the utilities of UTILITY).
        """

    @sqrt_cs_mueff.setter
    def sqrt_cs_mueff(self, arg: float, /) -> None: ...

    @property
    def lazy_update_interval(self) -> float:
        """
        The number of generations between two decompositions of the covariance matrix, 1 / (c1 + cmu) / d / 10: the eigendecomposition of COVARIANCE, the Cholesky factorization of COVARIANCE_NO_EIGV. The matrix itself is updated every generation.

        At most 1 (with the default rates up to d = 82), the matrix is decomposed every generation.
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
        (1.4 + 2 / (d + 1)) expected_length_z: the threshold of h_sigma (Hansen 2016, App. A, symbols of Fig. 6). When the length of ps, corrected for the generations of the run, is not below it, the rank-one update stalls (the update of the path pc; MATRIX's rank-one term).
        """

    @expected_length_ps.setter
    def expected_length_ps(self, arg: float, /) -> None: ...

    @property
    def beta(self) -> float:
        """
        The learning rate of the sampled step sizes (sample_sigma): the standard deviation of the logarithm of an individual's step size.

        1 / sqrt(2 d) with SA (CMSA's tau, Beyer & Sendhoff 2008); ln(2) / (sqrt(d) ln(max(d, 2))) with LPXNES (Krause et al. 2017, Table 3, with ln(max(d, 2)) for d = 1); 1 / sqrt(2 mueff) otherwise.
        """

    @beta.setter
    def beta(self, arg: float, /) -> None: ...

    @property
    def weights(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (lambda,) array: all weights by rank, positive followed by negative (with UTILITY the utilities, which sum to zero).
        """

    @weights.setter
    def weights(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def positive(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (mu,) array: the weights of the mu best individuals, normalised to sum 1.

        DEFAULT: ln((n + 1) / 2) - ln(i), i = 1..mu, with n = lambda for the default mu = lambda // 2 (Hansen 2016, Eq. (49)) and n = 2 mu for another mu (as pycma with CMA_mu). With UTILITY the non-negative utilities of the mu best, normalised: the convex weights of SA, CMSA and of XNES (without NATURAL_GRADIENT) and LPXNES.
        """

    @positive.setter
    def positive(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def negative(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (lambda - mu,) array: the weights of the other individuals, for the active update.

        Scaled so that their absolute values sum to max(0, min(1 + c1 / cmu, 1 + 2 mueff_neg / (mueff + 2), (1 - c1 - cmu) / (d cmu))) (Hansen 2016, Eq. (50) to (53)). DEFAULT with the default mu: ln((lambda + 1) / 2) - ln(i), i = mu + 1..lambda. DEFAULT with another mu, as pycma with CMA_mu: the mu negative weights ln(mu + 1/2) - ln(i), i = mu + 1..2 mu, of a population of 2 mu go to the worst ranks; for lambda > 2 mu the ranks in between get 0, for lambda < 2 mu the worst lambda - mu ranks get the last lambda - mu of them (the scaling is that of all mu). With UTILITY the utilities are used as they are (they sum to zero, as in NES).
        """

    @negative.setter
    def negative(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    def for_population(self, n: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The weights for a population of n evaluated individuals.

        After sequential selection stopped a generation early (mu < n < lambda), the positive weights are unchanged and the negative weights of ranks mu..n-1 are scaled so that their sum equals the sum of all lambda - mu negative weights. For n >= lambda, these are the weights themselves; for n <= mu, the first n positive weights. With UTILITY, the utilities of min(n, lambda) points.

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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Settings:
    """
    The settings of a run: dimension, modules, stopping conditions, search space and strategy parameters.

    A strategy parameter left None gets its default when the Parameters are created (see Weights). The constructor computes the derived values (db, center, diameter, volume, has_bounds) and adjusts the modules.

    Setting a field afterwards checks that field alone (ValueError for a wrong length, NaN, a sign or a range) and recomputes what is derived from it: lb and ub the derived values of the box, sigma0 the center, lambda0 the default mu0. The fields can therefore be set in any order. The checks between fields (lb <= ub, the integer variables in the box, c1 + cmu <= 1, mu0 <= lambda0, the budget against the first generation, lambda0 with the modules) run in the constructor and again when an optimizer is created from the settings (ModularCMAES(settings), Parameters(settings)): ValueError naming the fields. Changing the modules afterwards, in place, does not adjust them.
    """

    def __init__(self, dim: int, modules: Modules | None = None, *, target: float | None = None, max_generations: int | None = None, budget: int | None = None, sigma0: float | None = None, mu0: int | None = None, lambda0: int | None = None, x0: numpy.typing.ArrayLike | None = None, lb: numpy.typing.ArrayLike | None = None, ub: numpy.typing.ArrayLike | None = None, integer_variables: numpy.typing.ArrayLike | None = None, cs: float | None = None, cc: float | None = None, cmu: float | None = None, c1: float | None = None, damps: float | None = None, acov: float | None = None, verbose: bool = False, always_compute_eigv: bool = False, seed: int | None = None, fast_sampling: bool = False, C0: numpy.typing.ArrayLike | None = None) -> None:
        """
        Settings for a problem of dimension dim.

        dim and modules are positional, every other argument is keyword-only. None means a default computed from the dimension and the modules.

        The constructor makes the configuration consistent:

        - with mirrored PAIRWISE, an odd lambda0 above 1 is increased by one;
        - with ssa SA or matrix_adaptation CMSA, the default mu0 is lambda0 // 4;
        - with matrix_adaptation NATURAL_GRADIENT and active, weights DEFAULT become UTILITY;
        - with weights UTILITY, active is turned on;
        - with matrix_adaptation NONE or CMSA, active is turned off;
        - lambda0 = 1 gives a (1+1)-ES (one_plus_one): mu0 = 1, elitist, ssa SR, EQUAL weights, no active update, no sequential selection, mirrored PAIRWISE becomes NONE, and IPOP or BIPOP become RESTART;
        - then, when ssa is not CSA (so also for lambda0 = 1), matrix_adaptation COVARIANCE becomes COVARIANCE_NO_EIGV (unless always_compute_eigv);
        - when the box lb..ub is not finite, has_bounds is False, diameter 10, volume 10^dim; center is computed per coordinate (see `center`).

        Invalid values raise ValueError, naming the argument: dim 0, lambda0 0, mu0 0, a mu0 above lambda0 (except for lambda0 = 1, which uses mu0 = 1), a NaN target, max_generations 0, a sigma0 that is not positive and finite, x0, lb or ub of a length other than dim, a non-finite x0, NaN in lb or ub, +inf in lb or -inf in ub, lb above ub, an integer variable without an integer value in [lb, ub], a budget smaller than the first generation and the evaluation of the final center of mass, a learning rate out of its range (cs and cc in (0, 1], c1 and cmu in [0, 1], c1 + cmu above 1 when both are given, damps not positive, acov not positive and finite; #147), or a C0 that is not a symmetric positive definite (dim, dim) matrix (or with matrix_adaptation NONE). An index of integer_variables outside 0..dim-1 raises ValueError when the optimizer is created.

        Parameters
        ----------
        dim : int
            The dimension of the search space.
        modules : Modules, optional
            The module configuration (copied). Default Modules(): CMA-ES with CSA.
        target : float, optional
            Stop when the best objective value is at most target; not NaN. Default None: no target.
        max_generations : int, optional
            Stop after this many generations, at least 1. Default None: no limit.
        budget : int, optional
            The maximum number of evaluations, default 10000 * dim; at least the first generation (lambda0, and TPA's 2 points) and the final center of mass. A generation runs only when it leaves one evaluation: the run stops when the evaluations so far plus lambda (and TPA's 2 points, reserved also in a generation that skips them, #179) plus 1 would exceed it, and every run ends with the evaluation of its center of mass (#111).
        sigma0 : float, optional
            The initial step size, default 2.0, positive and finite; also the step size after a restart (the small-population runs of BIPOP use sigma0 * 10^(-2 u), u uniform in (0, 1)).
        mu0 : int, optional
            The number of parents, default lambda0 // 2 (lambda0 // 4 with ssa SA or matrix_adaptation CMSA); at least 1 and at most lambda0.
        lambda0 : int, optional
            The number of offspring per generation, default 4 + floor(3 ln(dim)); at least 1 (1 is the (1+1)-ES). Many modules fail with 2, a few with 3 (see the compatibility page).
        x0 : array_like, optional
            (dim,) the initial mean of the search distribution. Default None: `center`, the midpoint of the box (see `center` for infinite bounds); after a restart, center_placement X0 uses the same point.
        lb : array_like, optional
            (dim,) the lower bounds, default -5 in every coordinate.
        ub : array_like, optional
            (dim,) the upper bounds, default 5 in every coordinate.
        integer_variables : sequence of int, optional
            The indices of the variables that take integer values: they are rounded before evaluation (a value in [lb, ub] to the nearest integer in [lb, ub]) and their step size has a lower bound (the step size times sqrt(C_ii) is at least min(0.2, mueff / dim)). Each needs an integer value in [lb, ub]. Default: none.
        cs : float, optional
            The learning rate of the step-size adaptation, in (0, 1]. Default depends on ssa (see Weights.cs); for CSA (mueff + 2) / (dim + mueff + 5) (Hansen, The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772v2, Table 1, Eq. (55)).
        cc : float, optional
            The learning rate of the evolution path pc, in (0, 1]. Default (4 + mueff / dim) / (dim + 4 + 2 mueff / dim) (see Weights.cc).
        cmu : float, optional
            The learning rate of the rank-mu update, in [0, 1]. Default min(1 - c1, acov (mueff - 2 + 1 / mueff) / ((dim + 2)^2 + acov mueff / 2)) (see Weights.cmu).
        c1 : float, optional
            The learning rate of the rank-one update, in [0, 1]; with cmu, c1 + cmu <= 1. Default acov / ((dim + 1.3)^2 + mueff) (see Weights.c1).
        damps : float, optional
            The damping of the step-size update, positive. Default for CSA 1 + 2 max(0, sqrt((mueff - 1) / (dim + 1)) - 1) + cs (see Weights.damps).
        acov : float, optional
            The factor alpha_cov in the defaults of c1 and cmu, positive, default 2 (not used by SEPARABLE, whose defaults are pycma's separable rates).
        verbose : bool, default False
            Print a message at every restart and when a decomposition of the matrix fails.
        always_compute_eigv : bool, default False
            Keep matrix_adaptation COVARIANCE, with its eigendecomposition, when ssa is not CSA.
        seed : int, optional
            The seed of the optimizer's own random generator. Default None: the optimizer starts from a copy of the global generator (set_seed) when it is created.
        fast_sampling : bool, default False
            Sample with xoshiro256++ and a ziggurat instead of mt19937 and the inverse normal CDF (the default UNIFORM + GAUSSIAN sampler only): faster, but a different random stream.
        C0 : array_like, optional
            (dim, dim) the initial covariance matrix C, symmetric positive definite: the first generation samples N(x0, sigma0^2 C0). Default None: the identity. See the attribute C0; modcma.warm_start estimates x0, sigma0 and C0 from the solutions of a related task.
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
    def C0(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')] | None:
        """
        (dim, dim) array: the initial covariance matrix C, or None (default) for the identity.

        The first generation samples N(x0, sigma0^2 C0) (warm starting, Nomura et al. 2021; see modcma.warm_start). Each matrix adaptation starts its matrix from a factor F of C0, F F^T = C0: COVARIANCE from the eigendecomposition, COVARIANCE_NO_EIGV, CHOLESKY and CMSA from the lower Cholesky factor, MATRIX and NATURAL_GRADIENT from the symmetric square root. SEPARABLE uses only the diagonal of C0; matrix_adaptation NONE rejects C0 (ValueError). Every restart starts from C0 again, as from sigma0 and x0.

        Setting it checks it: a (dim, dim) finite matrix, symmetric up to rounding (a relative difference ||C0 - C0^T|| <= 1e-10 ||C0||, Frobenius norm; its symmetric part (C0 + C0^T) / 2 is stored) and positive definite, otherwise ValueError. Set it before the optimizer is created; the getter returns a copy.
        """

    @C0.setter
    def C0(self, arg: numpy.typing.ArrayLike | None, /) -> None: ...

    @property
    def modules(self) -> Modules:
        """
        The module configuration, after the adjustments of the constructor.

        The attribute is read-only. Its fields can still be changed in place, but the adjustments of the constructor are not applied again, and an optimizer that exists keeps the modules it was created with.
        """

    @property
    def target(self) -> float | None:
        """
        Stop when the best objective value is at most target; None (default): no target. NaN raises ValueError: it would never stop the run.
        """

    @target.setter
    def target(self, arg: float | None, /) -> None: ...

    @property
    def max_generations(self) -> int | None:
        """
        Stop after this many generations, at least 1 (ValueError); None (default): no limit.
        """

    @max_generations.setter
    def max_generations(self, arg: int | None, /) -> None: ...

    @property
    def budget(self) -> int:
        """
        The maximum number of evaluations, default 10000 * dim; at least 2 (ValueError).

        A generation runs only when it leaves one evaluation: the run stops when the evaluations so far plus lambda (and TPA's 2 points, reserved also in a generation that skips them, #179) plus 1 would exceed it, and every run ends with the evaluation of its center of mass (#111). The budget must hold the first generation (lambda0, and TPA's 2 points) and the final center of mass: the constructor checks it, and after a setter the optimizer does when it is created (ValueError).
        """

    @budget.setter
    def budget(self, arg: int, /) -> None: ...

    @property
    def sigma0(self) -> float:
        """
        The initial step size, default 2.0; positive and finite (ValueError). Also the step size after a restart (smaller for the small-population runs of BIPOP). Setting it recomputes `center`, which depends on it for a coordinate with one infinite bound.
        """

    @sigma0.setter
    def sigma0(self, arg: float, /) -> None: ...

    @property
    def mu0(self) -> int:
        """
        The initial number of parents, at least 1 (ValueError); default lambda0 // 2 (lambda0 // 4 with ssa SA or matrix_adaptation CMSA).

        Until mu0 is given, to the constructor or by setting it, it follows lambda0: setting lambda0 recomputes it. A mu0 above lambda0 raises ValueError: in the constructor, and after a setter when the optimizer is created. Parameters.mu is the current value, which a restart that changes lambda changes (IPOP, BIPOP, the cap of the first generation at the budget): to the default of the new lambda (as pycma), or, when mu0 was given, to mu0 / lambda0 of the new lambda.
        """

    @mu0.setter
    def mu0(self, arg: int, /) -> None: ...

    @property
    def lambda0(self) -> int:
        """
        The initial number of offspring per generation, at least 1 (ValueError); default 4 + floor(3 ln(dim)).

        Setting it recomputes mu0, unless mu0 was given. The setter does not apply the adjustments of the constructor; the optimizer raises ValueError when it is created with an odd lambda0 and mirrored PAIRWISE (the constructor adds one), with lambda0 = 1 set after construction or another lambda0 for settings created with lambda0 = 1 (the (1+1)-ES, for which the constructor adjusts the modules), or with a budget that does not hold the first generation. Parameters.lamb is the current value, which IPOP and BIPOP restarts change, and a restart near the end of the budget caps.
        """

    @lambda0.setter
    def lambda0(self, arg: int, /) -> None: ...

    @property
    def x0(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None:
        """
        (dim,) array: the initial mean of the search distribution, or None (default) for `center` (the midpoint of the bounds; see `center` for infinite bounds). A length other than dim, or a value that is not finite, raises ValueError.
        """

    @x0.setter
    def x0(self, arg: numpy.typing.ArrayLike | None, /) -> None: ...

    @property
    def lb(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (dim,) array: the lower bounds, default -5; -inf for none. lb_i == ub_i, finite, fixes the variable. Setting it updates `db`, `center`, `diameter`, `volume` and `has_bounds`; ValueError for a length other than dim, NaN or +inf. Its relation to `ub` (lb <= ub, the integer variables in the box) is checked when the optimizer is created, so lb and ub can be set in any order.
        """

    @lb.setter
    def lb(self, arg: numpy.typing.ArrayLike, /) -> None: ...

    @property
    def ub(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (dim,) array: the upper bounds, default 5; inf for none. lb_i == ub_i, finite, fixes the variable. Setting it updates `db`, `center`, `diameter`, `volume` and `has_bounds`; ValueError for a length other than dim, NaN or -inf. Its relation to `lb` (lb <= ub, the integer variables in the box) is checked when the optimizer is created, so lb and ub can be set in any order.
        """

    @ub.setter
    def ub(self, arg: numpy.typing.ArrayLike, /) -> None: ...

    @property
    def db(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)]:
        """
        (dim,) array: the widths of the box, ub - lb (read-only: derived from lb and ub).
        """

    @property
    def center(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)]:
        """
        (dim,) array: where the first run starts without x0, per coordinate: the midpoint (lb + ub) / 2 for two finite bounds; 0 for two infinite bounds; for one infinite bound, 0 if the half-line holds it, otherwise the finite bound moved inward by sigma0 (lb + sigma0, or ub - sigma0), so the start lies in the box (read-only: derived from lb, ub and sigma0; x0 sets the starting point).
        """

    @property
    def diameter(self) -> float:
        """
        The length of the diagonal of the box, ||ub - lb||; 10 when the box is not finite (then volume is 10^dim). Read-only: derived from lb and ub.
        """

    @property
    def volume(self) -> float:
        """
        The volume of the box, prod(ub - lb); 10^dim when the box is not finite. Read-only: derived from lb and ub.
        """

    @property
    def has_bounds(self) -> bool:
        """
        Whether the box lb..ub is finite (`diameter` and `volume` depend on it; `center` is computed per coordinate); read-only, derived from lb and ub. Bound correction does not depend on it: it corrects every finite bound.
        """

    @property
    def integer_variables(self) -> Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')]:
        """(k,) int array: the indices of the integer variables (read-only)."""

    @property
    def cs(self) -> float | None:
        """
        The learning rate cs, or None (default) for the module's default; in (0, 1] (ValueError).

        The value in use is Parameters.weights.cs.
        """

    @cs.setter
    def cs(self, arg: float | None, /) -> None: ...

    @property
    def cc(self) -> float | None:
        """
        The learning rate cc, or None (default) for the module's default; in (0, 1] (ValueError).

        The value in use is Parameters.weights.cc.
        """

    @cc.setter
    def cc(self, arg: float | None, /) -> None: ...

    @property
    def cmu(self) -> float | None:
        """
        The learning rate cmu, or None (default) for the module's default; in [0, 1] (ValueError). With c1 given, c1 + cmu must be at most 1 (ValueError when the optimizer is created).

        The value in use is Parameters.weights.cmu.
        """

    @cmu.setter
    def cmu(self, arg: float | None, /) -> None: ...

    @property
    def c1(self) -> float | None:
        """
        The learning rate c1, or None (default) for the module's default; in [0, 1] (ValueError). With cmu given, c1 + cmu must be at most 1 (ValueError when the optimizer is created).

        The value in use is Parameters.weights.c1.
        """

    @c1.setter
    def c1(self, arg: float | None, /) -> None: ...

    @property
    def damps(self) -> float | None:
        """
        The damping of the step-size update, or None (default) for the module's default; positive (ValueError).

        The value in use is Parameters.weights.damps.
        """

    @damps.setter
    def damps(self, arg: float | None, /) -> None: ...

    @property
    def acov(self) -> float | None:
        """
        The factor alpha_cov of the defaults of c1 and cmu, or None for the default 2; positive and finite (ValueError). SEPARABLE does not use it: its defaults are pycma's separable rates.
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
        """
        Whether the settings were created with lambda0 = 1: a (1+1)-ES (read-only; see the constructor).
        """

    def __repr__(self) -> str:
        """The settings as text."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class CoordinateMapping:
    """
    Maps a sample to the point that is evaluated: the integer variables are rounded.

    Rounding is to the nearest integer, halfway cases away from zero. A coordinate inside the bounds [lb, ub] is rounded to the nearest integer in [ceil(lb), floor(ub)], so it stays inside; a coordinate outside them is rounded to the nearest integer (#79).
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

    def transform(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], lb: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None = None, ub: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None = None) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        A copy of the point x with the integer variables rounded.

        Parameters
        ----------
        x : numpy.ndarray
            The point, shape (d,).
        lb, ub : numpy.ndarray, optional
            The bounds, shape (d,); an integer variable of x inside them is rounded to the nearest integer inside them. Default None: no bound.

        Returns
        -------
        numpy.ndarray
            The rounded point, shape (d,).
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """
