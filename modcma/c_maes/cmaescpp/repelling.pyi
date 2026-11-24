"""
Repelling restarts: an archive of the optima of earlier runs (taboo points), and the rejection of samples close to them.

The optimizer holds one as `es.p.repelling`, selected by `Modules.repelling_type` (`options.RepellingType`); `Modules.repelling_metric` (`options.RepellingMetric`) sets the distance. Helps restarts find new basins of attraction on multimodal functions. de Nobel et al. (2024).
"""

from collections.abc import Callable, Sequence
from typing import Annotated, overload

import numpy
import numpy.typing
from numpy.typing import NDArray

import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.parameters


class TabooPoint:
    """
    A taboo point: an optimum found by an earlier run, with a radius around it in which samples are rejected.

    The distance of a sample to it is measured in the metric of `Modules.repelling_metric` (EUCLIDEAN; CURRENT: the Mahalanobis distance in the current search distribution; ARCHIVED: `metric`) and divided by the current sigma. A sample is rejected when this distance is less than shrinkage^attempts * radius, with attempts the number of samples rejected so far in the population.
    """

    @overload
    def __init__(self, solution: modcma.c_maes.cmaescpp.parameters.Solution, radius: float, metric: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')] = ...) -> None:
        """
        Create a taboo point with an unbounded radius (as `CoverageRepelling` does).

        Parameters
        ----------
        solution : Solution
            The optimum (its x and objective value y).
        radius : float
            The rejection radius, in units of sigma.
        metric : numpy.ndarray
            For `RepellingMetric.ARCHIVED`: A^-1 of the run that found the point, scaled to unit determinant, shape (d, d). Empty (the default) for the other metrics.
        """

    @overload
    def __init__(self, solution: modcma.c_maes.cmaescpp.parameters.Solution, radius: float, min_radius: float, max_radius: float, metric: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')] = ...) -> None:
        """
        Create a taboo point whose radius stays in [min_radius, max_radius] (as `AdaptiveRepelling` does).

        Parameters
        ----------
        solution : Solution
            The optimum (its x and objective value y).
        radius : float
            The rejection radius, in units of sigma.
        min_radius : float
            The lower bound of the radius.
        max_radius : float
            The upper bound of the radius.
        metric : numpy.ndarray
            For `RepellingMetric.ARCHIVED`: A^-1 of the run that found the point, scaled to unit determinant, shape (d, d). Empty (the default) for the other metrics.
        """

    def distance(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], p: modcma.c_maes.cmaescpp.Parameters) -> float:
        """
        The distance from x to this point, divided by the current sigma.

        Parameters
        ----------
        x : numpy.ndarray
            The point, shape (d,).
        p : Parameters
            The state of the optimizer; `settings.modules.repelling_metric` selects the metric.

        Returns
        -------
        float
            The distance in units of sigma.
        """

    @property
    def metric(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        For `RepellingMetric.ARCHIVED`: A^-1 of the run that found the point, scaled to unit determinant (only the shape of its basin), shape (d, d). Empty for the other metrics, and when A^-1 was degenerate (the Euclidean distance is used then).
        """

    @metric.setter
    def metric(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    def rejects(self, xi: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], p: modcma.c_maes.cmaescpp.Parameters, attempts: int) -> bool:
        """
        Whether the sample xi lies within the rejection radius shrinkage^attempts * radius.

        Parameters
        ----------
        xi : numpy.ndarray
            The sample, shape (d,).
        p : Parameters
            The state of the optimizer.
        attempts : int
            The number of samples rejected so far in this population; every rejection shrinks the radius.

        Returns
        -------
        bool
        """

    def shares_basin(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], float], xi: modcma.c_maes.cmaescpp.parameters.Solution, p: modcma.c_maes.cmaescpp.Parameters) -> bool:
        """
        Whether a solution lies in the basin of this point, by a hill-valley test.

        The objective is evaluated at 10 points evenly spaced on the line between the two; they share the basin when none of them is worse than the worse of the two solutions. The evaluations are counted in `p.stats` (and can update its best solution).

        Parameters
        ----------
        objective : Callable[[numpy.ndarray], float]
            The objective function.
        xi : Solution
            The solution to test (x and its objective value y).
        p : Parameters
            The state of the optimizer.

        Returns
        -------
        bool
        """

    def calculate_criticality(self, p: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Compute `criticality` from the distance delta of the current mean to this point: Phi(delta + radius) - Phi(delta - radius), with Phi the standard normal CDF.

        Parameters
        ----------
        p : Parameters
            The state of the optimizer (the mean and sigma).
        """

    def clamp_radius(self) -> None:
        """Clamp `radius` to [min_radius, max_radius]."""

    @property
    def solution(self) -> modcma.c_maes.cmaescpp.parameters.Solution:
        """The optimum (a Solution: x, y, t, e)."""

    @solution.setter
    def solution(self, arg: modcma.c_maes.cmaescpp.parameters.Solution, /) -> None: ...

    @property
    def radius(self) -> float:
        """The rejection radius, in units of sigma."""

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def min_radius(self) -> float:
        """The lower bound of the radius (default 0)."""

    @min_radius.setter
    def min_radius(self, arg: float, /) -> None: ...

    @property
    def max_radius(self) -> float:
        """The upper bound of the radius (default inf)."""

    @max_radius.setter
    def max_radius(self, arg: float, /) -> None: ...

    @property
    def shrinkage(self) -> float:
        """
        The factor by which the radius shrinks for every sample rejected in a population, 0.95^(1/d).
        """

    @shrinkage.setter
    def shrinkage(self, arg: float, /) -> None: ...

    @property
    def n_rep(self) -> int:
        """
        The number of runs that ended in the basin of this point (1 when it is created).
        """

    @n_rep.setter
    def n_rep(self, arg: int, /) -> None: ...

    @property
    def criticality(self) -> float:
        """
        How close the current mean is to this point relative to its radius (see `calculate_criticality`); points with a criticality below 0.01 do not reject samples.
        """

    @criticality.setter
    def criticality(self, arg: float, /) -> None: ...

    @property
    def duplicate_evaluations(self) -> float:
        """
        `AdaptiveRepelling`: the evaluations of the runs that ended in this basin again.
        """

    @duplicate_evaluations.setter
    def duplicate_evaluations(self, arg: float, /) -> None: ...

    @property
    def rejected_count(self) -> int:
        """
        The samples this point rejected (reset after every population by `AdaptiveRepelling`).
        """

    @rejected_count.setter
    def rejected_count(self, arg: int, /) -> None: ...

    @property
    def checked_count(self) -> int:
        """
        The samples tested against this point (reset after every population by `AdaptiveRepelling`).
        """

    @checked_count.setter
    def checked_count(self, arg: int, /) -> None: ...

    @property
    def last_rejection_rate(self) -> float:
        """
        `AdaptiveRepelling`: rejected_count / checked_count of the last population.
        """

    @last_rejection_rate.setter
    def last_rejection_rate(self, arg: float, /) -> None: ...

    def __repr__(self) -> str:
        """A summary: n_rep, y, radius and criticality."""

class Repelling:
    """
    Base class of the repelling restarts: the archive of taboo points and the rejection of samples.

    Before each population the archive is sorted by criticality (`prepare_sampling`); each sample is tested against the points with criticality >= 0.01 (`is_rejected`) and sampled again when one rejects it. After a restart, the better of the last mean and the best solution of the run is compared with the archive by hill-valley tests (`update_archive`); the subclasses add it as a new taboo point or update the point whose basin it shares. The base class does not add points.
    """

    def __init__(self) -> None:
        """Create a repelling with an empty archive."""

    def is_rejected(self, xi: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], p: modcma.c_maes.cmaescpp.Parameters) -> bool:
        """
        Whether a critical taboo point rejects the sample xi; a rejection is counted in `attempts` and in the point.

        Parameters
        ----------
        xi : numpy.ndarray
            The sample, shape (d,).
        p : Parameters
            The state of the optimizer.

        Returns
        -------
        bool
        """

    def update_archive(self, objective: Callable[[numpy.typing.NDArray[numpy.float64]], float], p: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Update the archive after a restart, evaluating the hill-valley tests with objective.

        The candidate is the better of the last mean and the last best solution in `p.stats`; it is tested against every taboo point (10 evaluations per point, counted in `p.stats.evaluations`). The optimizer runs the same update through its ask-tell interface.

        Parameters
        ----------
        objective : Callable[[numpy.ndarray], float]
            The objective function.
        p : Parameters
            The state of the optimizer.
        """

    def prepare_sampling(self, p: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Before a population: reset `attempts` and sort the archive by criticality (`AdaptiveRepelling` adapts the radii first).

        Parameters
        ----------
        p : Parameters
            The state of the optimizer.
        """

    def adapt_radii_after_sampling(self, p: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        After a population: adapt the radii (`AdaptiveRepelling`; nothing for the others).

        Parameters
        ----------
        p : Parameters
            The state of the optimizer.
        """

    @property
    def archive(self) -> list[TabooPoint]:
        """
        The taboo points, a list of TabooPoint (sorted by criticality before each population). Reading returns a copy; assign a list to change it.
        """

    @archive.setter
    def archive(self, arg: Sequence[TabooPoint], /) -> None: ...

    @property
    def attempts(self) -> int:
        """The number of samples rejected in the current population."""

    @attempts.setter
    def attempts(self, arg: int, /) -> None: ...

    @property
    def coverage(self) -> float:
        """
        `CoverageRepelling`: the coverage factor in the formula of the radius (default 2).
        """

    @coverage.setter
    def coverage(self, arg: float, /) -> None: ...

class CoverageRepelling(Repelling):
    """
    Repelling with radii from a coverage of the search space (`RepellingType.COVERAGE`).

    A candidate that shares no basin with a taboo point becomes a new one; when it shares a basin, the n_rep of that point grows and the point keeps the better solution. After each update all radii are set to r = (V n_rep / (sigma0 coverage n))^(1/d) Gamma(d/2 + 1)^(1/d) / sqrt(pi d), with V the volume of the box and n the number of runs so far, so that points found more often repel more. de Nobel et al. (2024).
    """

    def __init__(self) -> None:
        """Create a coverage repelling with an empty archive."""

class AdaptiveRepelling(Repelling):
    """
    Repelling with adaptive radii (`RepellingType.ADAPTIVE`).

    A new taboo point gets the radius initial_radius_factor / sqrt(d), bounded to [min_radius_factor, max_radius_factor] / sqrt(d). When a run ends in the basin of a point again (the first one in the archive that shares it), its radius grows by exp(grow_eta * min(w, max_waste_ratio)), with w the evaluations of that run divided by their running average. After each population, the radius of a critical point shrinks by shrink_factor when the share of rejected samples of the population exceeds max_rejection_rate and the point's own rejection rate exceeds local_max_rejection_rate (after at least min_checks_before_local_shrink checks).
    """

    def __init__(self) -> None:
        """Create an adaptive repelling with an empty archive."""

    def initial_radius(self, p: modcma.c_maes.cmaescpp.Parameters) -> float:
        """
        The radius of a new taboo point, initial_radius_factor / sqrt(d).

        Parameters
        ----------
        p : Parameters
            The state of the optimizer (the dimension).

        Returns
        -------
        float
        """

    def min_radius(self, p: modcma.c_maes.cmaescpp.Parameters) -> float:
        """
        The lower bound of the radius of a new taboo point, min_radius_factor / sqrt(d).

        Parameters
        ----------
        p : Parameters
            The state of the optimizer (the dimension).

        Returns
        -------
        float
        """

    def max_radius(self, p: modcma.c_maes.cmaescpp.Parameters) -> float:
        """
        The upper bound of the radius of a new taboo point, max_radius_factor / sqrt(d).

        Parameters
        ----------
        p : Parameters
            The state of the optimizer (the dimension).

        Returns
        -------
        float
        """

    @property
    def initial_radius_factor(self) -> float:
        """The radius of a new taboo point times sqrt(d) (default 0.5)."""

    @initial_radius_factor.setter
    def initial_radius_factor(self, arg: float, /) -> None: ...

    @property
    def min_radius_factor(self) -> float:
        """The lower bound of a radius times sqrt(d) (default 0.05)."""

    @min_radius_factor.setter
    def min_radius_factor(self, arg: float, /) -> None: ...

    @property
    def max_radius_factor(self) -> float:
        """The upper bound of a radius times sqrt(d) (default 2)."""

    @max_radius_factor.setter
    def max_radius_factor(self, arg: float, /) -> None: ...

    @property
    def shrink_factor(self) -> float:
        """The factor applied to a radius that shrinks (default 0.9)."""

    @shrink_factor.setter
    def shrink_factor(self, arg: float, /) -> None: ...

    @property
    def target_rejection_rate(self) -> float:
        """
        A target share of rejected samples (default 0.2); not used by the current update.
        """

    @target_rejection_rate.setter
    def target_rejection_rate(self, arg: float, /) -> None: ...

    @property
    def max_rejection_rate(self) -> float:
        """
        The share of rejected samples of a population, rejected / (lambda + rejected), above which radii can shrink (default 0.5).
        """

    @max_rejection_rate.setter
    def max_rejection_rate(self, arg: float, /) -> None: ...

    @property
    def last_restart_evaluations(self) -> int:
        """The evaluation count at the last archive update (0 before the first)."""

    @last_restart_evaluations.setter
    def last_restart_evaluations(self, arg: int, /) -> None: ...

    @property
    def average_restart_evaluations(self) -> float:
        """
        The running average of the evaluations per run (default 1), smoothed with restart_eval_alpha.
        """

    @average_restart_evaluations.setter
    def average_restart_evaluations(self, arg: float, /) -> None: ...

    @property
    def restart_eval_alpha(self) -> float:
        """The smoothing factor of average_restart_evaluations (default 0.2)."""

    @restart_eval_alpha.setter
    def restart_eval_alpha(self, arg: float, /) -> None: ...

    @property
    def grow_eta(self) -> float:
        """
        The growth rate of the radius of a point whose basin is found again (default 0.15).
        """

    @grow_eta.setter
    def grow_eta(self, arg: float, /) -> None: ...

    @property
    def max_waste_ratio(self) -> float:
        """
        The cap on the ratio of a run's evaluations to their running average in the growth of a radius (default 3).
        """

    @max_waste_ratio.setter
    def max_waste_ratio(self, arg: float, /) -> None: ...

    @property
    def local_target_rejection_rate(self) -> float:
        """
        A target rejection rate of a single point (default 0.2); not used by the current update.
        """

    @local_target_rejection_rate.setter
    def local_target_rejection_rate(self, arg: float, /) -> None: ...

    @property
    def local_max_rejection_rate(self) -> float:
        """
        The rejection rate of a point above which its radius can shrink (default 0.5).
        """

    @local_max_rejection_rate.setter
    def local_max_rejection_rate(self, arg: float, /) -> None: ...

    @property
    def min_checks_before_local_shrink(self) -> int:
        """
        The number of samples a point must have tested in a population before its radius can shrink (default 5).
        """

    @min_checks_before_local_shrink.setter
    def min_checks_before_local_shrink(self, arg: int, /) -> None: ...

class NoRepelling(Repelling):
    """
    No repelling (`RepellingType.NONE`, the default): no sample is rejected and the archive stays empty.
    """

    def __init__(self) -> None:
        """Create the no-op repelling."""

def euclidian(u: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], v: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> float:
    """
    The Euclidean distance ||u - v||.

    Parameters
    ----------
    u, v : numpy.ndarray
        The points, shape (d,).

    Returns
    -------
    float
    """

def manhattan(u: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], v: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> float:
    """
    The Manhattan distance sum_i |u_i - v_i|.

    Parameters
    ----------
    u, v : numpy.ndarray
        The points, shape (d,).

    Returns
    -------
    float
    """

def mahanolobis(u: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], v: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], C_inv: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]) -> float:
    """
    The Mahalanobis distance sqrt((u - v)^T C_inv (u - v)).

    Parameters
    ----------
    u, v : numpy.ndarray
        The points, shape (d,).
    C_inv : numpy.ndarray
        The inverse of the covariance matrix, shape (d, d).

    Returns
    -------
    float
    """

def archived_metric(p: modcma.c_maes.cmaescpp.Parameters) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
    """
    The metric that `RepellingMetric.ARCHIVED` stores with a new taboo point.

    A^-1 of the current run (for every matrix adaptation), scaled to unit determinant, shape (d, d). Empty for the other metrics, and when A^-1 is degenerate.

    Parameters
    ----------
    p : Parameters
        The state of the optimizer.

    Returns
    -------
    numpy.ndarray
    """

def hill_valley_test(u: modcma.c_maes.cmaescpp.parameters.Solution, v: modcma.c_maes.cmaescpp.parameters.Solution, f: Callable[[numpy.typing.NDArray[numpy.float64]], float], n_evals: int) -> bool:
    """
    Whether two solutions lie in the same basin of attraction (hill-valley test).

    f is evaluated at n_evals points evenly spaced on the line between u.x and v.x; the two share a basin when none of them is worse than max(u.y, v.y). Stops at the first worse point. Maree et al. (2018).

    Parameters
    ----------
    u, v : Solution
        The two solutions (x and their objective values y).
    f : Callable[[numpy.ndarray], float]
        The objective function.
    n_evals : int
        The number of points on the line.

    Returns
    -------
    bool
        True when no point between them is worse (one basin).
    """
