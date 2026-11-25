"""
Mutation: the sampling of a population and the step-size adaptation.

The optimizer holds a `Strategy` subclass as `es.p.mutation`, selected by `Modules.ssa` (`options.StepSizeAdaptation`) and created anew at every restart. It holds the threshold convergence (`Modules.threshold_convergence`), the sequential selection (`Modules.sequential_selection`) and the sampler of the step sizes (`Modules.sample_sigma`). The step-size adaptations are described in de Nobel et al. (2021).
"""

from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.matrix_adaptation
import modcma.c_maes.cmaescpp.options
import modcma.c_maes.cmaescpp.parameters


class ThresholdConvergence:
    """
    Threshold convergence: samples shorter than a decreasing threshold are lengthened (`Modules.threshold_convergence`).

    The threshold is a length of z: t = init_threshold * E||z|| * ((budget - evaluations) / budget)^decay_factor, computed once per generation from the evaluations at its start (0 once the budget is used up). A sample with ||z|| < t is lengthened to ||z|| = 2 t - ||z|| (z and y are scaled). Keeps the search global early on and turns into local search as the budget is used. After Piad-Morffis et al. (2015), who put the threshold on x - m as a fraction of the box diameter; on z it is relative to the search distribution and does not depend on the box.
    """

    def __init__(self) -> None:
        """
        Create threshold convergence with init_threshold = 1 and decay_factor = 0.995.
        """

    @property
    def init_threshold(self) -> float:
        """
        The threshold at the start of a run as a multiple of E||z|| (default 1).
        """

    @init_threshold.setter
    def init_threshold(self, arg: float, /) -> None: ...

    @property
    def decay_factor(self) -> float:
        """
        The exponent of the decrease of the threshold with the remaining budget (default 0.995).
        """

    @decay_factor.setter
    def decay_factor(self, arg: float, /) -> None: ...

    def threshold(self, expected_length: float, budget: int, evaluations: int) -> float:
        """
        The threshold after `evaluations` of `budget`.

        Parameters
        ----------
        expected_length : float
            E||z|| of the sampler (`es.p.sampler.expected_length()`).
        budget : int
            The evaluation budget.
        evaluations : int
            The evaluations used so far (more than the budget gives 0).

        Returns
        -------
        float
            init_threshold * expected_length * ((budget - evaluations) / budget)^decay_factor.
        """

    def apply(self, z: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], y: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], t: float) -> None:
        """
        Lengthen one sample in place: when ||z|| < t, z and y are scaled so that ||z|| = 2 t - ||z||.

        Parameters
        ----------
        z, y : numpy.ndarray
            The sample and its y = A z, shape (d,), float64; changed in place.
        t : float
            The threshold.
        """

class NoThresholdConvergence(ThresholdConvergence):
    """No threshold convergence (the default): samples are not scaled."""

    def __init__(self) -> None:
        """Create the no-op threshold convergence."""

class SequentialSelection:
    """
    Sequential selection: stop evaluating a population once it improves on the previous generation (`Modules.sequential_selection`).

    The population is cut after individual i (0-based) when its objective value is better than the best of the previous generation (in the first generation of a run, never) and i >= mu * seq_cutoff_factor; with pairwise selection the factor is at least 2 and the cut falls after a complete pair. The updates use the evaluated individuals. Saves evaluations on functions where improvements are easy to find. Brockhoff et al. (2010).
    """

    @overload
    def __init__(self, mirror: modcma.c_maes.cmaescpp.options.Mirror, seq_cutoff_factor: float = 1.0) -> None:
        """
        Create the sequential selection.

        Parameters
        ----------
        mirror : options.Mirror
            The mirrored sampling option; with `Mirror.PAIRWISE` the cutoff factor is at least 2.
        seq_cutoff_factor : float
            The minimum number of evaluated individuals as a multiple of mu (default 1).
        """

    @overload
    def __init__(self, mirror: modcma.c_maes.cmaescpp.options.Mirror, *, seq_cuttoff_factor: float) -> None:
        """Deprecated: the old (misspelled) keyword of 1.x."""

    def break_conditions(self, i: int, f: float, fopt: float, mirror: modcma.c_maes.cmaescpp.options.Mirror, mu: int) -> bool:
        """
        Whether the population stops after individual i.

        Parameters
        ----------
        i : int
            The index of the individual just evaluated (0-based).
        f : float
            Its objective value.
        fopt : float
            The reference value: the best of the previous generation.
        mirror : options.Mirror
            The mirrored sampling option.
        mu : int
            The number of selected individuals.

        Returns
        -------
        bool
            True when f < fopt, i >= mu * seq_cutoff_factor and, with pairwise selection, i is odd.
        """

class NoSequentialSelection(SequentialSelection):
    """
    No sequential selection (the default): every individual of a population is evaluated.
    """

    @overload
    def __init__(self, mirror: modcma.c_maes.cmaescpp.options.Mirror, seq_cutoff_factor: float = 1.0) -> None:
        """
        Create the no-op sequential selection.

        Parameters
        ----------
        mirror : options.Mirror
            The mirrored sampling option (not used).
        seq_cutoff_factor : float
            Not used (default 1).
        """

    @overload
    def __init__(self, mirror: modcma.c_maes.cmaescpp.options.Mirror, *, seq_cuttoff_factor: float) -> None:
        """Deprecated: the old (misspelled) keyword of 1.x."""

class SigmaSampler:
    """
    Sampled step sizes: each individual gets its own step size per coordinate, sigma * exp(beta n) with n standard normal (`Modules.sample_sigma`; always used by the step-size adaptations LPXNES and SA).

    beta = `es.p.weights.beta` = 1 / sqrt(2 mueff) (ln 2 / (sqrt(d) ln d) with LPXNES). The step sizes are the columns of `es.p.pop.S`.
    """

    def __init__(self, dimension: float) -> None:
        """
        Create the sampler.

        Parameters
        ----------
        dimension : float
            The dimension d.
        """

    def apply_integer_bounds(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Raise the step sizes of the integer variables to their lower bound int_lb_sigma / sqrt(C_ii) (`es.p.weights.int_lb_sigma`, C_ii from `adaptation.coordinate_wise_variances`).

        Parameters
        ----------
        parameters : Parameters
            The state of the optimizer; `pop.S` is changed in place.
        """

    def sample(self, sigma: float, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Set the step sizes `pop.S` of the population from sigma, then apply the lower bounds of the integer variables.

        Parameters
        ----------
        sigma : float
            The global step size.
        parameters : Parameters
            The state of the optimizer; `pop.S` is written.
        """

class NoSigmaSampler(SigmaSampler):
    """
    No sampled step sizes (the default): every individual uses sigma (apart from the lower bounds of integer variables).
    """

    def __init__(self, dimension: float) -> None:
        """
        Create the sampler.

        Parameters
        ----------
        dimension : float
            The dimension d.
        """

class Strategy:
    """
    Base class of the step-size adaptations; also samples and evaluates a population.

    The optimizer samples a population through it, in batches, and `adapt` updates sigma after selection. The base class does not change sigma. The optimizer creates the subclass of `Modules.ssa`; each subclass has the same constructor.
    """

    def __init__(self, threshold_convergence: ThresholdConvergence, sequential_selection: SequentialSelection, sigma_sampler: SigmaSampler, sigma0: float) -> None:
        """
        Create a strategy that does not adapt sigma.

        Parameters
        ----------
        threshold_convergence : ThresholdConvergence
            The threshold convergence (or `NoThresholdConvergence`).
        sequential_selection : SequentialSelection
            The sequential selection (or `NoSequentialSelection`).
        sigma_sampler : SigmaSampler
            The sampler of the step sizes (or `NoSigmaSampler`).
        sigma0 : float
            The initial step size.
        """

    def adapt(self, weights: modcma.c_maes.cmaescpp.parameters.Weights, dynamic: modcma.c_maes.cmaescpp.matrix_adaptation.Adaptation, population: modcma.c_maes.cmaescpp.Population, old_population: modcma.c_maes.cmaescpp.Population, stats: modcma.c_maes.cmaescpp.parameters.Stats, lamb: int) -> None:
        """
        Update the step size sigma after selection.

        Parameters
        ----------
        weights : Weights
            The recombination weights and learning rates (`es.p.weights`).
        dynamic : matrix_adaptation.Adaptation
            The matrix adaptation (`es.p.adaptation`).
        population : Population
            The selected population of this generation.
        old_population : Population
            The population of the previous generation.
        stats : Stats
            The statistics of the run.
        lamb : int
            The population size lambda.
        """

    @property
    def threshold_convergence(self) -> ThresholdConvergence:
        """
        The threshold convergence (`ThresholdConvergence` or `NoThresholdConvergence`).
        """

    @threshold_convergence.setter
    def threshold_convergence(self, arg: ThresholdConvergence, /) -> None: ...

    @property
    def sequential_selection(self) -> SequentialSelection:
        """
        The sequential selection (`SequentialSelection` or `NoSequentialSelection`).
        """

    @sequential_selection.setter
    def sequential_selection(self, arg: SequentialSelection, /) -> None: ...

    @property
    def sigma_sampler(self) -> SigmaSampler:
        """The sampler of the step sizes (`SigmaSampler` or `NoSigmaSampler`)."""

    @sigma_sampler.setter
    def sigma_sampler(self, arg: SigmaSampler, /) -> None: ...

    @property
    def sigma(self) -> float:
        """The global step size sigma."""

    @sigma.setter
    def sigma(self, arg: float, /) -> None: ...

    @property
    def generation_threshold(self) -> float:
        """
        The threshold of threshold convergence in the current generation, computed at its start (`ThresholdConvergence.threshold`; also set without the module).
        """

    @property
    def s(self) -> float:
        """
        The smoothed success measure of TPA, MSR and PSR, whose exponential scales sigma (default 0).
        """

    @s.setter
    def s(self, arg: float, /) -> None: ...

class CSA(Strategy):
    """
    Cumulative step-size adaptation (`StepSizeAdaptation.CSA`, the default).

    sigma *= exp(l), l = (cs / damps) (||ps|| / E||z|| - 1) clamped to [-1, 1] (as in pycma): sigma grows when the evolution path ps is longer than under random selection and shrinks when it is shorter. Hansen & Ostermeier (2001).
    """

    def __init__(self, threshold_convergence: ThresholdConvergence, sequential_selection: SequentialSelection, sigma_sampler: SigmaSampler, sigma0: float) -> None:
        """
        Create the step-size adaptation (the optimizer creates it from `Modules.ssa`).

        Parameters
        ----------
        threshold_convergence : ThresholdConvergence
            The threshold convergence (or `NoThresholdConvergence`).
        sequential_selection : SequentialSelection
            The sequential selection (or `NoSequentialSelection`).
        sigma_sampler : SigmaSampler
            The sampler of the step sizes (or `NoSigmaSampler`).
        sigma0 : float
            The initial step size.
        """

class TPA(Strategy):
    """
    Two-point step-size adaptation (`StepSizeAdaptation.TPA`).

    After each population the two points m + sigma dm and m - sigma dm (along the last step of the mean) are evaluated; s = (1 - cs) s + cs r, with r = -a_tpa when m - sigma dm is better, a_tpa + b_tpa when m + sigma dm is better and 0 for a tie (as after a restart, when dm = 0), and sigma *= exp(s). Costs two evaluations per generation. Hansen (2008); a tie as in pycma.
    """

    def __init__(self, threshold_convergence: ThresholdConvergence, sequential_selection: SequentialSelection, sigma_sampler: SigmaSampler, sigma0: float) -> None:
        """
        Create the step-size adaptation (the optimizer creates it from `Modules.ssa`).

        Parameters
        ----------
        threshold_convergence : ThresholdConvergence
            The threshold convergence (or `NoThresholdConvergence`).
        sequential_selection : SequentialSelection
            The sequential selection (or `NoSequentialSelection`).
        sigma_sampler : SigmaSampler
            The sampler of the step sizes (or `NoSigmaSampler`).
        sigma0 : float
            The initial step size.
        """

    @property
    def a_tpa(self) -> float:
        """The magnitude of r (default 0.5)."""

    @a_tpa.setter
    def a_tpa(self, arg: float, /) -> None: ...

    @property
    def b_tpa(self) -> float:
        """The offset added to r when the forward point is better (default 0)."""

    @b_tpa.setter
    def b_tpa(self, arg: float, /) -> None: ...

    @property
    def rank_tpa(self) -> float:
        """The last r: -a_tpa, a_tpa + b_tpa or 0 (default 0)."""

    @rank_tpa.setter
    def rank_tpa(self, arg: float, /) -> None: ...

class MSR(Strategy):
    """
    Median success rule (`StepSizeAdaptation.MSR`).

    With k the number of individuals better than the j-th best of the previous population, j = quantile * lambda (0-based, interpolated between its neighbours, as pycma): s = (1 - cs) s + cs (2 / lambda) (k - lambda / 2) and sigma *= exp(s / damps), with cs = 0.3 and damps = 2 - 2 / d (1 for d = 1) by default: sigma grows when the median offspring beats the j-th best of the previous generation. Ait ElHara, Auger & Hansen (2013).
    """

    def __init__(self, threshold_convergence: ThresholdConvergence, sequential_selection: SequentialSelection, sigma_sampler: SigmaSampler, sigma0: float) -> None:
        """
        Create the step-size adaptation (the optimizer creates it from `Modules.ssa`).

        Parameters
        ----------
        threshold_convergence : ThresholdConvergence
            The threshold convergence (or `NoThresholdConvergence`).
        sequential_selection : SequentialSelection
            The sequential selection (or `NoSequentialSelection`).
        sigma_sampler : SigmaSampler
            The sampler of the step sizes (or `NoSigmaSampler`).
        sigma0 : float
            The initial step size.
        """

    @property
    def quantile(self) -> float:
        """
        The position j / lambda of the previous generation's value that the offspring are compared with (default 0.3).
        """

    @quantile.setter
    def quantile(self, arg: float, /) -> None: ...

class PSR(Strategy):
    """
    Population success rule (`StepSizeAdaptation.PSR`).

    The current and the previous population are ranked together; with z the difference of their summed ranks (previous minus current) divided by n^2, minus success_ratio: s = (1 - cs) s + cs z and sigma *= exp(s / damps), with cs = 0.3 and damps = 1 by default (Loshchilov 2014, Alg. 6).
    """

    def __init__(self, threshold_convergence: ThresholdConvergence, sequential_selection: SequentialSelection, sigma_sampler: SigmaSampler, sigma0: float) -> None:
        """
        Create the step-size adaptation (the optimizer creates it from `Modules.ssa`).

        Parameters
        ----------
        threshold_convergence : ThresholdConvergence
            The threshold convergence (or `NoThresholdConvergence`).
        sequential_selection : SequentialSelection
            The sequential selection (or `NoSequentialSelection`).
        sigma_sampler : SigmaSampler
            The sampler of the step sizes (or `NoSigmaSampler`).
        sigma0 : float
            The initial step size.
        """

    @property
    def success_ratio(self) -> float:
        """The target success ratio (default 0.25)."""

    @success_ratio.setter
    def success_ratio(self, arg: float, /) -> None: ...

class XNES(Strategy):
    """
    The step-size update of xNES (`StepSizeAdaptation.XNES`).

    With NATURAL_GRADIENT: sigma *= exp(cs / 2 * sigma_g), sigma_g the trace part of the natural gradient (`NaturalGradientAdaptation.sigma_g`). With the other matrix adaptations: sigma *= exp(cs / sqrt(d) * sum_i w_i (||z_i||^2 - d)) over the positive weights. Glasmachers et al. (2010).
    """

    def __init__(self, threshold_convergence: ThresholdConvergence, sequential_selection: SequentialSelection, sigma_sampler: SigmaSampler, sigma0: float) -> None:
        """
        Create the step-size adaptation (the optimizer creates it from `Modules.ssa`).

        Parameters
        ----------
        threshold_convergence : ThresholdConvergence
            The threshold convergence (or `NoThresholdConvergence`).
        sequential_selection : SequentialSelection
            The sequential selection (or `NoSequentialSelection`).
        sigma_sampler : SigmaSampler
            The sampler of the step sizes (or `NoSigmaSampler`).
        sigma0 : float
            The initial step size.
        """

class MXNES(Strategy):
    """
    A variant of the xNES step-size update on the mean step (`StepSizeAdaptation.MXNES`).

    sigma *= exp((cs / d) (mueff ||dz||^2 - d)), with dz the weighted mean of the selected z: sigma grows when the mean step is longer than under random selection. m-xNES of Krause, Glasmachers & Igel (2017), "Qualitative and quantitative assessment of step size adaptation rules".
    """

    def __init__(self, threshold_convergence: ThresholdConvergence, sequential_selection: SequentialSelection, sigma_sampler: SigmaSampler, sigma0: float) -> None:
        """
        Create the step-size adaptation (the optimizer creates it from `Modules.ssa`).

        Parameters
        ----------
        threshold_convergence : ThresholdConvergence
            The threshold convergence (or `NoThresholdConvergence`).
        sequential_selection : SequentialSelection
            The sequential selection (or `NoSequentialSelection`).
        sigma_sampler : SigmaSampler
            The sampler of the step sizes (or `NoSigmaSampler`).
        sigma0 : float
            The initial step size.
        """

class LPXNES(Strategy):
    """
    A variant of the xNES step-size update on sampled step sizes (`StepSizeAdaptation.LPXNES`).

    Every individual samples its own step sizes (a `SigmaSampler` is always used, with beta = ln 2 / (sqrt(d) ln d)); sigma *= exp(cs sum_i w_i mean_j ln(S_ij / sigma)) over the positive weights, which moves sigma towards the step sizes of the better individuals. lp-xNES of Krause, Glasmachers & Igel (2017), "Qualitative and quantitative assessment of step size adaptation rules".
    """

    def __init__(self, threshold_convergence: ThresholdConvergence, sequential_selection: SequentialSelection, sigma_sampler: SigmaSampler, sigma0: float) -> None:
        """
        Create the step-size adaptation (the optimizer creates it from `Modules.ssa`).

        Parameters
        ----------
        threshold_convergence : ThresholdConvergence
            The threshold convergence (or `NoThresholdConvergence`).
        sequential_selection : SequentialSelection
            The sequential selection (or `NoSequentialSelection`).
        sigma_sampler : SigmaSampler
            The sampler of the step sizes (or `NoSigmaSampler`).
        sigma0 : float
            The initial step size.
        """

class SR(Strategy):
    """
    The success rule of the (1+1)-CMA-ES (`StepSizeAdaptation.SR`; always used when lambda = 1).

    sigma *= exp((p_s - p_t) / (damps (1 - p_t))), with p_s the smoothed rate of successes (`es.p.stats.success_ratio`): a generation is a success when its best individual is better than the best of the run so far (with elitism, the parent). The target p_t = 2/11 and damps = 1 + d / 2 by default. Igel et al. (2006).

    The rule is designed for elitist selection, where the parent is the best point so far. Without elitism, the best of the run takes the parent's role: comparing with the previous generation's best instead would count a success about every other generation even at a good sigma, and sigma would diverge. For non-elitist selection see also `MSR`.
    """

    def __init__(self, threshold_convergence: ThresholdConvergence, sequential_selection: SequentialSelection, sigma_sampler: SigmaSampler, sigma0: float) -> None:
        """
        Create the step-size adaptation (the optimizer creates it from `Modules.ssa`).

        Parameters
        ----------
        threshold_convergence : ThresholdConvergence
            The threshold convergence (or `NoThresholdConvergence`).
        sequential_selection : SequentialSelection
            The sequential selection (or `NoSequentialSelection`).
        sigma_sampler : SigmaSampler
            The sampler of the step sizes (or `NoSigmaSampler`).
        sigma0 : float
            The initial step size.
        """

class SA(Strategy):
    """
    Self-adaptation of the step size (`StepSizeAdaptation.SA`).

    Every individual samples its own step sizes (a `SigmaSampler` is always used); sigma is multiplied by the weighted geometric mean of the step sizes of the mu selected individuals divided by the geometric mean over the whole population (Beyer & Sendhoff 2008, with the correction of Ahrari et al. 2017). With SA the default mu is lambda / 4.
    """

    def __init__(self, threshold_convergence: ThresholdConvergence, sequential_selection: SequentialSelection, sigma_sampler: SigmaSampler, sigma0: float) -> None:
        """
        Create the step-size adaptation (the optimizer creates it from `Modules.ssa`).

        Parameters
        ----------
        threshold_convergence : ThresholdConvergence
            The threshold convergence (or `NoThresholdConvergence`).
        sequential_selection : SequentialSelection
            The sequential selection (or `NoSequentialSelection`).
        sigma_sampler : SigmaSampler
            The sampler of the step sizes (or `NoSigmaSampler`).
        sigma0 : float
            The initial step size.
        """
