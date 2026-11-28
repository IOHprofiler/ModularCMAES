"""
Mutation: the sampling of a population and the step-size adaptation.

The optimizer holds a `Strategy` subclass as `es.p.mutation`, selected by `Modules.ssa` (`options.StepSizeAdaptation`) and created anew at every restart. It holds the threshold convergence (`Modules.threshold_convergence`), the sequential selection (`Modules.sequential_selection`) and the sampler of the step sizes (`Modules.sample_sigma`). The step-size adaptations are described in de Nobel et al. (2021); `TPAInPlace` (TPA_INPLACE, new in 2.0) in its own docstring.
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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class NoThresholdConvergence(ThresholdConvergence):
    """No threshold convergence (the default): samples are not scaled."""

    def __init__(self) -> None:
        """Create the no-op threshold convergence."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class SequentialSelection:
    """
    Sequential selection: stop evaluating a population once it improves on the previous generation (`Modules.sequential_selection`).

    The population is cut after individual i (0-based) when its objective value is better than the best of the previous generation (in the first generation of a run, never) and i >= floor(mu * seq_cutoff_factor); with pairwise selection the factor is at least 2 and the cut falls after a complete pair. The updates use the evaluated individuals. Saves evaluations on functions where improvements are easy to find. Brockhoff et al. (2010).

    Create it as SequentialSelection(mirror, seq_cutoff_factor=1.0), with mirror the `Modules.mirrored` option; the optimizer creates it with the defaults. `NoSequentialSelection` (the module off) takes the same arguments.
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
            The population can stop after individual i >= floor(mu * seq_cutoff_factor) (0-based), so at least floor(mu * seq_cutoff_factor) + 1 individuals are evaluated (default 1).
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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class NoSequentialSelection(SequentialSelection):
    """
    No sequential selection (the default): every individual of a population is evaluated.

    Create it as NoSequentialSelection(mirror, seq_cutoff_factor=1.0); neither argument is used.
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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class SigmaSampler:
    """
    Sampled step sizes: each individual gets its own step size, sigma_i = sigma * exp(beta n_i) with n_i standard normal, the same in every coordinate, so the point m + sigma_i y_i lies along y_i (`Modules.sample_sigma`; always used by the step-size adaptations LPXNES and SA). One step size per individual, as in CMSA-ES (Beyer & Sendhoff 2008) and lp-xNES (Krause, Glasmachers & Igel 2017, Sec. 4.4).

    beta = `es.p.weights.beta`: 1 / sqrt(2 d) with SA (CMSA's tau; Spettel, Beyer & Hellwig 2018, Table I), ln 2 / (sqrt(d) ln(max(d, 2))) with LPXNES (Krause et al. 2017, Table 3), 1 / sqrt(2 mueff) otherwise. The step sizes are the columns of `es.p.pop.S`; the lower bound of an integer variable can raise single coordinates.
    """

    def __init__(self, dimension: float) -> None:
        """
        Create the sampler. It draws from the global generator (`set_seed`); the sampler of an optimizer draws from the optimizer's own.

        Parameters
        ----------
        dimension : float
            The dimension d (not used: the step size is one number per individual).
        """

    def apply_integer_bounds(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Raise the step sizes of the integer variables to their lower bound min(0.2, mueff / d) / sqrt(C_ii) (C_ii from `adaptation.coordinate_wise_variances`).

        Parameters
        ----------
        parameters : Parameters
            The state of the optimizer; `pop.S` is changed in place.
        """

    def sample(self, sigma: float, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Set the step sizes `pop.S` of the population from sigma (one per individual, constant in its column), then apply the lower bounds of the integer variables.

        Parameters
        ----------
        sigma : float
            The global step size.
        parameters : Parameters
            The state of the optimizer; `pop.S` is written.
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
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
            The dimension d (not used).
        """

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
    Base class of the step-size adaptations; also samples the population.

    The optimizer samples a population through it, in batches, passes it the objective values, and `adapt` updates sigma after selection. The base class does not change sigma. The optimizer creates the subclass of `Modules.ssa`; each subclass has the same constructor.
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

    def adapt(self, weights: modcma.c_maes.cmaescpp.parameters.Weights, dynamic: modcma.c_maes.cmaescpp.matrix_adaptation.Adaptation, population: modcma.c_maes.cmaescpp.Population, old_population: modcma.c_maes.cmaescpp.Population, stats: modcma.c_maes.cmaescpp.parameters.Stats) -> None:
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
        The smoothed success measure of TPA, TPA_INPLACE, MSR and PSR (default 0): TPA multiplies sigma by exp(s), the others by exp(s / damps).
        """

    @s.setter
    def s(self, arg: float, /) -> None: ...

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class CSA(Strategy):
    """
    Cumulative step-size adaptation (`StepSizeAdaptation.CSA`, the default).

    sigma *= exp(l), l = (cs / damps) (||ps|| / E||z|| - 1) clamped to [-1, 1] (as in pycma): sigma grows when the evolution path ps is longer than under random selection and shrinks when it is shorter. E||z|| is the expected length of a sample of the sampler (`Weights.expected_length_z`). By default cs = (mueff + 2) / (d + mueff + 5) and damps = 1 + 2 max(0, sqrt((mueff - 1) / (d + 1)) - 1) + cs (Hansen 2016, Eqs. (44), (55)); modcma 1.1 and 1.2 used pycma's denominator d + mueff + 3 for cs (#138). Hansen & Ostermeier (2001).
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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class TPA(Strategy):
    """
    Two-point step-size adaptation (`StepSizeAdaptation.TPA`).

    With every population (in its first batch) the two points m + 0.5 sigma_old dm and m - 0.5 sigma_old dm are evaluated: half the last step of the mean, sigma_old dm, with the step size that sampled it (Hansen, CMA-ES with Two-Point Step-Size Adaptation, 2008, Eqs. (8), (9), alpha = 0.5). s = (1 - cs) s + cs r, with r = -a_tpa when the point behind is better, a_tpa + b_tpa when the point ahead is better and 0 for a tie (as after a restart, when dm = 0), and sigma *= exp(s), with cs = 0.3 by default (Hansen 2008, Eqs. (10) to (13); `damps` is not used). Costs two evaluations per generation. A tie gives r = 0, as the sign of the rank difference in pycma; Hansen (2008, Eq. (10)) counts it as a success, which made sigma grow by 16 % after every restart (#78). Before 2.0 the points were m +- sigma dm with the new sigma, about one step of the mean (#167). With a bound correction other than NONE both points move toward m by the same factor until both are in the box, so the test steps stay symmetric about m (Hansen 2008, Sec. 2.2; pycma's boundary handler moves each point on its own); `BoundCorrection.correct_point` corrects them only when m itself is outside the box.

    With elitism (#179): after a generation whose (mu + lambda) selection kept every parent (none of the mu selected points is an offspring), r = -a_tpa and the two points are not evaluated: the mean did not move, so both would be m up to rounding, and their comparison by rounding made sigma diverge. The test of the mean shift of generation t is made in generation t + 1, and so is this failure. The budget still reserves the two points in that generation. No TPA source covers elitism: Hansen (2008) and Hansen, Atamna & Auger (2014, Algorithm 3) define TPA for comma selection, and pycma's CMA_elitist adds only the best solution so far, so the mean always moves. The rule follows the success rules of elitist ESs (Rechenberg's 1/5th rule; the (1+1)-CMA-ES, Igel, Suttorp & Hansen 2006), in which a generation without a successful offspring is a failure.
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
        """
        The offset added to r when the forward point is better (default 0). Hansen (2008, Eq. (10)) adds his beta to the other case, r = -a_tpa + beta; with the default 0 the two agree.
        """

    @b_tpa.setter
    def b_tpa(self, arg: float, /) -> None: ...

    @property
    def rank_tpa(self) -> float:
        """
        The last r: -a_tpa, a_tpa + b_tpa or 0 (default 0); -a_tpa also after a generation whose elitist selection kept every parent.
        """

    @rank_tpa.setter
    def rank_tpa(self, arg: float, /) -> None: ...

    @property
    def sigma_old(self) -> float:
        """
        The step size that sampled the last generation, before the update changed it: the test steps are 0.5 sigma_old dm. At the start of a run, the step size the run starts with (sigma0, or the step size of the restart).
        """

    @sigma_old.setter
    def sigma_old(self, arg: float, /) -> None: ...

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class MSR(Strategy):
    """
    Median success rule (`StepSizeAdaptation.MSR`).

    With k the number of individuals better than the j-th best of the previous population, j = quantile * lambda (0-based, interpolated between its neighbours, as pycma): s = (1 - cs) s + cs (2 / lambda) (k - lambda / 2) and sigma *= exp(s / damps), with cs = 0.3 and damps = 2 - 2 / d (1 for d = 1) by default: sigma grows when the median offspring beats the j-th best of the previous generation. Ait ElHara, Auger & Hansen (2013).

    lambda is the number of ranked individuals of each generation: fewer after sequential selection stopped it, and with PAIRWISE only the winners of the mirrored pairs (Auger et al. 2011, Alg. 1).
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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class PSR(Strategy):
    """
    Population success rule (`StepSizeAdaptation.PSR`).

    The n best individuals of the current and of the previous population are ranked together, n the smaller number of ranked individuals of the two (fewer than lambda after a stop of sequential selection; with PAIRWISE the winners of the mirrored pairs). With z the difference of their summed ranks (previous minus current) divided by n^2, minus success_ratio: s = (1 - cs) s + cs z and sigma *= exp(s / damps), with cs = 0.3 and damps = 1 by default (Loshchilov 2014, Alg. 6). Equal values share their mean rank (mid-ranks; the paper does not say), so a tie counts for neither population; with elitism the parents are in both.
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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class XNES(Strategy):
    """
    The step-size update of xNES (`StepSizeAdaptation.XNES`).

    With NATURAL_GRADIENT: sigma *= exp(cs / 2 * sigma_g), sigma_g the trace part of the natural gradient (`NaturalGradientAdaptation.sigma_g`), cs = xNES's eta_sigma = (9 + 3 ln(d)) / (5 d sqrt(d)) (Glasmachers et al. 2010). With the other matrix adaptations, the xNES SA of Krause, Glasmachers & Igel (2017): sigma *= exp(cs / sqrt(d) * sum_i w_i (||z_i||^2 - d)) over the positive weights (`Weights.positive`, summing to 1), cs = mueff / (2 ln(max(2, d)) sqrt(d)).

    With elitism both forms read the offspring alone, ranked among themselves (the losers of pairwise selection last), with the same weights: xNES's gradient and xNES SA are estimates from the samples of the current distribution (Glasmachers et al. 2010, Algorithm 1; Krause et al. 2017, Sec. 4.2). The parents' z, recomputed against the mean they recombined, are short, and as part of the mu best they kept sigma 3 to 14 times too small (#172).
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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class MXNES(Strategy):
    """
    A variant of the xNES step-size update on the mean step (`StepSizeAdaptation.MXNES`).

    sigma *= exp((cs / d) (mueff ||dz||^2 - d)), with dz the weighted mean of the selected z (`Adaptation.dz`) and cs = 1 by default (Krause et al. 2017, Sec. 4.3, Table 3): sigma grows when the mean step is longer than under random selection. With UTILITY weights dz recombines all lambda z with their utilities, and mueff is theirs (`Weights.mueff_mean`). m-xNES of Krause, Glasmachers & Igel (2017), "Qualitative and quantitative assessment of step size adaptation rules".
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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class LPXNES(Strategy):
    """
    A variant of the xNES step-size update on sampled step sizes (`StepSizeAdaptation.LPXNES`).

    Every individual samples its own step size sigma_i (a `SigmaSampler` is always used, with beta = ln 2 / (sqrt(d) ln(max(d, 2)))); sigma *= exp(cs sum_i w_i ln(sigma_i / sigma)) over the positive weights (`Weights.positive`, summing to 1; ln sigma_i: the mean of the logarithms of the column of `pop.S`), which moves sigma towards the step sizes of the better individuals; cs = min(1, 9 mueff / (10 sqrt(d))). The rate is that of Krause et al.; the cap is modcma's: above 1 the update overshoots the weighted geometric mean of the step sizes (1.29 at d = 2, 1.05 at d = 5 with the default lambda). lp-xNES of Krause, Glasmachers & Igel (2017), Sec. 4.4, "Qualitative and quantitative assessment of step size adaptation rules".
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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class SR(Strategy):
    """
    The success rule of the (1+1)-CMA-ES (`StepSizeAdaptation.SR`; always used when lambda = 1).

    sigma *= exp((p_s - p_t) / (damps (1 - p_t))), with p_s the rate of successes, smoothed with weight 1/12 (`es.p.stats.success_ratio`): a generation is a success when its best individual is better than the best of the run so far (with elitism, the parent). The target p_t = 2/11 and damps = 1 + d / 2 by default. Igel et al. (2006).

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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class SA(Strategy):
    """
    Self-adaptation of the step size (`StepSizeAdaptation.SA`).

    Every individual samples its own step size sigma_i = sigma exp(n_i / sqrt(2 d)) (a `SigmaSampler` is always used; CMSA's tau = 1 / sqrt(2 d)); sigma is multiplied by the weighted geometric mean of the step sizes of the mu selected individuals (`Weights.positive`) divided by the geometric mean of the step sizes of all offspring of the generation, taken before elitism adds the parents (the CMSA-ES operator of Beyer & Sendhoff 2008, weighted, with the correction for elitism of Ahrari et al. 2017, as the framework paper describes). The step size of an individual is the geometric mean of its column of `pop.S`; a parent of elitism has the generation's sigma. With SA the default mu is lambda / 4.
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

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class TPAInPlace(Strategy):
    """
    Two-point step-size adaptation within the population (`StepSizeAdaptation.TPA_INPLACE`, #173).

    pycma 4.6.0's CMAAdaptSigmaTPA (sigma_adaptation.py, evolution_strategy.py:2251-2254, 2575-2576), after Hansen, Atamna & Auger (2014), How to Assess Step-Size Adaptation Mechanisms in Randomised Search, Eq. (6) and Algorithm 3. From the second generation of a run on (in a population of at least two), the first two offspring are a mirrored pair along the last mean shift: z_1 = ||n|| A^-1 dm / ||A^-1 dm|| with n an ordinary sample of the sampler, z_2 = -z_1 (both at the mean when it did not move); they are part of the lambda offspring, so TPA_INPLACE needs no extra evaluations. Once the offspring are evaluated, z = (#{f < f_2} - #{f < f_1}) / (n - 1) over the n offspring (the parents of elitism not included), and s = (1 - cs) s + cs sign(z) |z|^0.5, sigma *= exp(s / damps), with cs = 0.3 and damps = 0.7 + 2 ln(d) + 2 ln(max(1, lambda - d)) by default.

    Where pycma departs from the paper, and modcma follows pycma: |z|^0.5 instead of z; the damping (the paper: sqrt(d), tuned on the sphere with the default lambda); and the update in the first generation of a run, from the ranks of two ordinary samples (the paper updates from the second generation on). A^-1 dm costs O(d^2) with most matrix adaptations (a triangular solve, or B^T and D); MATRIX keeps M^-1 up to date for it, as with elitism; NATURAL_GRADIENT inverts A every generation (O(d^3)). With sampled step sizes the second point takes the step size of the first; bound correction and repelling treat the two as a pair, as a mirrored pair.

    With elitism (#179): after a generation whose (mu + lambda) selection kept every parent (none of the mu selected points is an offspring), the mean did not move, and the pair would lie along its rounding noise: the generation samples ordinary offspring instead, and z = -1. The pair for the mean shift of generation t is placed in generation t + 1, and so is this failure. No source covers elitism: Hansen, Atamna & Auger (2014, Algorithm 3) define TPA for the (mu/mu, lambda)-ES, and pycma's CMA_elitist adds only the best solution so far, so its mean always moves. The rule follows the success rules of elitist ESs (Rechenberg's 1/5th rule; the (1+1)-CMA-ES, Igel, Suttorp & Hansen 2006).
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
    def rank_difference(self) -> float:
        """
        The last normalised rank difference z of the two test points, in [-1, 1] (positive when the point ahead, along the mean shift, is the better one; 0 before the first update).
        """

    @rank_difference.setter
    def rank_difference(self, arg: float, /) -> None: ...

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """
