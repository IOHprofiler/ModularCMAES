"""
The options of the modules: one enum per module family.

A `parameters.Modules` object holds one value of each (and the on/off modules as bools); the options combine freely. The default of every enum gives the CMA-ES with cumulative step-size adaptation.
"""

import enum


class RecombinationWeights(enum.Enum):
    """
    The recombination weights (`Modules.weights`; default DEFAULT).

    The weights of the mu best individuals are positive and sum to 1; with `Modules.active`, the other lambda - mu individuals get negative weights.
    """

    DEFAULT = 0
    """
    Log-linear weights, the default: w_i proportional to ln((lambda + 1) / floor(lambda / mu)) - ln(i); with mu = lambda / 2 the weights of the standard CMA-ES (Hansen 2016, "The CMA Evolution Strategy: A Tutorial").
    """

    EQUAL = 1
    """
    Equal weights 1/mu (intermediate recombination): every selected individual counts the same.
    """

    EXPONENTIAL = 2
    """
    Exponentially decaying weights, w_i = 2^-i + 1 / (mu 2^mu), which sum to 1: they halve with every rank, more weight on the best individuals than DEFAULT (de Nobel et al. 2026); the negative ones the same for lambda - mu.
    """

    UTILITY = 3
    """
    The utilities of NES and xNES (Wierstra et al. 2014; Glasmachers et al. 2010), u_k = max(0, ln(lambda/2 + 1) - ln k) / sum - 1/lambda: log-shaped for the better half, minus a baseline, summing to zero. Unlike the other options they are not only the weights of the mu parents: the mean moves by the utility-weighted sum of all lambda steps, and every matrix update uses all of them (so `Modules.active` is set). With NATURAL_GRADIENT and StepSizeAdaptation.XNES this is xNES.
    """

class BaseSampler(enum.Enum):
    """
    The source of the uniform numbers in (0, 1)^d that become the samples (`Modules.sampler`; default UNIFORM).

    The `SampleTranformerType` (`Modules.sample_transformation`) maps them to the distribution of the mutations, by default the standard normal one. SOBOL and HALTON are quasi-random (low-discrepancy) sequences: their points cover the space more evenly than random ones (de Nobel et al. 2024, "Sampling in CMA-ES: low numbers of low discrepancy points").
    """

    UNIFORM = 0
    """
    Pseudo-random uniform numbers (mt19937, the optimizer's own generator), the default. With the GAUSSIAN transformation: ordinary random normal samples, which `Settings.fast_sampling` replaces by a faster generator.
    """

    SOBOL = 1
    """
    Points of the Sobol sequence (Sobol' 1967). A cache of points (see `constants.cache_max_doubles`) is computed when the sampler is created and drawn in a random order.
    """

    HALTON = 2
    """
    Points of the Halton sequence (Halton 1960), scrambled with random digit permutations.
    """

class SampleTranformerType(enum.Enum):
    """
    The distribution of the mutation vectors (`Modules.sample_transformation`; default GAUSSIAN).

    Each coordinate of a uniform sample of the `BaseSampler` is mapped through the inverse CDF of a symmetric distribution. Except for CAUCHY, the distributions have unit variance. de Nobel et al. (2025), "Abnormal mutations: evolution strategies don't require Gaussianity" (doi:10.1145/3712256.3726479).
    """

    NONE = 0
    """
    No transformation: the uniform numbers of the base sampler, in (0, 1) with mean 0.5 (not centred at 0; SCALED_UNIFORM is the centred uniform distribution).
    """

    GAUSSIAN = 1
    """
    Standard normal samples (the inverse normal CDF, or Box-Muller with `constants.use_box_muller`), the default: the CMA-ES.
    """

    SCALED_UNIFORM = 2
    """Uniform in (-sqrt(3), sqrt(3)): bounded mutations with unit variance."""

    LAPLACE = 3
    """
    Laplace distribution with scale 1/sqrt(2) (unit variance): heavier tails than the normal distribution.
    """

    LOGISTIC = 4
    """
    Logistic distribution with scale sqrt(3)/pi (unit variance): slightly heavier tails than the normal distribution.
    """

    CAUCHY = 5
    """
    Standard Cauchy distribution (scale 1): very heavy tails and no finite variance; occasional long jumps.
    """

    DOUBLE_WEIBULL = 6
    """
    Double Weibull distribution with shape 2 and scale 1 (unit variance), +-sqrt(-ln(2u)): bimodal, with little mass near 0.
    """

class Mirror(enum.Enum):
    """
    Mirrored sampling (`Modules.mirrored`; default NONE).

    With mirroring, every second mutation vector is the negation of the one before it, which spreads the samples more evenly around the mean.
    """

    NONE = 0
    """Independent samples, the default."""

    MIRRORED = 1
    """
    Mirrored sampling (Brockhoff et al. 2010): the samples come in pairs z, -z.
    """

    PAIRWISE = 2
    """
    Mirrored sampling with pairwise selection (Auger et al. 2011): of each mirrored pair only the better one can be selected, so that the two do not cancel out in the recombination. Makes lambda even; with lambda0 = 1 (the (1+1)-ES) Settings turn it into NONE.
    """

class StepSizeAdaptation(enum.Enum):
    """
    The adaptation of the global step size sigma (`Modules.ssa`; default CSA).

    With the matrix adaptation COVARIANCE, any method other than CSA switches it to COVARIANCE_NO_EIGV (no eigendecomposition), unless `Settings(always_compute_eigv=True)`. A population of one (lambda = 1, the (1+1)-ES) always uses SR. For a comparison of the methods see de Nobel et al. 2021 (doi:10.1145/3449726.3463167).
    """

    CSA = 0
    """
    Cumulative step-size adaptation (Hansen & Ostermeier 2001), the default: sigma grows when the evolution path p_sigma is longer than expected under random selection and shrinks when it is shorter.
    """

    TPA = 1
    """
    Two-point step-size adaptation (Hansen 2008): two extra points m +- sigma * dm along the last step dm of the mean are evaluated every generation (two more evaluations); sigma grows when the point ahead is the better one.
    """

    MSR = 2
    """
    Median success rule (Ait ElHara et al. 2013): counts the individuals that are better than the 0.3 lambda-th best of the previous generation; sigma grows when more than half are.
    """

    XNES = 3
    """
    The step-size update of xNES (Glasmachers et al. 2010), from the weighted squared lengths of the samples; with NATURAL_GRADIENT, from its natural gradient.
    """

    MXNES = 4
    """
    A variant of the xNES update from the length of the weighted mean of the selected samples, mueff ||sum w_i z_i||^2 - d, instead of their individual lengths (m-xNES, Krause, Glasmachers & Igel (2017), "Qualitative and quantitative assessment of step size adaptation rules").
    """

    LPXNES = 5
    """
    An xNES-like update from the logarithms of the individual step sizes of the samples, weighted by rank; samples a step size per individual (as `Modules.sample_sigma`) (lp-xNES, Krause, Glasmachers & Igel (2017), "Qualitative and quantitative assessment of step size adaptation rules").
    """

    PSR = 6
    """
    Population success rule (Loshchilov 2014, LM-CMA-ES): ranks the current and the previous generation together and adapts sigma towards a target success ratio of 0.25.
    """

    SR = 7
    """
    Success rule of the (1+1)-ES: sigma grows when the success ratio is above the target 2/11 and shrinks when it is below (Igel et al. 2006). Used automatically when lambda = 1. Designed for elitist selection; without elitism a success is measured against the best of the run, which plays the parent's role.
    """

    SA = 8
    """
    Self-adaptation: every individual gets its own step size, sampled log-normally around sigma, and sigma moves to the weighted geometric mean of the step sizes of the selected individuals, divided by that of the population (the CMSA-ES operator of Beyer & Sendhoff 2008, weighted, with the correction for elitism of Ahrari et al. 2017). Meant for CMSA; makes the default mu lambda / 4.
    """

class CorrectionMethod(enum.Enum):
    """
    What happens to a sample outside the box bounds (`Modules.bound_correction`; default NONE).

    The bounds are `Settings.lb` and `Settings.ub` (default [-5, 5] in every coordinate). The methods other than RESAMPLE move each coordinate that is out of bounds into [lb, ub]; the mutation vector of the sample is then recomputed from the corrected point, so the adaptation learns from the point that was evaluated. The methods are those of Caraffini, Kononova & Corne (2019), "Infeasibility and structural bias in differential evolution".

    Bounds may be infinite; each coordinate and each side is corrected separately, and an infinite bound is never crossed. MIRROR, COTN, UNIFORM_RESAMPLE and TOROIDAL need the width ub - lb: in a coordinate with one infinite bound they set a coordinate that crosses the finite bound to that bound, as SATURATE does.
    """

    NONE = 0
    """
    No correction, the default: points outside the bounds are evaluated as they are (and counted in `Parameters.bounds.n_out_of_bounds`).
    """

    MIRROR = 1
    """
    Reflect the coordinate at the violated bound (periodically, for points more than a domain width outside).
    """

    COTN = 2
    """
    Complete one-tailed normal correction: place the coordinate inside the violated bound, at a random distance sigma * |N(0, 1/3^2)| times the domain width ub - lb.
    """

    UNIFORM_RESAMPLE = 3
    """Replace the coordinate by a uniform random value in [lb, ub]."""

    SATURATE = 4
    """
    Set the coordinate to the violated bound (clipping): corrected points lie on the boundary.
    """

    TOROIDAL = 5
    """
    Wrap the coordinate around: leaving at one bound enters at the other (a torus).
    """

    RESAMPLE = 6
    """
    Sample the whole individual again until it lies inside the bounds, at most 5 d times; after that, the last sample is evaluated as it is.
    """

class RestartStrategy(enum.Enum):
    """
    When the optimizer restarts, and with which population size (`Modules.restart_strategy`; default NONE).

    A restart happens when a restart criterion (`Parameters.criteria`, module `restart`) is met: the mean is placed again (`CenterPlacement`), sigma is reset to sigma0 (BIPOP: possibly smaller) and the adaptation starts over. Except for NONE, the strategies use the full set of criteria (see `restart.Criteria`). With lambda = 1, IPOP and BIPOP become RESTART.
    """

    NONE = 0
    """
    No restart criteria other than `restart.UnableToAdapt`, the default: the optimizer only restarts (with the same population size) when its adaptation fails.
    """

    STOP = 2
    """
    Use the restart criteria as stopping criteria: the run ends when one of them is met, without a restart.
    """

    RESTART = 1
    """
    Restart with the same population size whenever a restart criterion is met.
    """

    IPOP = 3
    """
    IPOP-CMA-ES (Auger & Hansen 2005): every restart multiplies mu and lambda by `restart.IPOP.ipop_factor` (2). Larger populations help on multimodal functions with a global structure.
    """

    BIPOP = 4
    """
    BIPOP-CMA-ES (Hansen 2009): alternates between a large population, which doubles every time it is used, and a small one with a random size and a smaller initial sigma. Each restart uses the regime that has used fewer evaluations so far, so both get about half of the evaluations; a small run takes at most half of what the large regime has used. See `restart.BIPOP`.
    """

class MatrixAdaptationType(enum.Enum):
    """
    The adaptation of the shape of the search distribution (`Modules.matrix_adaptation`; default COVARIANCE).

    The mutations are y = A z, with z from the sampler and A a factor of the covariance matrix, C = A A^T. The options differ in how A is adapted and in their cost per generation (docs/performance.md, "ModularCMAES matrix adaptations"). C starts as the identity, or as `Settings.C0` (warm starting).
    """

    COVARIANCE = 1
    """
    The covariance matrix adaptation of the CMA-ES (Hansen & Ostermeier 2001), with rank-one and rank-mu update, the default. C is eigendecomposed every few generations (O(d^3), with LAPACK from `constants.lapack_min_dim` on). Keeps its metric (`Adaptation.metric()`), so the restart criteria ConditionC and NoEffectAxis are used, as with SEPARABLE and NONE.
    """

    NONE = 0
    """
    No matrix adaptation (C = I): only the step size is adapted. Disables `Modules.active`, rejects `Settings.C0`. O(d lambda) per generation.
    """

    MATRIX = 2
    """
    MA-ES (Beyer & Sendhoff 2017): adapts a matrix M with C = M M^T directly, without the evolution path p_c and without eigendecomposition; O(d^2 lambda) per generation, about 4x faster than COVARIANCE at d = 1000.
    """

    SEPARABLE = 3
    """
    sep-CMA-ES (Ros & Hansen 2008): adapts only the diagonal of C, O(d lambda) per generation. For separable problems and very high dimension; it cannot learn correlations between variables. Of `Settings.C0` it uses the diagonal.
    """

    CHOLESKY = 4
    """
    Cholesky-CMA-ES (Krause et al. 2016): adapts the lower triangular Cholesky factor of C with rank-one updates of the factor, without decomposing C; O(d^2 lambda) per generation.
    """

    CMSA = 5
    """
    CMSA-ES (Beyer & Sendhoff 2008), covariance matrix self-adaptation: C is an exponential average of the selected y y^T with time constant 1 + d (d + 1) / (2 mueff), factorized with a Cholesky decomposition every generation. Meant for StepSizeAdaptation.SA; makes the default mu lambda / 4.
    """

    COVARIANCE_NO_EIGV = 6
    """
    The covariance update of COVARIANCE, with C factorized by a Cholesky decomposition instead of an eigendecomposition (cheaper). The path p_sigma uses the inverse Cholesky factor instead of C^-1/2 (the same up to a rotation), which worked as well as COVARIANCE with CSA in tests. Chosen automatically when COVARIANCE is combined with a step-size adaptation other than CSA.
    """

    NATURAL_GRADIENT = 7
    """
    xNES (Glasmachers et al. 2010), exponential natural evolution strategies: A is updated with the matrix exponential of the natural gradient, A <- A expm(eta G); a low-rank form keeps it O(d^2 lambda) for lambda < d. With StepSizeAdaptation.XNES also xNES's update of sigma, and with RecombinationWeights.UTILITY (also selected by `Modules.active` with the default weights) xNES's utilities for the mean and the matrix: then it is xNES.
    """

class CenterPlacement(enum.Enum):
    """
    Where the mean of the search distribution is placed at every restart (`Modules.center_placement`; default X0). The first run always starts at `Settings.x0`, or at the center of the bounds.
    """

    X0 = 0
    """
    At `Settings.x0`, or at the centre of the bounds when x0 is not given (0 with the default bounds), the default.
    """

    ZERO = 1
    """At the origin."""

    UNIFORM = 2
    """
    At a uniform random point in the bounds: every restart starts elsewhere. In a coordinate with an infinite bound, uniform in x0 +- 2 sigma0 (moved inside a finite bound; x0 is `Settings.x0` or the center).
    """

    CENTER = 3
    """At the centre of the bounds, (lb + ub) / 2."""

    NOVELTY_WEIGHTED = 4
    """
    At one of 512 uniform random candidates in the bounds, drawn with a probability proportional to its squared distance to the nearest point of the repelling archive (distances relative to the domain width). Uniform while the archive is empty; the archive needs `Modules.repelling_type`. With an infinite bound, as UNIFORM.
    """

    MAXIMIN_TABOO = 5
    """
    At the one of 512 uniform random candidates in the bounds that is farthest from its nearest point of the repelling archive (maximin). Uniform while the archive is empty; the archive needs `Modules.repelling_type`. With an infinite bound, as UNIFORM.
    """

class RepellingType(enum.Enum):
    """
    Repelling restarts (`Modules.repelling_type`; default NONE).

    After a restart, the result of the run joins an archive of taboo points, unless hill-valley tests (which evaluate the objective) show that it lies in the basin of a point already there. New samples too close to a taboo point are rejected and drawn again, which steers later runs away from the basins already found (paper: doi:10.1007/978-3-031-70068-2_17). Only useful with a restart strategy; adds the restart criterion `restart.TooMuchRepelling`.
    """

    NONE = 0
    """No repelling, the default."""

    COVERAGE = 1
    """
    Radii from a coverage of the search volume: the radius of a taboo point grows with the number of runs that ended in its basin and shrinks as the total number of runs grows.
    """

    ADAPTIVE = 2
    """
    Radii adapted from the rejection rate of the samples and the evaluations the runs need, between a minimum and a maximum radius.
    """

class RepellingMetric(enum.Enum):
    """
    The distance between a sample and a taboo point of repelling (`Modules.repelling_metric`; default EUCLIDEAN).

    Every metric divides the distance by the current sigma before comparing it with the radius of the taboo point.
    """

    EUCLIDEAN = 0
    """||x - x_t||, the Euclidean distance, the default."""

    CURRENT = 1
    """
    ||A^-1 (x - x_t)||, with A of the current run: the Mahalanobis distance in the current search distribution.
    """

    ARCHIVED = 2
    """
    ||A_t^-1 (x - x_t)||, with A_t of the run that found x_t, scaled to unit determinant: the shape of the basin that run had learned.
    """
