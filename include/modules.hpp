#pragma once

namespace parameters
{
enum class RecombinationWeights
{
    DEFAULT,
    EQUAL,
    EXPONENTIAL,
    UTILITY
};

enum class BaseSampler
{
    UNIFORM,
    SOBOL,
    HALTON,
    TESTER
};

enum class SampleTranformerType
{
    NONE,
    GAUSSIAN,
    SCALED_UNIFORM,
    LAPLACE,
    LOGISTIC,
    CAUCHY,
    DOUBLE_WEIBULL
};

enum class Mirror
{
    NONE,
    MIRRORED,
    PAIRWISE
};

enum class StepSizeAdaptation
{
    CSA,
    TPA,
    MSR,
    XNES,
    MXNES,
    LPXNES,
    PSR,
    SR,
    SA,
    //! Two-point step-size adaptation within the population (#173); appended last, so that the
    //! other options keep their values
    TPA_INPLACE,
};

enum class CorrectionMethod
{
    NONE,
    MIRROR,
    COTN,
    UNIFORM_RESAMPLE,
    SATURATE,
    TOROIDAL,
    RESAMPLE
};

enum class RestartStrategyType
{
    NONE,
    RESTART,
    STOP,
    IPOP,
    BIPOP
};

enum class MatrixAdaptationType
{
    NONE,
    COVARIANCE,
    MATRIX,
    SEPARABLE,
    CHOLESKY,
    CMSA,
    COVARIANCE_NO_EIGV,
    NATURAL_GRADIENT
};

enum class CenterPlacement
{
    X0,
    ZERO,
    UNIFORM,
    CENTER,
    NOVELTY_WEIGHTED,
    MAXIMIN_TABOO
};

enum class RepellingType
{
    NONE,
    COVERAGE,
    ADAPTIVE,
};

//! The distance between a sample and a taboo point of repelling, divided by the current sigma
enum class RepellingMetric
{
    //! ||x - x_t||: the default from July 2026 (78ddafe) until #158
    EUCLIDEAN,
    //! ||A^-1 (x - x_t)||, with A of the current run (Mahalanobis in the current search
    //! distribution, the framework paper's Eq. (8)): the default (#158)
    CURRENT,
    //! ||A_t^-1 (x - x_t)||, with A_t of the run that found x_t, scaled to unit determinant
    ARCHIVED,
};

struct Modules
{
    bool elitist = false;
    bool active = false;
    bool orthogonal = false;
    bool sequential_selection = false;
    bool threshold_convergence = false;
    bool sample_sigma = false;
    RecombinationWeights weights = RecombinationWeights::DEFAULT;
    BaseSampler sampler = BaseSampler::UNIFORM;
    Mirror mirrored = Mirror::NONE;
    StepSizeAdaptation ssa = StepSizeAdaptation::CSA;
    CorrectionMethod bound_correction = CorrectionMethod::NONE;
    RestartStrategyType restart_strategy = RestartStrategyType::NONE;
    MatrixAdaptationType matrix_adaptation = MatrixAdaptationType::COVARIANCE;
    CenterPlacement center_placement = CenterPlacement::X0;
    SampleTranformerType sample_transformation = SampleTranformerType::GAUSSIAN;
    RepellingType repelling_type = RepellingType::NONE;
    //! The paper's Mahalanobis distance by default (the maintainer's decision on #158)
    RepellingMetric repelling_metric = RepellingMetric::CURRENT;
};
} // namespace parameters