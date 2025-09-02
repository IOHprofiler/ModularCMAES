#pragma once

namespace parameters
{
    enum class RecombinationWeights
    {
        DEFAULT,
        EQUAL,
        EXPONENTIAL
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
    };

    struct Modules
    {
        bool elitist = false;
        bool active = false;
        bool orthogonal = false;
        bool sequential_selection = false;
        bool threshold_convergence = false;
        bool sample_sigma = false;
        bool repelling_restart = false;
        RecombinationWeights weights = RecombinationWeights::DEFAULT;
        BaseSampler sampler = BaseSampler::UNIFORM;
        Mirror mirrored = Mirror::NONE;
        StepSizeAdaptation ssa = StepSizeAdaptation::CSA;
        CorrectionMethod bound_correction = CorrectionMethod::NONE;
        RestartStrategyType restart_strategy = RestartStrategyType::NONE;
        MatrixAdaptationType matrix_adaptation = MatrixAdaptationType::COVARIANCE;
        CenterPlacement center_placement = CenterPlacement::X0;
        SampleTranformerType sample_transformation = SampleTranformerType::GAUSSIAN;
    };
}