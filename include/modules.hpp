#pragma once

namespace parameters
{
    enum class RecombinationWeights
    {
        DEFAULT,
        EQUAL,
        HALF_POWER_LAMBDA
    };

    enum class BaseSampler
    {
        GAUSSIAN,
        SOBOL,
        HALTON,
        TESTER
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
        PSR
    };

    enum class CorrectionMethod
    {
        NONE,
        COUNT,
        MIRROR,
        COTN,
        UNIFORM_RESAMPLE,
        SATURATE,
        TOROIDAL
    };

    enum class RestartStrategyType
    {
        NONE,
        STOP,
        RESTART,
        IPOP,
        BIPOP
    };

    enum class MatrixAdaptationType {
        NONE,
        COVARIANCE,
        MATRIX,
        SEPERABLE        
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
        BaseSampler sampler = BaseSampler::GAUSSIAN;
        Mirror mirrored = Mirror::NONE;
        StepSizeAdaptation ssa = StepSizeAdaptation::CSA;
        CorrectionMethod bound_correction = CorrectionMethod::NONE;
        RestartStrategyType restart_strategy = RestartStrategyType::NONE;
        MatrixAdaptationType matrix_adaptation = MatrixAdaptationType::COVARIANCE;
    };
}