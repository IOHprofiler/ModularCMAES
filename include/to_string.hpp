#pragma once

#include "parameters.hpp"

namespace parameters {

inline std::string to_string(RecombinationWeights w) {
    switch (w) {
        case RecombinationWeights::DEFAULT: return "DEFAULT";
        case RecombinationWeights::EQUAL: return "EQUAL";
        case RecombinationWeights::EXPONENTIAL: return "EXPONENTIAL";
    }
    return "unknown";
}

inline std::string to_string(BaseSampler s) {
    switch (s) {
        case BaseSampler::UNIFORM: return "UNIFORM";
        case BaseSampler::SOBOL: return "SOBOL";
        case BaseSampler::HALTON: return "HALTON";
        case BaseSampler::TESTER: return "TESTER";
    }
    return "unknown";
}

inline std::string to_string(SampleTranformerType s) {
    switch (s) {
        case SampleTranformerType::NONE: return "NONE";
        case SampleTranformerType::GAUSSIAN: return "GAUSSIAN";
        case SampleTranformerType::SCALED_UNIFORM: return "SCALED_UNIFORM";
        case SampleTranformerType::LAPLACE: return "LAPLACE";
        case SampleTranformerType::LOGISTIC: return "LOGISTIC";
        case SampleTranformerType::CAUCHY: return "CAUCHY";
        case SampleTranformerType::DOUBLE_WEIBULL: return "DOUBLE_WEIBULL";
    }
    return "unknown";
}

inline std::string to_string(Mirror s) {
    switch (s) {
        case Mirror::NONE: return "NONE";
        case Mirror::MIRRORED: return "MIRRORED";
        case Mirror::PAIRWISE: return "PAIRWISE";
    }
    return "unknown";
}

inline std::string to_string(StepSizeAdaptation s) {
    switch (s) {
        case StepSizeAdaptation::CSA: return "CSA";
        case StepSizeAdaptation::TPA: return "TPA";
        case StepSizeAdaptation::MSR: return "MSR";
        case StepSizeAdaptation::XNES: return "XNES";
        case StepSizeAdaptation::MXNES: return "MXNES";
        case StepSizeAdaptation::LPXNES: return "LPXNES";
        case StepSizeAdaptation::PSR: return "PSR";
        case StepSizeAdaptation::SR: return "SR";
        case StepSizeAdaptation::SA: return "SA";
    }
    return "unknown";
}

inline std::string to_string(CorrectionMethod s) {
    switch (s) {
        case CorrectionMethod::NONE: return "NONE";
        case CorrectionMethod::MIRROR: return "MIRROR";
        case CorrectionMethod::COTN: return "COTN";
        case CorrectionMethod::UNIFORM_RESAMPLE: return "UNIFORM_RESAMPLE";
        case CorrectionMethod::SATURATE: return "SATURATE";
        case CorrectionMethod::TOROIDAL: return "TOROIDAL";
        case CorrectionMethod::RESAMPLE: return "RESAMPLE";
    }
    return "unknown";
}

inline std::string to_string(RestartStrategyType s) {
    switch (s) {
        case RestartStrategyType::NONE: return "NONE";
        case RestartStrategyType::RESTART: return "RESTART";
        case RestartStrategyType::STOP: return "STOP";
        case RestartStrategyType::IPOP: return "IPOP";
        case RestartStrategyType::BIPOP: return "BIPOP";
    }
    return "unknown";
}

inline std::string to_string(MatrixAdaptationType s) {
    switch (s) {
        case MatrixAdaptationType::NONE: return "NONE";
        case MatrixAdaptationType::COVARIANCE: return "COVARIANCE";
        case MatrixAdaptationType::MATRIX: return "MATRIX";
        case MatrixAdaptationType::SEPARABLE: return "SEPARABLE";
        case MatrixAdaptationType::CHOLESKY: return "CHOLESKY";
        case MatrixAdaptationType::CMSA: return "CMSA";
        case MatrixAdaptationType::COVARIANCE_NO_EIGV: return "COVARIANCE_NO_EIGV";
        case MatrixAdaptationType::NATURAL_GRADIENT: return "NATURAL_GRADIENT";
    }
    return "unknown";
}

inline std::string to_string(CenterPlacement s) {
    switch (s) {
        case CenterPlacement::X0: return "X0";
        case CenterPlacement::ZERO: return "ZERO";
        case CenterPlacement::UNIFORM: return "UNIFORM";
        case CenterPlacement::CENTER: return "CENTER";
    }
    return "unknown";
}

inline std::string to_string(const Modules& mod) {
    std::stringstream ss;
    ss << std::boolalpha;
    ss << "<Modules";
    ss << " elitist: " << mod.elitist;
    ss << " active: " << mod.active;
    ss << " orthogonal: " << mod.orthogonal;
    ss << " sequential_selection: " << mod.sequential_selection;
    ss << " threshold_convergence: " << mod.threshold_convergence;
    ss << " sample_sigma: " << mod.sample_sigma;
    ss << " repelling_restart: " << mod.repelling_restart;
    ss << " weights: " << parameters::to_string(mod.weights);
    ss << " sampler: " << parameters::to_string(mod.sampler);
    ss << " mirrored: " << parameters::to_string(mod.mirrored);
    ss << " ssa: " << parameters::to_string(mod.ssa);
    ss << " bound_correction: " << parameters::to_string(mod.bound_correction);
    ss << " restart_strategy: " << parameters::to_string(mod.restart_strategy);
    ss << " matrix_adaptation: " << parameters::to_string(mod.matrix_adaptation);
    ss << " center_placement: " << parameters::to_string(mod.center_placement);
    ss << " sample_transformation: " << parameters::to_string(mod.sample_transformation);
    ss << ">";
    return ss.str();
}

template <typename T>
std::string to_string(const std::optional<T>& t) {
    if (!t)
        return "None";

    std::stringstream ss;
    ss << *t;
    return ss.str();
}
}  // namespace parameters