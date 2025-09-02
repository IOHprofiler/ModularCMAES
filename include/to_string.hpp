#pragma once

#include "parameters.hpp"

namespace parameters
{

    inline std::string to_string(const RecombinationWeights &w)
    {
        switch (w)
        {
        case RecombinationWeights::EQUAL:
            return "EQUAL";
        case RecombinationWeights::EXPONENTIAL:
            return "EXPONENTIAL";
        default:
        case RecombinationWeights::DEFAULT:
            return "DEFAULT";
        }
    }

    inline std::string to_string(const BaseSampler &s)
    {
        switch (s)
        {
        case BaseSampler::UNIFORM:
            return "UNIFORM";
        case BaseSampler::SOBOL:
            return "SOBOL";
        case BaseSampler::HALTON:
            return "HALTON";
        default:
        case BaseSampler::TESTER:
            return "TESTER";
        }
    }
    inline std::string to_string(const Mirror &s)
    {
        switch (s)
        {
        case Mirror::NONE:
            return "NONE";
        case Mirror::MIRRORED:
            return "MIRRORED";
        default:
        case Mirror::PAIRWISE:
            return "PAIRWISE";
        }
    }
    inline std::string to_string(const StepSizeAdaptation &s)
    {
        switch (s)
        {
        case StepSizeAdaptation::CSA:
            return "CSA";
        case StepSizeAdaptation::TPA:
            return "TPA";
        case StepSizeAdaptation::MSR:
            return "MSR";
        case StepSizeAdaptation::XNES:
            return "XNES";
        case StepSizeAdaptation::MXNES:
            return "MXNES";
        case StepSizeAdaptation::LPXNES:
            return "LPXNES";
        case StepSizeAdaptation::SR:
            return "SR";
        case StepSizeAdaptation::PSR:
            return "PSR";
        case StepSizeAdaptation::SA:
            return "SA";
        default:
            return "unknown";
        }
    }
    inline std::string to_string(const CorrectionMethod &s)
    {
        switch (s)
        {
        case CorrectionMethod::NONE:
            return "NONE";
        case CorrectionMethod::MIRROR:
            return "MIRROR";
        case CorrectionMethod::COTN:
            return "COTN";
        case CorrectionMethod::UNIFORM_RESAMPLE:
            return "UNIFORM_RESAMPLE";
        case CorrectionMethod::SATURATE:
            return "SATURATE";
        default:
        case CorrectionMethod::TOROIDAL:
            return "TOROIDAL";
        }
    }
    inline std::string to_string(const RestartStrategyType &s)
    {
        switch (s)
        {
        case RestartStrategyType::NONE:
            return "NONE";
        case RestartStrategyType::RESTART:
            return "RESTART";
        case RestartStrategyType::IPOP:
            return "IPOP";
        case RestartStrategyType::STOP:
            return "STOP";
        default:
        case RestartStrategyType::BIPOP:
            return "BIPOP";
        }
    }

    inline std::string to_string(const MatrixAdaptationType &s)
    {
        switch (s)
        {
        case MatrixAdaptationType::MATRIX:
            return "MATRIX";
        case MatrixAdaptationType::CHOLESKY:
            return "CHOLESKY";
        case MatrixAdaptationType::NONE:
            return "NONE";
        case MatrixAdaptationType::SEPARABLE:
            return "SEPARABLE";
        case MatrixAdaptationType::COVARIANCE:
            return "COVARIANCE";
        case MatrixAdaptationType::CMSA:
            return "CMSA";
        case MatrixAdaptationType::COVARIANCE_NO_EIGV:
            return "COVARIANCE_NO_EIGV";
        case MatrixAdaptationType::NATURAL_GRADIENT:
            return "NATURAL_GRADIENT";
        default:
            return "unkown";
        }
    }


    inline std::string to_string(const Modules &mod)
    {
        std::stringstream ss;
        ss << std::boolalpha;
        ss << "<Modules";
        ss << " elitist: " << mod.elitist;
        ss << " active: " << mod.active;
        ss << " orthogonal: " << mod.orthogonal;
        ss << " sequential_selection: " << mod.sequential_selection;
        ss << " threshold_convergence: " << mod.threshold_convergence;
        ss << " sample_sigma: " << mod.sample_sigma;
        ss << " weights: " << parameters::to_string(mod.weights);
        ss << " sampler: " << parameters::to_string(mod.sampler);
        ss << " mirrored: " << parameters::to_string(mod.mirrored);
        ss << " ssa: " << parameters::to_string(mod.ssa);
        ss << " bound_correction: " << parameters::to_string(mod.bound_correction);
        ss << " restart_strategy: " << parameters::to_string(mod.restart_strategy);
        ss << " matrix_adaptation: " << parameters::to_string(mod.matrix_adaptation);
        ss << ">";
        return ss.str();
    }

    template <typename T>
    std::string to_string(const std::optional<T> &t)
    {
        if (t)
        {
            std::stringstream ss;
            ss << t.value();
            return ss.str();
        }
        return "None";
    }
}

