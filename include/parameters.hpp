#pragma once

#include "bounds.hpp"
#include "mutation.hpp"
#include "population.hpp"
#include "restart.hpp"
#include "sampling.hpp"
#include "selection.hpp"

using size_to = std::optional<size_t>;

namespace parameters
{

    enum class RecombinationWeights
    {
        DEFAULT,
        EQUAL,
        HALF_POWER_LAMBDA
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
        sampling::BaseSampler sampler = sampling::BaseSampler::GAUSSIAN;
        sampling::Mirror mirrored = sampling::Mirror::NONE;
        mutation::StepSizeAdaptation ssa = mutation::StepSizeAdaptation::CSA;
        bounds::CorrectionMethod bound_correction = bounds::CorrectionMethod::NONE;
        restart::StrategyType restart_strategy = restart::StrategyType::NONE;
    };

    struct Settings
    {
        size_t dim;
        Modules modules;

        std::optional<double> target;
        std::optional<size_t> max_generations;
        size_t budget;

        double sigma0;
        size_t lambda0;
        size_t mu0;

        std::optional<Vector> x0;
        Vector lb;
        Vector ub;
        std::optional<double> cs;
        std::optional<double> cc;
        std::optional<double> cmu;
        std::optional<double> c1;        
        bool verbose;

        Settings(size_t dim,
                 std::optional<Modules> mod = std::nullopt,
                 std::optional<double> target = std::nullopt,
                 std::optional<size_t> max_generations = std::nullopt,
                 std::optional<size_t> budget = std::nullopt,
                 std::optional<double> sigma = std::nullopt,
                 std::optional<size_t> lambda = std::nullopt,
                 std::optional<size_t> mu = std::nullopt,
                 std::optional<Vector> x0 = std::nullopt,
                 std::optional<Vector> lb = std::nullopt,
                 std::optional<Vector> ub = std::nullopt,
                 std::optional<double> cs = std::nullopt,
                 std::optional<double> cc = std::nullopt,
                 std::optional<double> cmu = std::nullopt,
                 std::optional<double> c1 = std::nullopt,
                 bool verbose = false) : dim(dim),
                                         modules(mod.value_or(Modules())),
                                         target(target),
                                         max_generations(max_generations),
                                         budget(budget.value_or(dim * 1e4)),
                                         sigma0(sigma.value_or(2.0)),
                                         lambda0(lambda.value_or(4 + std::floor(3 * std::log(dim)))),
                                         mu0(mu.value_or(lambda0 / 2)),
                                         x0(x0),
                                         lb(lb.value_or(Vector::Ones(dim) * -5)),
                                         ub(ub.value_or(Vector::Ones(dim) * 5)),
                                         cs(cs),
                                         cc(cc),
                                         cmu(cmu),
                                         c1(c1),
                                         verbose(verbose)
        {
            if (modules.mirrored == sampling::Mirror::PAIRWISE and lambda0 % 2 != 0)
                lambda0++;

            if (mu0 > lambda0)
            {
                mu0 = lambda0 / 2;
            }
        }
    };

    struct Stats
    {
        size_t t = 0;
        size_t evaluations = 0;
        Vector xopt = Vector(0);
        double fopt = std::numeric_limits<double>::infinity();
    };

    struct Weights
    {
        Vector weights;
        Vector positive;
        Vector negative;

        double mueff, mueff_neg;
        double c1, cmu, cc;

        Weights(const size_t dim, const size_t mu, const size_t lambda, const Settings &settings);

        void weights_default(const size_t lambda);

        void weights_equal(const size_t mu);

        void weights_half_power_lambda(const size_t mu, const size_t lambda);

        Vector clipped() const;
    };

    struct Dynamic
    {
        Vector m, m_old, dm;
        Vector pc, ps, d;
        Matrix B, C;
        Matrix inv_root_C;
        double dd;
        double chiN;
        bool hs = true;

        Dynamic(const size_t dim, const Vector &x0);

        void adapt_evolution_paths(const Weights &w, const std::shared_ptr<mutation::Strategy> &mutation, const Stats &stats, const size_t lambda);

        void adapt_covariance_matrix(const Weights &w, const Modules &m, const Population &pop, const size_t mu);

        bool perform_eigendecomposition(const Settings &settings);
    };

    struct Parameters
    {
        size_t lambda;
        size_t mu;

        Settings settings;
        Dynamic dynamic;
        Stats stats;
        Weights weights;

        Population pop;
        Population old_pop;

        std::shared_ptr<sampling::Sampler> sampler;
        std::shared_ptr<mutation::Strategy> mutation;
        std::shared_ptr<selection::Strategy> selection;
        std::shared_ptr<restart::Strategy> restart;
        std::shared_ptr<bounds::BoundCorrection> bounds;

        Parameters(const size_t dim);
        Parameters(const Settings &settings);

        void adapt();

        void perform_restart(const std::optional<double> &sigma = std::nullopt);

        bool invalid_state() const;
    };

    inline std::string to_string(const RecombinationWeights &w)
    {
        switch (w)
        {
        case RecombinationWeights::EQUAL:
            return "EQUAL";
        case RecombinationWeights::HALF_POWER_LAMBDA:
            return "HALF_POWER_LAMBDA";
        default:
        case RecombinationWeights::DEFAULT:
            return "DEFAULT";
        }
    }

    inline std::string to_string(const sampling::BaseSampler &s)
    {
        switch (s)
        {
        case sampling::BaseSampler::GAUSSIAN:
            return "GAUSSIAN";
        case sampling::BaseSampler::SOBOL:
            return "SOBOL";
        case sampling::BaseSampler::HALTON:
            return "HALTON";
        default:
        case sampling::BaseSampler::TESTER:
            return "TESTER";
        }
    }
    inline std::string to_string(const sampling::Mirror &s)
    {
        switch (s)
        {
        case sampling::Mirror::NONE:
            return "NONE";
        case sampling::Mirror::MIRRORED:
            return "MIRRORED";
        default:
        case sampling::Mirror::PAIRWISE:
            return "PAIRWISE";
        }
    }
    inline std::string to_string(const mutation::StepSizeAdaptation &s)
    {
        switch (s)
        {
        case mutation::StepSizeAdaptation::CSA:
            return "CSA";
        case mutation::StepSizeAdaptation::TPA:
            return "TPA";
        case mutation::StepSizeAdaptation::MSR:
            return "MSR";
        case mutation::StepSizeAdaptation::XNES:
            return "XNES";
        case mutation::StepSizeAdaptation::MXNES:
            return "MXNES";
        case mutation::StepSizeAdaptation::LPXNES:
            return "LPXNES";
        default:
        case mutation::StepSizeAdaptation::PSR:
            return "PSR";
        }
    }
    inline std::string to_string(const bounds::CorrectionMethod &s)
    {
        switch (s)
        {
        case bounds::CorrectionMethod::NONE:
            return "NONE";
        case bounds::CorrectionMethod::COUNT:
            return "COUNT";
        case bounds::CorrectionMethod::MIRROR:
            return "MIRROR";
        case bounds::CorrectionMethod::COTN:
            return "COTN";
        case bounds::CorrectionMethod::UNIFORM_RESAMPLE:
            return "UNIFORM_RESAMPLE";
        case bounds::CorrectionMethod::SATURATE:
            return "SATURATE";
        default:
        case bounds::CorrectionMethod::TOROIDAL:
            return "TOROIDAL";
        }
    }
    inline std::string to_string(const restart::StrategyType &s)
    {
        switch (s)
        {
        case restart::StrategyType::NONE:
            return "NONE";
        case restart::StrategyType::RESTART:
            return "RESTART";
        case restart::StrategyType::IPOP:
            return "IPOP";
        default:
        case restart::StrategyType::BIPOP:
            return "BIPOP";
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

std::ostream &operator<<(std::ostream &os, const parameters::Stats &dt);