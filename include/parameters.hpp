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

    struct Stats
    {
        size_t t = 0;
        size_t evaluations = 0;
        double target = 1e-8;
        size_t max_generations = -1;
        size_t budget = 100000;
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

        Weights(const size_t dim, const size_t mu, const size_t lambda, const Modules &m);

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

        Dynamic(const size_t dim);

        void adapt_evolution_paths(const Weights &w, const std::shared_ptr<mutation::Strategy> &mutation, const Stats &stats, const size_t lambda);

        void adapt_covariance_matrix(const Weights &w, const Modules &m, const Population &pop, const size_t mu);

        bool perform_eigendecomposition(const Stats &stats);
    };
        

    struct Parameters
    {
        size_t dim;
        size_t lambda;
        size_t mu;

        Modules modules;
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

        bool verbose = true;

        Parameters(const size_t dim);
        Parameters(const size_t dim, const Modules &m);

        void adapt();

        void perform_restart(const std::optional<double> &sigma = std::nullopt);
    };
}

std::ostream &operator<<(std::ostream &os, const parameters::Stats &dt);