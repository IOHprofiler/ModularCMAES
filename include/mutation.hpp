#pragma once

#include "common.hpp"
#include "population.hpp"
#include "sampling.hpp"
#include "modules.hpp"

namespace parameters
{
    struct Stats;
    struct Parameters;
    struct Weights;
    struct Strategy;
    struct Modules;
}
namespace matrix_adaptation {
    struct Adaptation;
}
namespace bounds
{
    struct BoundCorrection;
}

namespace mutation
{

    struct ThresholdConvergence
    {
        double init_threshold = 0.1;
        double decay_factor = 0.995;
        virtual void scale(Population &pop, const double diameter, const size_t budget, const size_t evaluations);
    };

    struct NoThresholdConvergence : ThresholdConvergence
    {
        void scale(Population &pop, const double diameter, const size_t budget, const size_t evaluations) override {}
    };

    class SequentialSelection
    {
        double seq_cutoff_factor;
        size_t seq_cutoff;

    public:
        SequentialSelection(const parameters::Mirror &m, const size_t mu, const double seq_cutoff_factor = 1.0) : seq_cutoff_factor(m == parameters::Mirror::PAIRWISE ? std::max(2., seq_cutoff_factor) : seq_cutoff_factor),
                                                                                                                seq_cutoff(static_cast<size_t>(mu * seq_cutoff_factor))
        {
        }
        virtual bool break_conditions(const size_t i, const double f, double fopt, const parameters::Mirror &m);
    };

    struct NoSequentialSelection : SequentialSelection
    {

        using SequentialSelection::SequentialSelection;

        bool break_conditions(const size_t i, const double f, double fopt, const parameters::Mirror &m) override { return false; }
    };

    struct SigmaSampler
    {
        double beta;

        SigmaSampler(const double d) : beta(std::log(2.0) / std::max((std::sqrt(d) * std::log(d)), 1.0)) {}

        virtual void sample(const double sigma, Population &pop) const
        {
            pop.s = sampling::Random<std::lognormal_distribution<>>(pop.s.size(),
                                                                    std::lognormal_distribution<>(std::log(sigma), beta))();
        }
    };

    struct NoSigmaSampler : SigmaSampler
    {
        using SigmaSampler::SigmaSampler;

        void sample(const double sigma, Population &pop) const override
        {
            pop.s = pop.s.Constant(pop.s.size(), sigma);
        }
    };

    struct Strategy
    {
        std::shared_ptr<ThresholdConvergence> tc;
        std::shared_ptr<SequentialSelection> sq;
        std::shared_ptr<SigmaSampler> ss;
        double cs;
        double sigma;
        double s = 0;

        Strategy(
            const std::shared_ptr<ThresholdConvergence> &threshold_covergence,
            const std::shared_ptr<SequentialSelection> &sequential_selection,
            const std::shared_ptr<SigmaSampler> &sigma_sampler,
            const double cs, const double sigma0) : tc(threshold_covergence), sq(sequential_selection), ss(sigma_sampler), cs(cs), sigma(sigma0) {}


        virtual void mutate(FunctionType& objective, const size_t n_offspring, parameters::Parameters& p) = 0;

        virtual void adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population &pop,
                           const Population &old_pop, const parameters::Stats &stats, const size_t lambda) = 0;
    };

    struct CSA : Strategy
    {
        double damps;

        CSA(const std::shared_ptr<ThresholdConvergence> &threshold_covergence,
            const std::shared_ptr<SequentialSelection> &sequential_selection,
            const std::shared_ptr<SigmaSampler> &sigma_sampler,
            const double cs, const double damps, const double sigma0) : Strategy(threshold_covergence, sequential_selection, sigma_sampler, cs, sigma0), damps(damps) {}

        void mutate(FunctionType& objective, const size_t n_offspring, parameters::Parameters& p) override;

        void adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population &pop,
                   const Population &old_pop, const parameters::Stats &stats, const size_t lambda) override;
    };

    struct TPA : CSA
    {
        using CSA::CSA;

        double a_tpa = 0.5;
        double b_tpa = 0.0;
        double rank_tpa = 0.0;

        void mutate(FunctionType& objective, const size_t n_offspring, parameters::Parameters &p) override;

        void adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population &pop,
                   const Population &old_pop, const parameters::Stats &stats, const size_t lambda) override;
    };

    struct MSR : CSA
    {
        using CSA::CSA;

        void adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population &pop,
                   const Population &old_pop, const parameters::Stats &stats, const size_t lambda) override;
    };

    struct PSR : CSA
    {
        double succes_ratio = .25;
        
        Vector combined;

        using CSA::CSA;

        void adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population &pop,
                   const Population &old_pop, const parameters::Stats &stats, const size_t lambda) override;
    };

    struct XNES : CSA
    {
        using CSA::CSA;

        void adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population &pop,
                   const Population &old_pop, const parameters::Stats &stats, const size_t lambda) override;
    };

    struct MXNES : CSA
    {
        using CSA::CSA;

        void adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population &pop,
                   const Population &old_pop, const parameters::Stats &stats, const size_t lambda) override;
    };

    struct LPXNES : CSA
    {
        using CSA::CSA;

        void adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population &pop,
                   const Population &old_pop, const parameters::Stats &stats, const size_t lambda) override;
    };

    std::shared_ptr<Strategy> get(const parameters::Modules &m, const size_t mu,
                                  const double mueff, const double d, const double sigma, const std::optional<double> cs);

}