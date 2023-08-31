#pragma once

#include "common.hpp"
#include "sampling.hpp"
#include "population.hpp"

namespace parameters {
    struct Stats;
    struct Parameters;   
    struct Weights;
    struct Dynamic;
    struct Strategy;
    struct Modules;
}

namespace bounds {
    struct BoundCorrection;
}

namespace mutation {
    
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

    class ThresholdConvergence {
        
        double init_threshold = 0.1;
        double decay_factor = 0.995;

    public:
        virtual void scale(Matrix& z, const parameters::Stats& s, const std::shared_ptr<bounds::BoundCorrection>& bounds);
    };

    struct NoThresholdConvergence : ThresholdConvergence {
        void scale(Matrix& z, const parameters::Stats& s, const std::shared_ptr<bounds::BoundCorrection>& bounds) override {}
    };

    class SequentialSelection {
        double seq_cutoff_factor;
        size_t seq_cutoff;

    public:
        SequentialSelection(const sampling::Mirror& m, const size_t mu, const double seq_cutoff_factor = 1.0) :
            seq_cutoff_factor(m == sampling::Mirror::PAIRWISE ? std::max(2., seq_cutoff_factor) : seq_cutoff_factor),
            seq_cutoff(static_cast<size_t>(mu * seq_cutoff_factor)) {

        }
        virtual bool break_conditions(const size_t i, const double f, double fopt, const sampling::Mirror& m);
    };

    struct NoSequentialSelection: SequentialSelection {
        
        using SequentialSelection::SequentialSelection;
        
        bool break_conditions(const size_t i, const double f, double fopt, const sampling::Mirror& m) override { return false; }
    };

    struct SigmaSampler {
        double beta;
        
        SigmaSampler(const double d): beta(std::log(2.0) / std::max((std::sqrt(d) * std::log(d)), 1.0)) {}

        virtual void sample(const double sigma, Population& pop) const {
            pop.s = sampling::Random<std::lognormal_distribution<>>(pop.s.size(),
                std::lognormal_distribution<>(std::log(sigma), beta))();
        }
    };

    struct NoSigmaSampler : SigmaSampler  {
        
        using SigmaSampler::SigmaSampler;

        void sample(const double sigma, Population& pop) const override {
            pop.s = pop.s.Constant(pop.s.size(), sigma);
        }
    };

    struct Strategy {
        std::shared_ptr<ThresholdConvergence> tc;
        std::shared_ptr<SequentialSelection> sq;
        std::shared_ptr<SigmaSampler> ss;
        double cs;
        double sigma0;
        double sigma;
        double s = 0;

        Strategy(
                const std::shared_ptr<ThresholdConvergence>& threshold_covergence,
                const std::shared_ptr<SequentialSelection>& sequential_selection,
                const std::shared_ptr<SigmaSampler>& sigma_sampler,
                const double cs, const double sigma0
            ): 
            tc(threshold_covergence), sq(sequential_selection), ss(sigma_sampler), cs(cs), sigma0(sigma0), sigma(sigma0) {}

        virtual void mutate(std::function<double(Vector)> objective, const size_t n_offspring, parameters::Parameters& p) = 0;
        
        virtual void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const size_t lambda) = 0;

        //! Calls adapt_sigma and then ss->sample(pop);
        void adapt(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const size_t lambda) {
            adapt_sigma(w, dynamic, pop, old_pop, stats, lambda);
            sample_sigma(pop);
        }

        void sample_sigma(Population& pop) const {
            ss->sample(sigma, pop);
        }
    };

    struct CSA: Strategy {
        double damps;
        
        CSA(const std::shared_ptr<ThresholdConvergence>& threshold_covergence,
            const std::shared_ptr<SequentialSelection>& sequential_selection,
            const std::shared_ptr<SigmaSampler>& sigma_sampler,
            const double cs, const double damps, const double sigma0
        ): Strategy(threshold_covergence, sequential_selection, sigma_sampler, cs, sigma0), damps(damps) {}
        
        void mutate(std::function<double(Vector)> objective, const size_t n_offspring, parameters::Parameters& p) override;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
        
    };

    struct TPA: CSA {
        using CSA::CSA;

        double a_tpa = 0.5;
        double b_tpa = 0.0;
        double rank_tpa = 0.0;

        void mutate(std::function<double(Vector)> objective, const size_t n_offspring, parameters::Parameters& p) override;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
    };

    struct MSR: CSA {
        using CSA::CSA;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
    };

    struct PSR : CSA {
        double succes_ratio = .25;

        using CSA::CSA;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
    };

    struct XNES : CSA {
        using CSA::CSA;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
    };

    struct MXNES : CSA {
        using CSA::CSA;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
    };

    struct LPXNES : CSA {
        using CSA::CSA;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
    };

    std::shared_ptr<Strategy> get(const parameters::Modules& m, const size_t mu,
        const double mueff, const double d, const double sigma);
             
}