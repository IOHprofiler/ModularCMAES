#include "mutation.hpp"
#include "parameters.hpp"
#include "bounds.hpp"


namespace mutation {
    
    void ThresholdConvergence::scale(Matrix& z, const parameters::Stats& s, const std::shared_ptr<bounds::BoundCorrection>& bounds)
    {
        const double t = init_threshold * bounds->diameter * pow(static_cast<double>(s.budget - s.evaluations) / static_cast<double>(s.budget), decay_factor);
        const auto norm = z.colwise().norm().array().replicate(z.cols() - 1, 1);
        z = (norm < t).select(z.array() * ((t + (t - norm)) / norm), z);
    }

    bool SequentialSelection::break_conditions(const size_t i, const double f, double fopt, const sampling::Mirror& m) {
        return (f < fopt) and (i >= seq_cutoff) and (m != sampling::Mirror::PAIRWISE or i % 2 == 0);
    }

    void CSA::adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
        const Population& old_pop, const parameters::Stats& stats, const size_t lambda) {
        sigma *= std::exp((cs / damps) * ((dynamic.ps.norm() / dynamic.chiN) - 1));
    }
    
    void CSA::mutate(std::function<double(Vector)> objective, const size_t n_offspring, parameters::Parameters& p) {
        for (size_t i = 0; i < n_offspring; ++i)
            p.pop.Z.col(i) = (*p.sampler)();

        p.mutation->tc->scale(p.pop.Z, p.stats, p.bounds);

        p.pop.Y = p.dynamic.B * (p.dynamic.d.asDiagonal() * p.pop.Z);
        p.pop.X = (p.pop.Y * p.pop.s.asDiagonal()).colwise() + p.dynamic.m;

        p.bounds->correct(p.pop.X, p.pop.Y, p.pop.s, p.dynamic.m);

        bool sequential_break_conditions = false;
        for (auto i = 0; i < p.pop.X.cols() and !sequential_break_conditions; ++i)
        {
            p.pop.f(i) = objective(p.pop.X.col(i));
            p.stats.evaluations++;
            sequential_break_conditions = sq->break_conditions(i, p.pop.f(i), p.stats.fopt, p.modules.mirrored);
        }
    }

    void TPA::mutate(std::function<double(Vector)> objective, const size_t n_offspring_, parameters::Parameters& p) {
        const size_t n_offspring = n_offspring_ - 2;

        CSA::mutate(objective, n_offspring, p);

        p.pop.Y.col(n_offspring) = p.dynamic.dm;
        p.pop.Y.col(n_offspring + 1) = -p.dynamic.dm;

        for (auto i = n_offspring; i < n_offspring + 2; i++) {
            p.pop.X.col(i) = p.dynamic.m + (p.pop.s(i) * p.pop.Y.col(i));
            
            // TODO: this only needs to happen for a single column
            p.bounds->correct(p.pop.X, p.pop.Y, p.pop.s, p.dynamic.m);
            
            p.pop.f(i) = objective(p.pop.X.col(i));
            p.stats.evaluations++;
        }

        rank_tpa = p.pop.f(n_offspring + 1) < p.pop.f(n_offspring) ?
            -a_tpa : a_tpa + b_tpa;
    }

    void TPA::adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
        const Population& old_pop, const parameters::Stats& stats, const size_t lambda) {
        s = ((1.0 - cs) * s) + (cs * rank_tpa);
        sigma *= std::exp(s);
    }

    //! Assumes the vector to be arready sorted
    double median(const Vector& x)
    {
        if (x.size() % 2 == 0)
            return (x(x.size() / 2) + x(x.size() / 2 - 1)) / 2.0;
        return x(x.size() / 2);
    }   

    void MSR::adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
        const Population& old_pop, const parameters::Stats& stats, const size_t lamb) {
        if (stats.t != 0)
        {
            const double lambda = static_cast<double>(lamb);
            const double k = (pop.f.array() < median(old_pop.f)).cast<double>().sum();
            const auto z = (2.0 / lambda) * (k - ((lambda + 1.0) / 2.0));
            s = ((1.0 - cs) * s) + (cs * z);
            sigma *= std::exp(s / (2.0 - (2.0 / dynamic.dd)));
        }
    }

    //! Returns the indices of the elements of query in database
    Vector searchsorted(const Vector& query, const Vector& database) {
        Vector res(query.size());
        auto i = 0;
      
        for (const auto& xi : query) {
            auto it = std::find(std::begin(database), std::end(database), xi);
            res(i++) = static_cast<double>(std::distance(std::begin(database), it));
        }
        return res;
    }

    void PSR::adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
        const Population& old_pop, const parameters::Stats& stats, const size_t lambda) {

        if (stats.t != 0)
        {
            const auto n = std::min(pop.n_finite(), old_pop.n_finite());
            auto combined = Vector(n+n); 
            combined << pop.f.head(n), old_pop.f.head(n);
            const auto idx = utils::sort_indexes(combined);
            combined = combined(idx).eval();

            auto r = searchsorted(pop.f.head(n), combined);
            auto r_old = searchsorted(old_pop.f.head(n), combined);
            const auto z = (r_old - r).sum() / std::pow(n, 2) - succes_ratio;

            s = (1.0 - cs) * s + (cs * z);
            sigma *= std::exp(s / (2.0 - (2.0 / dynamic.dd)));
        }
    }

    void XNES::adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
        const Population& old_pop, const parameters::Stats& stats, const size_t lambda) {
        
        const double z = ((dynamic.inv_root_C * pop.Y).colwise().norm().array().pow(2.) - dynamic.dd).matrix() * w.clipped();
        sigma *= std::exp((cs / std::sqrt(dynamic.dd)) * z);
    }
    void MXNES::adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
        const Population& old_pop, const parameters::Stats& stats, const size_t lambda) {
        if (stats.t != 0)
        {
            const auto z = (w.mueff * std::pow((dynamic.inv_root_C * dynamic.dm).norm(), 2)) - dynamic.dd;
            sigma *= std::exp((cs / dynamic.dd) * z);
        }
    }
    void LPXNES::adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dynamic, Population& pop,
        const Population& old_pop, const parameters::Stats& stats, const size_t lambda) {
        const auto z = std::exp(cs * pop.s.array().log().matrix().dot(w.clipped()));
        sigma = std::pow(sigma, 1.0 - cs) * z;
    }


    std::shared_ptr<Strategy> get(const parameters::Modules& m, const size_t mu, const double mueff, const double d, const double sigma) {

        auto tc = m.threshold_convergence ? std::make_shared<ThresholdConvergence>()
            : std::make_shared<NoThresholdConvergence>();

        auto sq = m.sequential_selection ? std::make_shared<SequentialSelection>(m.mirrored, mu) :
            std::make_shared<NoSequentialSelection>(m.mirrored, mu);

        auto ss = (m.sample_sigma or m.ssa == StepSizeAdaptation::LPXNES) ?
            std::make_shared<SigmaSampler>(d)
            : std::make_shared<NoSigmaSampler>(d);

        double cs = 0.3;
        double damps = 0.0;

        switch (m.ssa)
        {
        case StepSizeAdaptation::TPA:
            return std::make_shared<TPA>(tc, sq, ss, cs, damps, sigma);
        case StepSizeAdaptation::MSR:
            return std::make_shared<MSR>(tc, sq, ss, cs, damps, sigma);
        case StepSizeAdaptation::XNES:
            cs = mueff / (2.0 * std::log(std::max(2., d)) * sqrt(d));
            return std::make_shared<XNES>(tc, sq, ss, cs, damps, sigma);
        case StepSizeAdaptation::MXNES:
            cs = 1.;
            return std::make_shared<MXNES>(tc, sq, ss, cs, damps, sigma);
        case StepSizeAdaptation::LPXNES:
            cs = 9.0 * mueff / (10.0 * sqrt(d));
            return std::make_shared<LPXNES>(tc, sq, ss, cs, damps, sigma);
        case StepSizeAdaptation::PSR:
            cs = .9;
            return std::make_shared<PSR>(tc, sq, ss, cs, 0., sigma);
        default:
        case StepSizeAdaptation::CSA:
            cs = (mueff + 2.0) / (d + mueff + 5.0);
            damps = 1.0 + (2.0 * std::max(0.0, sqrt((mueff - 1.0) / (d + 1)) - 1) + cs);
            return std::make_shared<CSA>(tc, sq, ss, cs, damps, sigma);
        }
}
   
}