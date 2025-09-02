#include <algorithm>
#include "restart_criteria.hpp"
#include "parameters.hpp"

namespace
{
    //! max - min, for the last n elements of a vector
    Float ptp_tail(const std::vector<Float> &v, const size_t n)
    {
        const auto na = std::min(v.size(), n);
        if (na == 1)
        {
            return v[0];
        }

        const Float min = *std::min_element(v.end() - na, v.end());
        const Float max = *std::max_element(v.end() - na, v.end());
        return max - min;
    }

    // TODO: this is duplicate code
    Float median(const Vector &x)
    {
        if (x.size() == 1)
            return x(0);

        if (x.size() % 2 == 0)
            return (x(x.size() / 2) + x(x.size() / 2 - 1)) / 2.0;
        return x(x.size() / 2);
    }

    Float median(const std::vector<Float> &v, const size_t from, const size_t to)
    {
        if (v.size() == 1)
            return v[0];

        const size_t n = to - from;
        if (n % 2 == 0)
            return (v[from + (n / 2)] + v[from + (n / 2) - 1]) / 2.0;
        return v[from + (n / 2)];
    }
}

namespace restart
{
    void Criterion::reset(const parameters::Parameters &p)
    {
        last_restart = p.stats.t;
        met = false;
        on_reset(p);
    }

    void ExceededMaxIter::on_reset(const parameters::Parameters &p)
    {
        max_iter = static_cast<size_t>(
            100 + 50 * std::pow((static_cast<Float>(p.settings.dim) + 3), 2.0) / std::sqrt(static_cast<Float>(p.lambda))
        );
    }

    void ExceededMaxIter::update(const parameters::Parameters &p)
    {
        const auto time_since_restart = p.stats.t - last_restart;
        met = max_iter < time_since_restart;
    }

    void NoImprovement::on_reset(const parameters::Parameters &p)
    {
        n_bin = 10 + static_cast<size_t>(std::ceil(30 * static_cast<Float>(p.settings.dim) / static_cast<Float>(p.lambda)));
    }

    void NoImprovement::update(const parameters::Parameters& p)
    {
        const size_t time_since_restart = p.stats.t - last_restart;
        best_fitnesses.push_back(p.pop.f(0));
        met = false;
        if (time_since_restart > n_bin)
        {
            const auto recent_improvement = ptp_tail(best_fitnesses, n_bin);
            met = recent_improvement == 0;
        }
    }

    void MaxSigma::update(const parameters::Parameters &p)
    {
        met = p.mutation->sigma > tolerance;
    }

    void MinSigma::update(const parameters::Parameters &p)
    {
        met = p.mutation->sigma < tolerance;
    }

    void UnableToAdapt::update(const parameters::Parameters &p)
    {
        met = !p.successfull_adaptation or !std::isfinite(p.mutation->sigma);
    }

    void FlatFitness::update(const parameters::Parameters &p)
    {
        const size_t time_since_restart = p.stats.t - last_restart;
        flat_fitnesses(p.stats.t % p.settings.dim) = p.pop.f(0) == p.pop.f(flat_fitness_index);
        met = false;
        if (time_since_restart > static_cast<size_t>(flat_fitnesses.size()))
        {
            const size_t n_flat_fitness = static_cast<size_t>(flat_fitnesses.sum());
            met = n_flat_fitness > max_flat_fitness;
        }
    }

    void FlatFitness::on_reset(const parameters::Parameters &p)
    {
        flat_fitnesses = Eigen::Array<int, Eigen::Dynamic, 1>::Constant(p.settings.dim, 0);
        max_flat_fitness = static_cast<size_t>(std::ceil(static_cast<Float>(p.settings.dim) / 3));
        flat_fitness_index = static_cast<size_t>(std::round(.1 + static_cast<Float>(p.lambda) / 4));
    }

    void TolX::update(const parameters::Parameters &p)
    {
        // TODO: This should be another sigma0, the one that has been used to restart
        if (const auto dynamic = std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
        {
            const Float d_sigma = p.mutation->sigma / p.settings.sigma0;
            const Float tolx_condition = tolerance * p.settings.sigma0;
            tolx_vector.head(p.settings.dim) = dynamic->C.diagonal().cwiseSqrt() * d_sigma;
            tolx_vector.tail(p.settings.dim) = dynamic->pc * d_sigma;
            met = (tolx_vector.array() < tolx_condition).all();
        }
    }

    void TolX::on_reset(const parameters::Parameters &p)
    {
        tolx_vector = Vector::Ones(p.settings.dim * 2);
    }

    void MaxDSigma::update(const parameters::Parameters &p)
    {
        const Float d_sigma = p.mutation->sigma / p.settings.sigma0;
        Float root_max_d = 1.0;
        if (const auto dynamic = std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
        {
            root_max_d = std::sqrt(dynamic->d.maxCoeff());
        }
        met = d_sigma > (tolerance * root_max_d);
    }

    void MinDSigma::update(const parameters::Parameters &p)
    {
        const Float d_sigma = p.mutation->sigma / p.settings.sigma0;

        Float root_min_d = 1.0;
        if (const auto dynamic = std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
        {
            root_min_d = std::sqrt(dynamic->d.minCoeff());
        }
        met = d_sigma < (tolerance * root_min_d);
    }

    void ConditionC::update(const parameters::Parameters &p)
    {
        if (const auto dynamic = std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
        {
            const Float condition_c = pow(dynamic->d.maxCoeff(), 2.0) / pow(dynamic->d.minCoeff(), 2);
            met = condition_c > tolerance;
        }
    }

    void NoEffectAxis::update(const parameters::Parameters &p)
    {
        if (const auto dynamic = std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
        {
            const Eigen::Index t = p.stats.t % p.settings.dim;
            const auto effect_axis = 0.1 * p.mutation->sigma * std::sqrt(dynamic->d(t)) * dynamic->B.col(t);
            met = (effect_axis.array().abs() < tolerance).all();
        }
    }

    void NoEffectCoord::update(const parameters::Parameters &p)
    {
        if (const auto dynamic = std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
        {
            const auto effect_coord = 0.2 * p.mutation->sigma * dynamic->C.diagonal().cwiseSqrt();
            met = (effect_coord.array().abs() < tolerance).any();
        }
    }

    void Stagnation::update(const parameters::Parameters &p)
    {
        const size_t time_since_restart = p.stats.t - last_restart;
        const size_t pt = static_cast<size_t>(tolerance * time_since_restart);
        median_fitnesses.push_back(median(p.pop.f));
        best_fitnesses.push_back(p.pop.f(0));

        met = false;
        if (time_since_restart > n_stagnation)
        {
            const bool best_better = median(best_fitnesses, pt, time_since_restart) >= median(best_fitnesses, 0, pt);
            const bool median_better = median(median_fitnesses, pt, time_since_restart) >= median(median_fitnesses, 0, pt);
            met = best_better and median_better;
        }      
    }

    void Stagnation::on_reset(const parameters::Parameters &p)
    {
        const auto d = static_cast<Float>(p.settings.dim);
        const auto lambda = static_cast<Float>(p.lambda);
        n_stagnation = static_cast<size_t>(
            100 + 100 * std::pow(p.settings.dim, 1.5) / static_cast<Float>(p.lambda)
        );

        median_fitnesses = {};
        best_fitnesses = {};
    }

    void TooMuchRepelling::update(const parameters::Parameters& p)
    {
        const Float average_repelling = static_cast<Float>(p.repelling->attempts) / static_cast<Float>(p.lambda);
        met = average_repelling >= tolerance;
    }

    Criteria Criteria::get(const parameters::Modules modules)
    {
        vCriteria criteria{
            std::make_shared<restart::UnableToAdapt>()};

        if (modules.restart_strategy >= parameters::RestartStrategyType::RESTART)
        {
            criteria.push_back(std::make_shared<restart::MinSigma>());
            criteria.push_back(std::make_shared<restart::MaxSigma>());
            criteria.push_back(std::make_shared<restart::ExceededMaxIter>());
            criteria.push_back(std::make_shared<restart::NoImprovement>());
            criteria.push_back(std::make_shared<restart::FlatFitness>());
            criteria.push_back(std::make_shared<restart::Stagnation>());
            criteria.push_back(std::make_shared<restart::MinDSigma>());
            criteria.push_back(std::make_shared<restart::MaxDSigma>());

            //! TODO: make these compatible with other MA
            if (modules.matrix_adaptation == parameters::MatrixAdaptationType::COVARIANCE)
            {
                criteria.push_back(std::make_shared<restart::TolX>());
                criteria.push_back(std::make_shared<restart::ConditionC>());
                criteria.push_back(std::make_shared<restart::NoEffectAxis>());
                criteria.push_back(std::make_shared<restart::NoEffectCoord>());
            }
        }

        if (modules.repelling_restart)
        {
            criteria.push_back(std::make_shared<restart::TooMuchRepelling>());
        }
        return Criteria(criteria);
    }
}