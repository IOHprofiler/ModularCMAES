#include "restart_criteria.hpp"

#include <algorithm>

#include "parameters.hpp"

namespace
{
//! max - min, for the last n elements of a vector
Float ptp_tail(const std::vector<Float> &v, const size_t n)
{
    const auto na = std::min(v.size(), n);

    if (na == 0)
        return 0.0;

    if (na == 1)
        return 0.0;

    const Float min = *std::min_element(v.end() - na, v.end());
    const Float max = *std::max_element(v.end() - na, v.end());

    return max - min;
}

Float median_sorted_copy(std::vector<Float> values)
{
    if (values.empty())
        return std::numeric_limits<Float>::quiet_NaN();

    const size_t n = values.size();
    const size_t mid = n / 2;

    std::nth_element(values.begin(), values.begin() + mid, values.end());

    if (n % 2 == 1)
        return values[mid];

    const Float upper = values[mid];
    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    const Float lower = values[mid - 1];

    return 0.5 * (lower + upper);
}

Float median_tail_range(const std::vector<Float> &v, const size_t from, const size_t to)
{
    if (from >= to || from >= v.size())
        return std::numeric_limits<Float>::quiet_NaN();

    const size_t clipped_to = std::min(to, v.size());

    std::vector<Float> values;
    values.reserve(clipped_to - from);

    for (size_t i = from; i < clipped_to; ++i)
        values.push_back(v[i]);

    return median_sorted_copy(std::move(values));
}

Float median_vector(Vector x)
{
    if (x.size() == 0)
        return std::numeric_limits<Float>::quiet_NaN();

    std::vector<Float> values;
    values.reserve(static_cast<size_t>(x.size()));

    for (Eigen::Index i = 0; i < x.size(); ++i)
        values.push_back(x(i));

    return median_sorted_copy(std::move(values));
}

} // namespace

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
    max_iter =
        static_cast<size_t>(100 + 50 * std::pow((static_cast<Float>(p.settings.dim) + 3), 2.0) /
                                      std::sqrt(static_cast<Float>(p.lambda)));
}

void ExceededMaxIter::update(const parameters::Parameters &p)
{
    const auto time_since_restart = p.stats.t - last_restart;
    met = max_iter < time_since_restart;
}

void NoImprovement::on_reset(const parameters::Parameters &p)
{
    const auto lambda = std::max<int>(1, p.lambda);

    n_bin = 10 + static_cast<size_t>(std::ceil(30.0 * static_cast<Float>(p.settings.dim) /
                                               static_cast<Float>(lambda)));

    best_fitnesses = {};
}

void NoImprovement::update(const parameters::Parameters &p)
{
    const size_t time_since_restart = p.stats.t - last_restart;

    best_fitnesses.push_back(p.pop.f(0));
    met = false;

    if (time_since_restart > n_bin)
    {
        const auto recent_improvement = ptp_tail(best_fitnesses, n_bin);
        met = recent_improvement <= tolerance;
    }
}

void MaxSigma::update(const parameters::Parameters &p) { met = p.mutation->sigma > tolerance; }

void MinSigma::update(const parameters::Parameters &p) { met = p.mutation->sigma < tolerance; }

void UnableToAdapt::update(const parameters::Parameters &p)
{
    met = !p.successful_adaptation or !std::isfinite(p.mutation->sigma);
}

void FlatFitness::update(const parameters::Parameters &p)
{
    met = false;
    if (p.lambda <= 1)
        return;

    const size_t time_since_restart = p.stats.t - last_restart;
    flat_fitnesses(p.stats.t % p.settings.dim) = p.pop.f(0) == p.pop.f(flat_fitness_index);
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
    if (const auto dynamic =
            std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
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
    if (const auto dynamic =
            std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
    {
        root_max_d = std::sqrt(dynamic->d.maxCoeff());
    }
    met = d_sigma > (tolerance * root_max_d);
}

void MinDSigma::update(const parameters::Parameters &p)
{
    const Float d_sigma = p.mutation->sigma / p.settings.sigma0;

    Float root_min_d = 1.0;
    if (const auto dynamic =
            std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
    {
        root_min_d = std::sqrt(dynamic->d.minCoeff());
    }
    met = d_sigma < (tolerance * root_min_d);
}

void ConditionC::update(const parameters::Parameters &p)
{
    if (const auto dynamic =
            std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
    {
        const Float condition_c = pow(dynamic->d.maxCoeff(), 2.0) / pow(dynamic->d.minCoeff(), 2);
        met = condition_c > tolerance;
    }
}

void NoEffectAxis::update(const parameters::Parameters &p)
{
    if (const auto dynamic =
            std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
    {
        const Eigen::Index t = p.stats.t % p.settings.dim;
        const auto effect_axis =
            0.1 * p.mutation->sigma * std::sqrt(dynamic->d(t)) * dynamic->B.col(t);
        met = (effect_axis.array().abs() < tolerance).all();
    }
}

void NoEffectCoord::update(const parameters::Parameters &p)
{
    if (const auto dynamic =
            std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
    {
        const auto effect_coord = 0.2 * p.mutation->sigma * dynamic->C.diagonal().cwiseSqrt();
        met = (effect_coord.array().abs() < tolerance).any();
    }
}

void Stagnation::update(const parameters::Parameters &p)
{
    const size_t time_since_restart = p.stats.t - last_restart;

    median_fitnesses.push_back(median_vector(p.pop.f));
    best_fitnesses.push_back(p.pop.f(0));

    met = false;

    if (time_since_restart <= n_stagnation)
        return;

    const size_t n = best_fitnesses.size();

    if (n < 4)
        return;

    size_t split = static_cast<size_t>(tolerance * static_cast<Float>(n));
    split = std::max<size_t>(1, std::min(split, n - 1));

    const Float old_best_median = median_tail_range(best_fitnesses, 0, split);

    const Float new_best_median = median_tail_range(best_fitnesses, split, n);

    const Float old_median_median = median_tail_range(median_fitnesses, 0, split);

    const Float new_median_median = median_tail_range(median_fitnesses, split, n);

    // Minimization: if newer medians are not better, stagnation.
    const bool best_not_improving = new_best_median >= old_best_median;
    const bool median_not_improving = new_median_median >= old_median_median;

    met = best_not_improving && median_not_improving;
}

void Stagnation::on_reset(const parameters::Parameters &p)
{
    const auto d = static_cast<Float>(p.settings.dim);
    n_stagnation = static_cast<size_t>(100 + 100 * std::pow(d, 1.5) / static_cast<Float>(p.lambda));

    median_fitnesses = {};
    best_fitnesses = {};
}

void TooMuchRepelling::update(const parameters::Parameters &p)
{
    const Float average_repelling =
        static_cast<Float>(p.repelling->attempts) / static_cast<Float>(p.lambda);
    met = average_repelling >= tolerance;
}

Criteria Criteria::get(const parameters::Modules modules, const int lambda)
{
    vCriteria criteria{std::make_shared<restart::UnableToAdapt>()};

    if (modules.restart_strategy >= parameters::RestartStrategyType::RESTART)
    {
        criteria.push_back(std::make_shared<restart::MinSigma>());
        criteria.push_back(std::make_shared<restart::MaxSigma>());
        criteria.push_back(std::make_shared<restart::ExceededMaxIter>());
        criteria.push_back(std::make_shared<restart::NoImprovement>());
        criteria.push_back(std::make_shared<restart::Stagnation>());
        criteria.push_back(std::make_shared<restart::MinDSigma>());
        criteria.push_back(std::make_shared<restart::MaxDSigma>());

        if (lambda > 1)
            criteria.push_back(std::make_shared<restart::FlatFitness>());

        //! TODO: make these compatible with other MA
        if (modules.matrix_adaptation == parameters::MatrixAdaptationType::COVARIANCE)
        {
            criteria.push_back(std::make_shared<restart::TolX>());
            criteria.push_back(std::make_shared<restart::ConditionC>());
            criteria.push_back(std::make_shared<restart::NoEffectAxis>());
            criteria.push_back(std::make_shared<restart::NoEffectCoord>());
        }
        if (modules.repelling_type != parameters::RepellingType::NONE)
        {
            criteria.push_back(std::make_shared<restart::TooMuchRepelling>());
        }
    }

    return Criteria(criteria);
}
} // namespace restart