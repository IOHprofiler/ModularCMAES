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

} // namespace

namespace restart
{
void SplitMedian::Median::rebalance()
{
    if (lo.size() > hi.size() + 1)
    {
        auto last = std::prev(lo.end());
        hi.insert(*last);
        lo.erase(last);
    }
    else if (hi.size() > lo.size())
    {
        lo.insert(*hi.begin());
        hi.erase(hi.begin());
    }
}

void SplitMedian::Median::insert(const Float x)
{
    if (lo.empty() || !NanLast{}(*lo.rbegin(), x))
        lo.insert(x); // x <= max(lo)
    else
        hi.insert(x);
    rebalance();
}

void SplitMedian::Median::erase(const Float x)
{
    if (!lo.empty() && !NanLast{}(*lo.rbegin(), x))
        lo.erase(lo.find(x));
    else
        hi.erase(hi.find(x));
    rebalance();
}

Float SplitMedian::Median::median() const
{
    if (lo.empty())
        return std::numeric_limits<Float>::quiet_NaN();
    if (lo.size() > hi.size())
        return *lo.rbegin();
    return 0.5 * (*lo.rbegin() + *hi.begin());
}

void SplitMedian::push_back(const Float x)
{
    values_.push_back(x);
    tail_.insert(x);
}

void SplitMedian::assign(const std::vector<Float> &values)
{
    values_.clear();
    head_ = {};
    tail_ = {};
    split_ = 0;
    for (const Float x : values)
        push_back(x);
}

void SplitMedian::set_split(const size_t split)
{
    const size_t target = std::min(split, values_.size());
    for (; split_ < target; ++split_)
    {
        tail_.erase(values_[split_]);
        head_.insert(values_[split_]);
    }
    for (; split_ > target; --split_)
    {
        head_.erase(values_[split_ - 1]);
        tail_.insert(values_[split_ - 1]);
    }
}
} // namespace restart

namespace
{
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
    was_reset = true;
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
    reset_if_new(p);
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
    reset_if_new(p);
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
    reset_if_new(p);
    met = false;
    if (p.lambda <= 1)
        return;

    const size_t time_since_restart = p.stats.t - last_restart;
    // With sequential selection the population can be shorter than lambda: its last individual
    // then (the columns after it are stale, #119)
    const auto index = std::min<size_t>(flat_fitness_index, p.pop.n - 1);
    flat_fitnesses(p.stats.t % p.settings.dim) = p.pop.f(0) == p.pop.f(index);
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

namespace
{
/**
 * The covariance adaptation, when B and d hold the eigendecomposition of C; null otherwise. B14:
 * COVARIANCE_NO_EIGV is a CovarianceAdaptation, but its B and d are not kept up to date (they are
 * the identity and ones, or the last metric() asked, #123), so the criteria must not read them.
 */
const matrix_adaptation::CovarianceAdaptation *eigendecomposition(const parameters::Parameters &p)
{
    const auto *covariance =
        dynamic_cast<const matrix_adaptation::CovarianceAdaptation *>(p.adaptation.get());
    return covariance != nullptr && covariance->has_eigendecomposition() ? covariance : nullptr;
}
} // namespace

void TolX::update(const parameters::Parameters &p)
{
    reset_if_new(p);
    // Met when sigma sqrt(C_ii) < tol sigma_0 and sigma |pc_i| < tol sigma_0 for every i, with
    // sigma_0 the step size at the start of the run (Hansen 2016, Appendix B; pycma's tolx)
    if (const auto dynamic =
            std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(p.adaptation))
    {
        const Float sigma = p.mutation->sigma;
        const Float tolx_condition = tolerance * sigma_start;
        tolx_vector.head(p.settings.dim) = dynamic->C.diagonal().cwiseSqrt() * sigma;
        tolx_vector.tail(p.settings.dim) = dynamic->pc.cwiseAbs() * sigma;
        met = (tolx_vector.array() < tolx_condition).all();
    }
}

void TolX::on_reset(const parameters::Parameters &p)
{
    tolx_vector = Vector::Ones(p.settings.dim * 2);
    sigma_start = p.mutation->sigma;
}

void MaxDSigma::update(const parameters::Parameters &p)
{
    const Float d_sigma = p.mutation->sigma / p.settings.sigma0;
    // B15: d holds the square roots of the eigenvalues of C (pycma's D): tolupsigma compares with
    // max(D) itself, not its square root
    Float root_max_d = 1.0;
    if (const auto *dynamic = eigendecomposition(p))
    {
        root_max_d = dynamic->d.maxCoeff();
    }
    met = d_sigma > (tolerance * root_max_d);
}

void MinDSigma::update(const parameters::Parameters &p)
{
    const Float d_sigma = p.mutation->sigma / p.settings.sigma0;

    // B15: as MaxDSigma, with min(D)
    Float root_min_d = 1.0;
    if (const auto *dynamic = eigendecomposition(p))
    {
        root_min_d = dynamic->d.minCoeff();
    }
    met = d_sigma < (tolerance * root_min_d);
}

void ConditionC::update(const parameters::Parameters &p)
{
    if (const auto *dynamic = eigendecomposition(p))
    {
        const Float condition_c = pow(dynamic->d.maxCoeff(), 2.0) / pow(dynamic->d.minCoeff(), 2);
        met = condition_c > tolerance;
    }
}

void NoEffectAxis::update(const parameters::Parameters &p)
{
    if (const auto *dynamic = eigendecomposition(p))
    {
        const Eigen::Index t = p.stats.t % p.settings.dim;
        // B15: as pycma, 0.1 sigma D_t b_t with D_t = d(t) the square root of the eigenvalue
        const auto effect_axis = 0.1 * p.mutation->sigma * dynamic->d(t) * dynamic->B.col(t);
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
    reset_if_new(p);
    const size_t time_since_restart = p.stats.t - last_restart;

    median_fitnesses.push_back(median_vector(p.pop.f.head(p.pop.n)));
    best_fitnesses.push_back(p.pop.f(0));

    met = false;

    if (time_since_restart <= n_stagnation)
        return;

    const size_t n = best_fitnesses.values().size();

    if (n < 4)
        return;

    size_t split = static_cast<size_t>(tolerance * static_cast<Float>(n));
    split = std::max<size_t>(1, std::min(split, n - 1));

    // The medians of [0, split) and [split, n), kept up to date as the split moves (B10)
    best_fitnesses.set_split(split);
    median_fitnesses.set_split(split);

    const Float old_best_median = best_fitnesses.head_median();

    const Float new_best_median = best_fitnesses.tail_median();

    const Float old_median_median = median_fitnesses.head_median();

    const Float new_median_median = median_fitnesses.tail_median();

    // Minimization: if newer medians are not better, stagnation.
    const bool best_not_improving = new_best_median >= old_best_median;
    const bool median_not_improving = new_median_median >= old_median_median;

    met = best_not_improving && median_not_improving;
}

void Stagnation::on_reset(const parameters::Parameters &p)
{
    const auto d = static_cast<Float>(p.settings.dim);
    n_stagnation = static_cast<size_t>(100 + 100 * std::pow(d, 1.5) / static_cast<Float>(p.lambda));

    median_fitnesses.clear();
    best_fitnesses.clear();
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