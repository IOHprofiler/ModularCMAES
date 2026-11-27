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
    // The restart strategy may limit the run further: BIPOP's small runs (#80)
    if (p.restart_strategy)
        max_iter = std::min(max_iter, p.restart_strategy->max_iter);
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
/*
 * The longest and the shortest axis of the sampled distribution, without sigma: max(D) and min(D),
 * the square roots of the extreme eigenvalues of C, from metric() where it costs nothing
 * (COVARIANCE, SEPARABLE, NONE). The others would decompose their factor every generation, O(d^3);
 * they use sqrt(max C_ii) and sqrt(min C_ii), the diagonal of C (coordinate_wise_variances) that
 * every adaptation keeps (decided by the maintainer for MaxDSigma and MinDSigma, #80). The diagonal
 * bounds the eigenvalues, max C_ii <= lambda_max and min C_ii >= lambda_min, so the criteria fire
 * earlier (MaxDSigma, MinDSigma: no source does this) or later (MaxSigma, which becomes pycma's
 * tolfacupx, sigma sqrt(C_ii) > tol sigma0 for some i) than with the eigenvalues. For SEPARABLE
 * and NONE the two agree: C is diagonal.
 */
Float longest_axis(const parameters::Parameters &p)
{
    if (p.adaptation->keeps_metric())
        return p.adaptation->metric().d.maxCoeff();
    return std::sqrt(p.adaptation->coordinate_wise_variances.maxCoeff());
}

Float shortest_axis(const parameters::Parameters &p)
{
    if (p.adaptation->keeps_metric())
        return p.adaptation->metric().d.minCoeff();
    return std::sqrt(p.adaptation->coordinate_wise_variances.minCoeff());
}
} // namespace

void MaxSigma::on_reset(const parameters::Parameters &p)
{
    start = p.mutation->sigma * longest_axis(p);
}

void MaxSigma::update(const parameters::Parameters &p)
{
    reset_if_new(p);
    // TolXUp: "stop if sigma x max(diag(D)) increased by more than 10^4" (Hansen 2016, App. B.3),
    // since the start of the run. It compared sigma with 1e4, whatever sigma0 and C (#80).
    met = p.mutation->sigma * longest_axis(p) > tolerance * start;
}

void TolX::update(const parameters::Parameters &p)
{
    reset_if_new(p);
    // Met when sigma sqrt(C_ii) < tol sigma_0 and sigma |pc_i| < tol sigma_0 for every i, with
    // sigma_0 the step size at the start of the run (Hansen 2016, App. B.3: "TolX to 10^-12 times
    // the initial sigma"; pycma's tolx). For every adaptation: C_ii from coordinate_wise_variances,
    // and pc where the adaptation has one (#80)
    const Float sigma = p.mutation->sigma;
    const Float tolx_condition = tolerance * sigma_start;
    const auto d = static_cast<Eigen::Index>(p.settings.dim);
    tolx_vector.head(d) = p.adaptation->coordinate_wise_variances.cwiseSqrt() * sigma;
    if (const Vector *pc = p.adaptation->rank_one_path(); pc && tolx_vector.size() == 2 * d)
        tolx_vector.tail(d) = pc->cwiseAbs() * sigma;
    met = (tolx_vector.array() < tolx_condition).all();
}

void TolX::on_reset(const parameters::Parameters &p)
{
    const auto d = static_cast<Eigen::Index>(p.settings.dim);
    tolx_vector = Vector::Ones(p.adaptation->rank_one_path() != nullptr ? 2 * d : d);
    sigma_start = p.mutation->sigma;
}

void MaxDSigma::update(const parameters::Parameters &p)
{
    // pycma's tolupsigma: sigma / sigma0 > tolerance max(D), with D the square roots of the
    // eigenvalues of C (B15: not their fourth roots)
    const Float d_sigma = p.mutation->sigma / p.settings.sigma0;
    met = d_sigma > (tolerance * longest_axis(p));
}

void MinDSigma::update(const parameters::Parameters &p)
{
    // As MaxDSigma, with min(D)
    const Float d_sigma = p.mutation->sigma / p.settings.sigma0;
    met = d_sigma < (tolerance * shortest_axis(p));
}

void ConditionC::update(const parameters::Parameters &p)
{
    // pycma's tolconditioncov: the condition of C, max(D)^2 / min(D)^2, above 1e14. From metric():
    // created by default only where it costs nothing (#80)
    const auto metric = p.adaptation->metric();
    const Float condition_c = pow(metric.d.maxCoeff(), 2.0) / pow(metric.d.minCoeff(), 2);
    met = condition_c > tolerance;
}

void NoEffectAxis::update(const parameters::Parameters &p)
{
    // pycma's noeffectaxis: all(m == m + 0.1 sigma D_i B[:, i]), i = t mod d, in floating point
    // (B15: D_i the square root of the eigenvalue). It compared the step with 10 epsilon (#80).
    const auto metric = p.adaptation->metric();
    const Vector &m = p.adaptation->m;
    const auto i = static_cast<Eigen::Index>(p.stats.t % p.settings.dim);
    const Float step = tolerance * p.mutation->sigma * metric.d(i);
    if (metric.is_diagonal())
    {
        met = m(i) + step == m(i);
        return;
    }
    met = true;
    for (Eigen::Index j = 0; j < m.size() && met; ++j)
        met = m(j) + step * (*metric.B)(j, i) == m(j);
}

void NoEffectCoord::update(const parameters::Parameters &p)
{
    // pycma's noeffectcoord: any(m == m + 0.2 sigma sqrt(C_ii)), in floating point, from the
    // diagonal of C that every adaptation keeps. It compared the step with 10 epsilon (#80).
    const Vector &m = p.adaptation->m;
    const Vector &variances = p.adaptation->coordinate_wise_variances;
    const Float sigma = p.mutation->sigma;
    met = false;
    for (Eigen::Index j = 0; j < m.size() && !met; ++j)
        met = m(j) + tolerance * (sigma * std::sqrt(variances(j))) == m(j);
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

        // TolX and NoEffectCoord read the diagonal of C, which every adaptation keeps; ConditionC
        // and NoEffectAxis read metric(), by default only where it costs nothing (#80; the
        // others would decompose their factor every generation, O(d^3))
        using parameters::MatrixAdaptationType;
        const bool keeps_metric = modules.matrix_adaptation == MatrixAdaptationType::COVARIANCE ||
                                  modules.matrix_adaptation == MatrixAdaptationType::SEPARABLE ||
                                  modules.matrix_adaptation == MatrixAdaptationType::NONE;
        criteria.push_back(std::make_shared<restart::TolX>());
        if (keeps_metric)
        {
            criteria.push_back(std::make_shared<restart::ConditionC>());
            criteria.push_back(std::make_shared<restart::NoEffectAxis>());
        }
        criteria.push_back(std::make_shared<restart::NoEffectCoord>());
        if (modules.repelling_type != parameters::RepellingType::NONE)
        {
            criteria.push_back(std::make_shared<restart::TooMuchRepelling>());
        }
    }

    return Criteria(criteria);
}
} // namespace restart