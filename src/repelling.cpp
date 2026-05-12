#include "repelling.hpp"

#include <algorithm>
#include <cmath>

#include "parameters.hpp"

namespace repelling
{

namespace distance
{

Float manhattan(const Vector &u, const Vector &v) { return (u - v).cwiseAbs().sum(); }

Float euclidian(const Vector &u, const Vector &v) { return (u - v).norm(); }

Float mahanolobis(const Vector &u, const Vector &v, const Matrix &C_inv)
{
    const auto delta = u - v;
    return std::sqrt(delta.transpose() * C_inv * delta);
}

//! Returns true when u belongs to the same basin as v.
bool hill_valley_test_p(const Solution &u,
                        const Solution &v,
                        FunctionType &f,
                        const size_t n_evals,
                        parameters::Parameters &p)
{
    const Float max_f = std::max(u.y, v.y);

    for (size_t k = 1; k < n_evals + 1; k++)
    {
        const Float a = static_cast<Float>(k) / (static_cast<Float>(n_evals) + 1.0);

        const Vector x = p.coordinate_mapping->transform(v.x + a * (u.x - v.x));

        const Float y = f(x);

        p.stats.evaluations++;

        if (max_f < y)
            return false;

        p.stats.update_best(x, y);
    }

    return true;
}

bool hill_valley_test(const Solution &u, const Solution &v, FunctionType &f, const size_t n_evals)
{
    const Float max_f = std::max(u.y, v.y);

    for (size_t k = 1; k < n_evals + 1; k++)
    {
        const Float a = static_cast<Float>(k) / (static_cast<Float>(n_evals) + 1.0);

        const Vector x = v.x + a * (u.x - v.x);
        const Float y = f(x);

        if (max_f < y)
            return false;
    }

    return true;
}

} // namespace distance

bool TabooPoint::rejects(const Vector &xi,
                         const parameters::Parameters &p,
                         const int attempts) const
{
    const Float rejection_radius = std::pow(shrinkage, attempts) * radius;

    const Float delta_xi = p.adaptation->distance(xi, solution.x) / p.mutation->sigma;

    return delta_xi < rejection_radius;
}

bool TabooPoint::shares_basin(FunctionType &objective,
                              const Solution &sol,
                              parameters::Parameters &p) const
{
    return distance::hill_valley_test_p(sol, solution, objective, 10, p);
}

void TabooPoint::calculate_criticality(const parameters::Parameters &p)
{
    const Float delta_m = p.adaptation->distance_from_center(solution.x) / p.mutation->sigma;

    const auto u = delta_m + radius;
    const auto l = delta_m - radius;

    criticality = cdf(u) - cdf(l);
}

// -----------------------------------------------------------------------------
// Base Repelling
// -----------------------------------------------------------------------------

void Repelling::sort_archive_by_criticality(const parameters::Parameters &p)
{
    for (auto &point : archive)
    {
        point.calculate_criticality(p);
    }

    std::sort(archive.begin(), archive.end(), [](const TabooPoint &a, const TabooPoint &b) {
        return a.criticality > b.criticality;
    });
}

void Repelling::prepare_sampling(parameters::Parameters &p)
{
    attempts = 0;
    sort_archive_by_criticality(p);
}

void Repelling::adapt_radii_after_sampling(parameters::Parameters & /*p*/)
{
    // Base implementation intentionally does nothing.
}

void Repelling::update_archive(FunctionType & /*objective*/, parameters::Parameters & /*p*/
)
{
    // Base implementation intentionally does nothing.
}

bool Repelling::is_rejected(const Vector &xi, parameters::Parameters &p)
{
    static constexpr Float criticality_threshold = 0.01;

    for (auto &point : archive)
    {
        // This assumes archive is sorted by criticality.
        if (point.criticality < criticality_threshold)
            break;

        point.checked_count++;

        if (point.rejects(xi, p, attempts))
        {
            point.rejected_count++;
            attempts++;
            return true;
        }
    }

    return false;
}

static Solution get_candidate(parameters::Parameters &p)
{
    const auto candidate_center = p.stats.centers.back();
    const auto candidate_point = p.stats.solutions.back();
    if (candidate_center.y < candidate_point.y)
        return candidate_center;
    return candidate_point;
}

// -----------------------------------------------------------------------------
// CoverageRepelling: previous implementation
// -----------------------------------------------------------------------------

void CoverageRepelling::update_archive(FunctionType &objective, parameters::Parameters &p)
{
    const auto candidate_point = get_candidate(p);

    bool accept_candidate = true;

    for (auto &point : archive)
    {
        if (point.shares_basin(objective, candidate_point, p))
        {
            point.n_rep++;
            accept_candidate = false;

            if (point.solution > candidate_point)
            {
                point.solution = candidate_point;
            }
        }
    }

    if (accept_candidate)
    {
        archive.emplace_back(candidate_point, 1.0);
    }

    const Float n_solutions = static_cast<Float>(std::max<size_t>(1, p.stats.solutions.size()));

    const Float volume_per_n = p.settings.volume / (p.settings.sigma0 * coverage * n_solutions);

    const Float n = p.adaptation->dd;

    const Float gamma_f = std::pow(std::tgamma(n / 2.0 + 1.0), 1.0 / n) / std::sqrt(M_PI);

    for (auto &point : archive)
    {
        point.radius = (std::pow(volume_per_n * point.n_rep, 1.0 / n) * gamma_f) / std::sqrt(n);
    }
}

// -----------------------------------------------------------------------------
// AdaptiveRepelling
// -----------------------------------------------------------------------------

Float AdaptiveRepelling::initial_radius(const parameters::Parameters &p) const
{
    const Float d = static_cast<Float>(p.settings.dim);
    return initial_radius_factor / std::sqrt(d);
}

Float AdaptiveRepelling::min_radius(const parameters::Parameters &p) const
{
    const Float d = static_cast<Float>(p.settings.dim);
    return min_radius_factor / std::sqrt(d);
}

Float AdaptiveRepelling::max_radius(const parameters::Parameters &p) const
{
    const Float d = static_cast<Float>(p.settings.dim);
    return max_radius_factor / std::sqrt(d);
}

void AdaptiveRepelling::prepare_sampling(parameters::Parameters &p)
{
    // Uses rejection statistics from the previous sampling phase.
    adapt_radii_after_sampling(p);

    attempts = 0;
    sort_archive_by_criticality(p);
}

void AdaptiveRepelling::adapt_radii_after_sampling(parameters::Parameters &p)
{
    const Float accepted = static_cast<Float>(std::max<int>(1, p.lambda));

    const Float rejected = static_cast<Float>(attempts);

    const Float global_rejection_rate = rejected / std::max<Float>(1.0, accepted + rejected);

    const bool global_pressure_too_high = global_rejection_rate > max_rejection_rate;

    for (auto &point : archive)
    {
        if (point.checked_count < min_checks_before_local_shrink)
        {
            point.last_rejection_rate = 0.0;
            point.rejected_count = 0;
            point.checked_count = 0;
            continue;
        }

        point.last_rejection_rate = static_cast<Float>(point.rejected_count) /
                                    std::max<Float>(1.0, static_cast<Float>(point.checked_count));

        const bool local_pressure_too_high = point.last_rejection_rate > local_max_rejection_rate;

        if (point.criticality >= 0.01 && global_pressure_too_high && local_pressure_too_high)
        {
            point.radius *= shrink_factor;
            point.clamp_radius();
        }

        point.rejected_count = 0;
        point.checked_count = 0;
    }
}

void AdaptiveRepelling::update_archive(FunctionType &objective, parameters::Parameters &p)
{
    const auto candidate_point = get_candidate(p);

    const int current_evaluations = p.stats.evaluations;

    Float restart_evals = 0.0;

    if (last_restart_evaluations > 0)
    {
        restart_evals = static_cast<Float>(current_evaluations - last_restart_evaluations);
    }

    last_restart_evaluations = current_evaluations;
    restart_evals = std::max<Float>(1.0, restart_evals);

    average_restart_evaluations = (1.0 - restart_eval_alpha) * average_restart_evaluations +
                                  restart_eval_alpha * restart_evals;

    bool accept_candidate = true;

    for (auto &point : archive)
    {
        if (point.shares_basin(objective, candidate_point, p))
        {
            point.n_rep++;
            accept_candidate = false;

            point.duplicate_evaluations += restart_evals;

            const Float waste_ratio =
                restart_evals / std::max<Float>(1.0, average_restart_evaluations);

            const Float capped_waste_ratio = std::min<Float>(waste_ratio, max_waste_ratio);

            const Float growth = std::exp(grow_eta * capped_waste_ratio);

            point.radius *= growth;
            point.clamp_radius();

            if (point.solution > candidate_point)
            {
                point.solution = candidate_point;
            }

            break;
        }
    }

    if (accept_candidate)
    {
        archive.emplace_back(candidate_point, initial_radius(p), min_radius(p), max_radius(p));
    }
}

} // namespace repelling