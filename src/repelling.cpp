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

} // namespace distance

Matrix archived_metric(parameters::Parameters &p)
{
    if (p.settings.modules.repelling_metric != parameters::RepellingMetric::ARCHIVED)
        return Matrix();

    const Eigen::Index d = static_cast<Eigen::Index>(p.settings.dim);
    Matrix inverse(d, d);
    for (Eigen::Index j = 0; j < d; j++)
        inverse.col(j) = p.adaptation->invert_y(Vector::Unit(d, j));

    // Unit determinant: keep the shape of the basin; the current sigma sets the scale
    const Eigen::PartialPivLU<Matrix> lu(inverse);
    const Float log_abs_det = lu.matrixLU().diagonal().array().abs().log().sum();
    if (!std::isfinite(log_abs_det) || !inverse.allFinite())
        return Matrix(); // degenerate: this point falls back to the Euclidean distance
    return inverse * std::exp(-log_abs_det / static_cast<Float>(d));
}

Float TabooPoint::distance(Eigen::Ref<const Vector> x, const parameters::Parameters &p) const
{
    Float delta;
    switch (p.settings.modules.repelling_metric)
    {
    case parameters::RepellingMetric::CURRENT:
        delta = p.adaptation->distance(x, solution.x);
        break;
    case parameters::RepellingMetric::ARCHIVED:
        if (metric.size() != 0)
        {
            delta = (metric * (x - solution.x)).norm();
            break;
        }
        [[fallthrough]];
    case parameters::RepellingMetric::EUCLIDEAN:
    default:
        delta = (x - solution.x).norm();
        break;
    }
    return delta / p.mutation->sigma;
}

bool TabooPoint::rejects(Eigen::Ref<const Vector> xi,
                         const parameters::Parameters &p,
                         const int attempts) const
{
    const Float rejection_radius = std::pow(shrinkage, attempts) * radius;

    const Float delta_xi = distance(xi, p);

    return delta_xi < rejection_radius;
}

void TabooPoint::calculate_criticality(const parameters::Parameters &p)
{
    const Float delta_m = distance(p.adaptation->m, p);

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

bool Repelling::is_rejected(Eigen::Ref<const Vector> xi, parameters::Parameters &p)
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

static bool is_valid_candidate(const Solution &solution, const parameters::Parameters &p)
{
    return solution.x.size() == static_cast<Eigen::Index>(p.settings.dim) &&
           solution.x.allFinite() && std::isfinite(solution.y);
}

static std::optional<Solution> get_candidate(const parameters::Parameters &p)
{
    if (p.stats.centers.empty() || p.stats.solutions.empty())
        return std::nullopt;

    const auto &center = p.stats.centers.back();
    const auto &solution = p.stats.solutions.back();

    const bool valid_center = is_valid_candidate(center, p);

    const bool valid_solution = is_valid_candidate(solution, p);

    if (!valid_center && !valid_solution)
        return std::nullopt;

    if (!valid_solution)
        return center;

    if (!valid_center)
        return solution;

    return center.y < solution.y ? center : solution;
}

// -----------------------------------------------------------------------------
// The archive update after a restart: hill-valley tests of the candidate against every taboo
// point (Maree et al. 2018), one point at a time (has_query, query_point, answer). The optimizer
// evaluates the points (ModularCMAES::advance, the RESTART step)
// -----------------------------------------------------------------------------

void Repelling::begin_update(parameters::Parameters &p)
{
    query = Query{};
    if (!updates_archive())
        return;

    query.candidate = get_candidate(p);
    if (!query.candidate.has_value())
        return;

    on_update_begin(*query.candidate, p);
    query.active = true;
    next_query(p);
}

void Repelling::next_query(parameters::Parameters &p)
{
    // The remaining tests are skipped when the next point would leave too little budget for the
    // first generation after the restart (the run ends soon after: the archive keeps the candidate
    // unless a test so far found a shared basin)
    if (query.j >= archive.size() || !p.restart_fits())
    {
        query.active = false;
        on_update_end(*query.candidate, query.accept, p);
        query.candidate.reset();
        return;
    }
    const Solution &u = *query.candidate;
    const Solution &v = archive[query.j].solution;
    const Float a =
        static_cast<Float>(query.k) / (static_cast<Float>(hill_valley_points) + 1.0);
    // One point at a time (has_query, query_point): the next depends on this one's value (answer)
    query.x =
        p.coordinate_mapping->transform(v.x + a * (u.x - v.x), p.settings.lb, p.settings.ub);
}

void Repelling::answer(const Float y, Eigen::Ref<const Vector> x, parameters::Parameters &p)
{
    if (!query.active)
        throw std::logic_error("Repelling::answer: no hill-valley point is pending");

    const Solution &u = *query.candidate;
    TabooPoint &point = archive[query.j];
    const Float max_f = std::max(u.y, point.solution.y);

    if (max_f < y)
    {
        // A hill between them: not the same basin, test the next point
        query.j++;
        query.k = 1;
    }
    else
    {
        p.stats.update_best(x, y);
        if (++query.k > hill_valley_points)
        {
            query.accept = false;
            query.j = on_shared_basin(point, u, p) ? query.j + 1 : archive.size();
            query.k = 1;
        }
    }
    next_query(p);
}

// -----------------------------------------------------------------------------
// CoverageRepelling: previous implementation
// -----------------------------------------------------------------------------

bool CoverageRepelling::on_shared_basin(TabooPoint &point,
                                        const Solution &candidate,
                                        parameters::Parameters &)
{
    point.n_rep++;
    if (point.solution > candidate)
        point.solution = candidate;
    return true;
}

void CoverageRepelling::on_update_end(const Solution &candidate_point,
                                      const bool accept_candidate,
                                      parameters::Parameters &p)
{
    if (accept_candidate)
    {
        archive.emplace_back(candidate_point, 1.0, archived_metric(p));
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

void AdaptiveRepelling::on_update_begin(const Solution &, parameters::Parameters &p)
{
    const int current_evaluations = static_cast<int>(p.stats.evaluations);

    // The evaluations of the run that ended, since the previous update (0 before the first): the
    // first run counts its full length. It counted as 1 evaluation (#121), which made the waste
    // ratio of the first shared basins about one run length too high.
    restart_evals =
        std::max<Float>(1.0, static_cast<Float>(current_evaluations - last_restart_evaluations));
    last_restart_evaluations = current_evaluations;

    average_restart_evaluations = (1.0 - restart_eval_alpha) * average_restart_evaluations +
                                  restart_eval_alpha * restart_evals;
}

bool AdaptiveRepelling::on_shared_basin(TabooPoint &point,
                                        const Solution &candidate_point,
                                        parameters::Parameters &)
{
    point.n_rep++;

    point.duplicate_evaluations += restart_evals;

    const Float waste_ratio = restart_evals / std::max<Float>(1.0, average_restart_evaluations);

    const Float capped_waste_ratio = std::min<Float>(waste_ratio, max_waste_ratio);

    const Float growth = std::exp(grow_eta * capped_waste_ratio);

    point.radius *= growth;
    point.clamp_radius();

    if (point.solution > candidate_point)
    {
        point.solution = candidate_point;
    }

    // Only the first point that shares the basin
    return false;
}

void AdaptiveRepelling::on_update_end(const Solution &candidate_point,
                                      const bool accept_candidate,
                                      parameters::Parameters &p)
{
    if (accept_candidate)
    {
        archive.emplace_back(
            candidate_point, initial_radius(p), min_radius(p), max_radius(p), archived_metric(p));
    }
}

} // namespace repelling