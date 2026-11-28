#include "center_placement.hpp"

#include "parameters.hpp"

namespace center
{
namespace
{
//! A point uniform in (0, 1)^d from the optimizer's generator (B6: it used std::rand, which only
//! the global set_seed seeds)
Vector uniform_vector(const size_t d, rng::Generator &gen)
{
    rng::uniform<Float> uniform;
    Vector u(d);
    for (auto &ui : u)
        ui = uniform(gen);
    return u;
}

//! The candidates of NOVELTY_WEIGHTED and MAXIMIN_TABOO
constexpr int n_candidates = 512;

//! A point uniform in the region, from the optimizer's generator
Vector uniform_point(const Region &region, parameters::Parameters &p)
{
    return region.point(uniform_vector(p.settings.dim, *p.generator));
}

//! The distance from x to the nearest point of the repelling archive, with each coordinate divided
//! by the width of the region
Float nearest_taboo_point(const Vector &x, const Region &region, const parameters::Parameters &p)
{
    Float nearest = std::numeric_limits<Float>::infinity();
    for (const auto &point : p.repelling->archive)
    {
        const Vector diff = (x - point.solution.x).cwiseQuotient(region.width);
        nearest = std::min(nearest, diff.norm());
    }
    return nearest;
}
} // namespace

Region::Region(const parameters::Settings &settings) : lower(settings.lb), width(settings.db)
{
    const Float h = infinite_box_half_width * settings.sigma0;
    const Vector &x0 = settings.x0 ? *settings.x0 : settings.center;
    for (Eigen::Index j = 0; j < lower.size(); ++j)
    {
        if (std::isfinite(width(j)))
            continue;
        const Float lb = settings.lb(j), ub = settings.ub(j);
        if (std::isfinite(lb))
            lower(j) = std::max(x0(j) - h, lb);
        else if (std::isfinite(ub))
            lower(j) = std::min(x0(j) + h, ub) - 2 * h;
        else
            lower(j) = x0(j) - h;
        width(j) = 2 * h;
    }
}

void X0::operator()(parameters::Parameters &p)
{
    p.adaptation->m = p.settings.x0.value_or(p.settings.center);
}

void Uniform::operator()(parameters::Parameters &p)
{
    p.adaptation->m = uniform_point(Region(p.settings), p);
}

void Zero::operator()(parameters::Parameters &p) { p.adaptation->m.setZero(); }

void Center::operator()(parameters::Parameters &p) { p.adaptation->m = p.settings.center; }

void MaximinTaboo::operator()(parameters::Parameters &p)
{
    const Region region(p.settings);

    if (p.repelling->archive.empty())
    {
        p.adaptation->m = uniform_point(region, p);
        return;
    }

    Vector best_x = p.settings.center;
    Float best_score = -std::numeric_limits<Float>::infinity();

    for (int i = 0; i < n_candidates; ++i)
    {
        const Vector x = uniform_point(region, p);
        const Float nearest = nearest_taboo_point(x, region, p);

        if (nearest > best_score)
        {
            best_score = nearest;
            best_x = x;
        }
    }

    p.adaptation->m = best_x;
}

void NoveltyWeighted::operator()(parameters::Parameters &p)
{
    const Float temperature = 2.0;
    const Region region(p.settings);

    if (p.repelling->archive.empty())
    {
        p.adaptation->m = uniform_point(region, p);
        return;
    }

    std::vector<Vector> candidates;
    std::vector<Float> scores;

    candidates.reserve(n_candidates);
    scores.reserve(n_candidates);

    for (int i = 0; i < n_candidates; ++i)
    {
        const Vector x = uniform_point(region, p);
        const Float nearest = nearest_taboo_point(x, region, p);

        const Float score = std::pow(std::max<Float>(nearest, 1e-12), temperature);

        candidates.push_back(x);
        scores.push_back(score);
    }

    const Float total_score =
        std::accumulate(scores.begin(), scores.end(), static_cast<Float>(0.0));

    if (!(total_score > 0.0) || !std::isfinite(total_score))
    {
        // rng::below, from the raw outputs of the generator, not std::uniform_int_distribution,
        // whose algorithm is implementation-defined (#164)
        const auto idx = rng::below(static_cast<uint32_t>(n_candidates), *p.generator);
        p.adaptation->m = candidates[idx];
        return;
    }

    const Float r = rng::uniform<Float>{}(*p.generator) * total_score;

    Float cumsum = 0.0;

    for (int i = 0; i < n_candidates; ++i)
    {
        cumsum += scores[i];

        if (r <= cumsum)
        {
            p.adaptation->m = candidates[i];
            return;
        }
    }

    p.adaptation->m = candidates.back();
}
} // namespace center
