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
    const Vector u = uniform_vector(p.settings.dim, *p.generator);

    p.adaptation->m = Region(p.settings).point(u);
}

void Zero::operator()(parameters::Parameters &p) { p.adaptation->m.setZero(); }

void Center::operator()(parameters::Parameters &p) { p.adaptation->m = p.settings.center; }

void MaximinTaboo::operator()(parameters::Parameters &p)
{
    const int dim = p.settings.dim;
    const int n_candidates = 512;
    const Region region(p.settings);

    if (p.repelling->archive.empty())
    {
        const Vector u = uniform_vector(dim, *p.generator);

        p.adaptation->m = region.point(u);

        return;
    }

    Vector best_x = p.settings.center;
    Float best_score = -std::numeric_limits<Float>::infinity();

    for (int i = 0; i < n_candidates; ++i)
    {
        const Vector u = uniform_vector(dim, *p.generator);

        const Vector x = region.point(u);

        Float nearest = std::numeric_limits<Float>::infinity();

        for (const auto &point : p.repelling->archive)
        {
            const Vector diff = (x - point.solution.x).cwiseQuotient(region.width);

            const Float d = diff.norm();

            nearest = std::min(nearest, d);
        }

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
    const int dim = p.settings.dim;
    const int n_candidates = 512;
    const Float temperature = 2.0;
    const Region region(p.settings);

    if (p.repelling->archive.empty())
    {
        const Vector u = uniform_vector(dim, *p.generator);

        p.adaptation->m = region.point(u);

        return;
    }

    std::vector<Vector> candidates;
    std::vector<Float> scores;

    candidates.reserve(n_candidates);
    scores.reserve(n_candidates);

    for (int i = 0; i < n_candidates; ++i)
    {
        const Vector u = uniform_vector(dim, *p.generator);

        const Vector x = region.point(u);

        Float nearest = std::numeric_limits<Float>::infinity();

        for (const auto &point : p.repelling->archive)
        {
            const Vector diff = (x - point.solution.x).cwiseQuotient(region.width);

            const Float d = diff.norm();

            nearest = std::min(nearest, d);
        }

        const Float score = std::pow(std::max<Float>(nearest, 1e-12), temperature);

        candidates.push_back(x);
        scores.push_back(score);
    }

    const Float total_score =
        std::accumulate(scores.begin(), scores.end(), static_cast<Float>(0.0));

    if (!(total_score > 0.0) || !std::isfinite(total_score))
    {
        // rng::random_integer, not std::uniform_int_distribution, whose algorithm is
        // implementation-defined (#164); with #162 it draws from the raw outputs (rng::below)
        const int idx = rng::random_integer(0, n_candidates, *p.generator);
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
