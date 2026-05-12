#include "center_placement.hpp"

#include "parameters.hpp"

namespace center
{
void X0::operator()(parameters::Parameters &p)
{
    p.adaptation->m = p.settings.x0.value_or(p.settings.center);
}

void Uniform::operator()(parameters::Parameters &p)
{
    const Vector u = ((Vector::Random(p.settings.dim).array() + 1.0) * 0.5).matrix();

    p.adaptation->m = p.settings.lb + u.cwiseProduct(p.settings.db);
}

void Zero::operator()(parameters::Parameters &p) { p.adaptation->m.setZero(); }

void Center::operator()(parameters::Parameters &p) { p.adaptation->m = p.settings.center; }

void MaximinTaboo::operator()(parameters::Parameters &p)
{
    const int dim = p.settings.dim;
    const int n_candidates = 512;

    if (p.repelling->archive.empty())
    {
        const Vector u = ((Vector::Random(dim).array() + 1.0) * 0.5).matrix();

        p.adaptation->m = p.settings.lb + u.cwiseProduct(p.settings.db);

        return;
    }

    Vector best_x = p.settings.center;
    Float best_score = -std::numeric_limits<Float>::infinity();

    for (int i = 0; i < n_candidates; ++i)
    {
        const Vector u = ((Vector::Random(dim).array() + 1.0) * 0.5).matrix();

        const Vector x = p.settings.lb + u.cwiseProduct(p.settings.db);

        Float nearest = std::numeric_limits<Float>::infinity();

        for (const auto &point : p.repelling->archive)
        {
            const Vector diff = (x - point.solution.x).cwiseQuotient(p.settings.db);

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

    if (p.repelling->archive.empty())
    {
        const Vector u = ((Vector::Random(dim).array() + 1.0) * 0.5).matrix();

        p.adaptation->m = p.settings.lb + u.cwiseProduct(p.settings.db);

        return;
    }

    std::vector<Vector> candidates;
    std::vector<Float> scores;

    candidates.reserve(n_candidates);
    scores.reserve(n_candidates);

    for (int i = 0; i < n_candidates; ++i)
    {
        const Vector u = ((Vector::Random(dim).array() + 1.0) * 0.5).matrix();

        const Vector x = p.settings.lb + u.cwiseProduct(p.settings.db);

        Float nearest = std::numeric_limits<Float>::infinity();

        for (const auto &point : p.repelling->archive)
        {
            const Vector diff = (x - point.solution.x).cwiseQuotient(p.settings.db);

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
        const int idx = std::rand() % n_candidates;
        p.adaptation->m = candidates[idx];
        return;
    }

    const Float r =
        ((static_cast<Float>(std::rand()) / static_cast<Float>(RAND_MAX))) * total_score;

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
