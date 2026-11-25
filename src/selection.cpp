#include "selection.hpp"
#include "parameters.hpp"

namespace selection
{
Strategy::Strategy(const parameters::Modules &modules) :
    pairwise(modules.mirrored == parameters::Mirror::PAIRWISE ? std::make_shared<Pairwise>()
                                                              : std::make_shared<NoPairwise>()),
    elitsm(modules.elitist ? std::make_shared<Elitsm>() : std::make_shared<NoElitsm>())
{
}

void Strategy::select(parameters::Parameters &p) const
{
    (*pairwise)(p);

    (*elitsm)(p);

    p.pop.sort();
    p.pop.resize_cols(p.lambda);

    p.stats.current_avg = p.pop.f.head(p.pop.n).array().mean();
    p.stats.update_best(p.pop.X_transformed(Eigen::all, 0), p.pop.f(0));
    // old_pop is the previous generation, sorted (infinite before the first generation of a run)
    p.stats.update_success(p.pop.f(0) < p.old_pop.f(0));
}

void Pairwise::operator()(parameters::Parameters &p) const
{
    assert(p.pop.n % 2 == 0);
    for (size_t i = 0; i + 1 < p.pop.n; i += 2)
    {
        Eigen::Index idx = i + (1 * (p.pop.f(i) < p.pop.f(i + 1)));
        p.pop.f(idx) = std::numeric_limits<Float>::infinity();
    }
}

void Elitsm::operator()(parameters::Parameters &p) const
{
    // No parents in the first generation of a run (old_pop is then empty: infinite values)
    if (p.stats.t == 0 || p.old_pop.n_finite() == 0)
        return;

    // (mu + lambda) selection (the framework paper, Sec. 4.1): the mu parents, the selected
    // individuals of the previous generation (the first columns of old_pop, which is sorted),
    // compete with the offspring. old_pop keeps all its columns: MSR and PSR compare with it.
    const size_t n_old = p.old_pop.n;
    const auto k = static_cast<Eigen::Index>(std::min(p.mu, p.old_pop.n_finite()));

    if (!p.settings.one_plus_one)
    {
        // y and z of the parents are recomputed in the current distribution, with the step size
        // of this generation, as for a repaired point (Eq. 3 of the paper): y = (x - m) / sigma,
        // z = A^-1 y. (The integer variables keep their lower bound on the step size.)
        p.old_pop.S.leftCols(k).setConstant(p.mutation->sigma);
        mutation::SigmaSampler::apply_integer_bounds(p, p.old_pop.S, 0, k);
        for (Eigen::Index i = 0; i < k; i++)
        {
            p.adaptation->invert_x_into(p.old_pop.X.col(i), p.old_pop.S.col(i), p.old_pop.Y.col(i));
            p.adaptation->invert_y_into(p.old_pop.Y.col(i), p.old_pop.Z.col(i));
        }
    }
    p.old_pop.n = static_cast<size_t>(k);
    p.pop += p.old_pop;
    p.old_pop.n = n_old;
}
} // namespace selection
