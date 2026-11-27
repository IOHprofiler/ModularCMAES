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
    // Elitism appends the parents after the offspring; the pairs are the offspring's
    const size_t n_offspring = p.pop.n;
    (*elitsm)(p);
    pairwise->sort(p.pop, n_offspring);
    p.pop.resize_cols(p.lambda);

    p.stats.current_avg = p.pop.f.head(p.pop.n).array().mean();
    // A success: the best of the generation beats the best of the run so far (infinite at the start
    // of a run). The best of the previous generation, as the framework paper describes SR, does not
    // work without elitism: it is beaten about every other generation even at a good sigma, far
    // above the target 2/11, and sigma diverges (#83). Counted once per generation.
    const bool success = p.pop.f(0) < p.stats.current_best.y;
    p.stats.update_best(p.pop.X_transformed(Eigen::all, 0), p.pop.f(0));
    p.stats.update_success(success);
}

void Pairwise::operator()(parameters::Parameters &p) const { sort(p.pop, p.pop.n); }

void Pairwise::sort(Population &pop, const size_t n_offspring) const
{
    const size_t n = pop.n;
    const auto f = [&pop](const size_t i) { return pop.f(static_cast<Eigen::Index>(i)); };

    // The loser of each pair: the second when the first is better (a NaN value is worse than any
    // number), the first otherwise (on a tie, as before #75). An odd last offspring is unpaired.
    loser.assign(n, 0);
    size_t n_losers = 0;
    for (size_t i = 0; i + 1 < std::min(n_offspring, n); i += 2, ++n_losers)
    {
        const bool first_better = f(i) < f(i + 1) || (std::isnan(f(i + 1)) && !std::isnan(f(i)));
        loser[first_better ? i + 1 : i] = 1;
    }

    // The ranked individuals by f, then the losers by f; within each, ties in index order and NaN
    // last, as Population::sort
    order.resize(n);
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](const size_t a, const size_t b) {
        if (loser[a] != loser[b])
            return loser[a] < loser[b];
        const Float fa = f(a), fb = f(b);
        const bool nan_a = std::isnan(fa), nan_b = std::isnan(fb);
        if (nan_a != nan_b)
            return nan_b;
        if (!nan_a && fa != fb)
            return fa < fb;
        return a < b;
    });
    pop.reorder(order);
    pop.n_ranked = n - n_losers;
}

void NoPairwise::sort(Population &pop, const size_t /*n_offspring*/) const { pop.sort(); }

void Elitsm::operator()(parameters::Parameters &p) const
{
    // No parents in the first generation of a run (old_pop is then empty: infinite values)
    if (p.stats.t == 0 || p.old_pop.n_finite() == 0)
        return;

    // (mu + lambda) selection (the framework paper, Sec. 4.1): the mu parents, the selected
    // individuals of the previous generation (the first columns of old_pop, which is sorted),
    // compete with the offspring. old_pop keeps all its columns: MSR and PSR compare with it.
    const size_t n_old = p.old_pop.n;
    // Not the losers of pairwise selection, which follow the ranked individuals
    const auto k =
        static_cast<Eigen::Index>(std::min({p.mu, p.old_pop.n_finite(), p.old_pop.ranked()}));

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
