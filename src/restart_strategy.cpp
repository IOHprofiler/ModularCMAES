#include "restart_strategy.hpp"

#include "matrix_adaptation.hpp"
#include "parameters.hpp"

namespace restart
{

Float Strategy::update(parameters::Parameters &p) { return p.settings.sigma0; }

Float IPOP::update(parameters::Parameters &p)
{
    // The population grows by ipop_factor at every restart (Auger & Hansen 2005), up to
    // 2^9 lambda_0 (the framework paper, Sec. 6.1)
    const auto max_lambda = max_lambda_factor * p.settings.lambda0;
    if (p.lambda < max_lambda)
    {
        p.lambda = std::min(
            max_lambda,
            static_cast<size_t>(std::round(static_cast<Float>(p.lambda) * ipop_factor)));
        // mu of the new lambda, as BIPOP (#150): the default, lambda / 2 (pycma's popsize // 2;
        // the framework paper's Alg. 1 sets mu = floor(lambda_0 / 2) in its restart block, after
        // the strategy changed lambda_0), or the ratio of a given mu0. It kept the ratio mu /
        // lambda of the previous run, rounded
        p.mu = p.settings.mu_for(p.lambda);
    }
    return p.settings.sigma0;
}

Float BIPOP::update(parameters::Parameters &p)
{
    // The generator's own uniform in (0, 1), as the samplers: std::uniform_real_distribution is
    // implementation-defined, so the same seed gave other runs with another standard library (#164)
    rng::uniform<Float> dist;

    // The evaluations of the run that ended (with the center of mass and the hill-valley points of
    // the restart before it) go to its regime. The first run, with lambda_init, counts as small, as
    // in pycma's fmin ("the first run is accounted on the small budget"); Hansen (2009) disregards
    // it. Either way the first restart is large.
    const size_t last_used_budget =
        p.stats.evaluations > used_budget ? p.stats.evaluations - used_budget : 0;
    used_budget = p.stats.evaluations;
    (large_run ? used_large : used_small) += last_used_budget;

    // The regime that has used fewer evaluations runs next, the large one on a tie: a small run "if
    // and only if its recent budget is smaller than the one for the first regime" (Hansen 2009;
    // pycma: sum(small_i) < max(1, sum(large_i)))
    large_run = large();

    // A large run doubles the latest large population (the first run's lambda_init before the
    // first large run): the k-th large run has 2^k lambda_init, as pycma's popsize0 *
    // incpopsize^(irun - runs_with_small) (pycma 4.6.0, cma/evolution_strategy.py:5380-5381; its
    // popsize0 is the option as evaluated in the first run, with the default popsize the unrounded
    // 4 + 3 ln d, which gives 17, 35, 70, ... at d = 5). Up to max_lambda_factor lambda_init
    // (2^9 lambda_init), the limit of IPOP; after it the large runs keep that size (maintainer,
    // #140). pycma has no limit on the population, but its fmin stops altogether after `restarts`
    // large runs (evolution_strategy.py:5591; 9 with restarts=True, options_parameters.py:39), and
    // Hansen (2009) after nine ("the largest population size is lambda = 2^9 lambda_def"): their
    // largest population is this limit. modcma keeps restarting until the budget is used.
    const size_t max_lambda = std::max<size_t>(1, max_lambda_factor) * lambda_init;
    if (large_run)
        lambda_large = std::min(2 * std::max(lambda_large, lambda_init), max_lambda);

    // lambda_small = floor(lambda_init (lambda_large / lambda_init)^(U^2)): between lambda_init and
    // the latest large population, as pycma 4.6.0's fmin, int(popsize0 *
    // popsize_multiplier**(U**2)) with the multiplier of the latest large run: incpopsize^(irun -
    // runs_with_small) after runs_with_small += 1 (evolution_strategy.py:5367, 5370-5371). Hansen
    // (2009) writes floor(lambda_def (lambda_l / (2 lambda_def))^(U^2)), with lambda_l "the latest
    // population size from the first regime", and the range [lambda_def, lambda_l / 2]: half of
    // pycma's. modcma follows pycma (maintainer, #140). u is drawn at every restart; a large run
    // does not use lambda_small.
    const Float u = dist(*p.generator);
    lambda_small = static_cast<size_t>(std::floor(
        static_cast<Float>(lambda_init) *
        std::pow(static_cast<Float>(lambda_large) / static_cast<Float>(lambda_init), u * u)));

    // Rounded down, as Hansen (2009) and pycma; up to even only for PAIRWISE, whose selection
    // compares pairs (as Settings does with lambda0)
    if (p.settings.modules.mirrored == parameters::Mirror::PAIRWISE && lambda_small % 2 != 0)
        lambda_small++;

    p.lambda = std::max(size_t{2}, large_run ? lambda_large : lambda_small);
    // mu of the new lambda, as IPOP (#150): the default, lambda / 2 as pycma's fmin (each run
    // takes the default CMA_mu, popsize // 2) and Hansen (2009), or the ratio of a given mu0. It
    // was lambda mu0 / lambda0, truncated
    p.mu = p.settings.mu_for(p.lambda);

    // A small run uses about half the evaluations of the large regime so far: "a maximum number
    // of function evaluations of half of the recent large budget is enforced" (Hansen 2009). As
    // pycma 4.6.0's fmin (maintainer, #139 D3): maxiter = min(maxiter0, 0.5 * sum(large_i) /
    // popsize), not rounded (cma/evolution_strategy.py:5372), and a run stops when countiter >=
    // maxiter (4527-4529), after ceil(maxiter) generations. ExceededMaxIter stops after max_iter
    // generations, so max_iter is the rounded-up cap; it was rounded down, a generation shorter
    // when the cap is not an integer.
    max_iter = large_run ? std::numeric_limits<size_t>::max()
                         : static_cast<size_t>(std::ceil(0.5 * static_cast<Float>(used_large) /
                                                         static_cast<Float>(p.lambda)));

    return large_run ? p.settings.sigma0
                     : p.settings.sigma0 * std::pow(10, -2 * dist(*p.generator));
}
} // namespace restart
