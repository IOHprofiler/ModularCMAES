#include "restart_strategy.hpp"

#include "matrix_adaptation.hpp"
#include "parameters.hpp"

namespace restart
{

Float Strategy::update(parameters::Parameters &p) { return p.settings.sigma0; }

Float IPOP::update(parameters::Parameters &p)
{
    // The population grows by ipop_factor at every restart (Auger & Hansen 2005), up to
    // 2^9 lambda_0 (the framework paper, Sec. 4.5)
    const auto max_lambda = max_lambda_factor * p.settings.lambda0;
    if (p.lambda < max_lambda)
    {
        const Float ratio = static_cast<Float>(p.mu) / static_cast<Float>(p.lambda);
        p.lambda = std::min(
            max_lambda,
            static_cast<size_t>(std::round(static_cast<Float>(p.lambda) * ipop_factor)));
        p.mu = std::max<size_t>(1, static_cast<size_t>(std::round(ratio * p.lambda)));
    }
    return p.settings.sigma0;
}

Float BIPOP::update(parameters::Parameters &p)
{
    std::uniform_real_distribution<> dist;

    // The evaluations of the run that ended (with the center of mass and the hill-valley points of
    // the restart before it) go to its regime
    const size_t last_used_budget =
        p.stats.evaluations > used_budget ? p.stats.evaluations - used_budget : 0;
    used_budget = p.stats.evaluations;
    if (lambda_large == 0)
    {
        // The first run, with lambda_init, counts as small, as in pycma's fmin ("the first run is
        // accounted on the small budget"); Hansen (2009) disregards it. Either way the first
        // restart is large.
        used_small += last_used_budget;
        lambda_large = lambda_init * 2;
    }
    else if (large_run)
    {
        used_large += last_used_budget;
        lambda_large *= 2;
    }
    else
        used_small += last_used_budget;

    // The regime that has used fewer evaluations runs next, the large one on a tie: a small run "if
    // and only if its recent budget is smaller than the one for the first regime" (Hansen 2009;
    // pycma: sum(small_i) < max(1, sum(large_i)))
    large_run = large();

    // lambda_small = floor(lambda_init (lambda_large / (2 lambda_init))^(U^2)): between lambda_init
    // and half the next large population, that is the last one (Hansen 2009; pycma)
    const Float u = dist(*p.generator);
    lambda_small = static_cast<size_t>(std::floor(
        static_cast<Float>(lambda_init) *
        std::pow(0.5 * static_cast<Float>(lambda_large) / static_cast<Float>(lambda_init), u * u)));

    // Rounded down, as Hansen (2009) and pycma; up to even only for PAIRWISE, whose selection
    // compares pairs (as Settings does with lambda0)
    if (p.settings.modules.mirrored == parameters::Mirror::PAIRWISE && lambda_small % 2 != 0)
        lambda_small++;

    p.lambda = std::max(size_t{2}, large_run ? lambda_large : lambda_small);
    p.mu = std::max(Float{1.0}, p.lambda * mu_factor);

    // A small run uses at most half the evaluations of the large regime so far: "a maximum number
    // of function evaluations of half of the recent large budget is enforced" (Hansen 2009); pycma
    // sets maxiter = min(maxiter0, 0.5 sum(large_i) / popsize). ExceededMaxIter enforces it.
    max_iter = large_run ? std::numeric_limits<size_t>::max()
                         : static_cast<size_t>(0.5 * static_cast<Float>(used_large) /
                                               static_cast<Float>(p.lambda));

    return large_run ? p.settings.sigma0
                     : p.settings.sigma0 * std::pow(10, -2 * dist(*p.generator));
}
} // namespace restart
