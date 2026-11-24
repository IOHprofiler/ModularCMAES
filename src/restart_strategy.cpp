#include "restart_strategy.hpp"

#include "matrix_adaptation.hpp"
#include "parameters.hpp"

namespace restart
{

Float Strategy::update(parameters::Parameters &p) { return p.settings.sigma0; }

Float IPOP::update(parameters::Parameters &p)
{
    // The population grows by ipop_factor at every restart (Auger & Hansen 2005), up to
    // (d lambda_0)^2, a limit from the initial population size
    const Float max_lambda = std::pow(static_cast<Float>(p.settings.dim * p.settings.lambda0), 2);
    if (static_cast<Float>(p.lambda) < max_lambda)
    {
        const Float ratio = static_cast<Float>(p.mu) / static_cast<Float>(p.lambda);
        p.lambda = static_cast<size_t>(std::round(static_cast<Float>(p.lambda) * ipop_factor));
        p.mu = std::max<size_t>(1, static_cast<size_t>(std::round(ratio * p.lambda)));
    }
    return p.settings.sigma0;
}

Float BIPOP::update(parameters::Parameters &p)
{
    std::uniform_real_distribution<> dist;

    const auto last_used_budget = p.stats.evaluations - used_budget;
    used_budget += last_used_budget;
    const auto remaining_budget = budget - used_budget;

    if (!lambda_large)
    {
        lambda_large = lambda_init * 2;
        budget_small = remaining_budget / 2;
        budget_large = remaining_budget - budget_small;
    }
    else if (large())
    {
        budget_large -= last_used_budget;
        lambda_large *= 2;
    }
    else
    {
        budget_small -= last_used_budget;
    }

    // lambda_small = floor(lambda_init (lambda_large / (2 lambda_init))^(U^2)): between lambda_init
    // and half the current large population (Hansen 2009; pycma)
    const Float u = dist(*p.generator);
    lambda_small = static_cast<size_t>(std::floor(
        static_cast<Float>(lambda_init) *
        std::pow(0.5 * static_cast<Float>(lambda_large) / static_cast<Float>(lambda_init), u * u)));

    if (lambda_small % 2 != 0)
        lambda_small++;

    p.lambda = std::max(size_t{2}, large() ? lambda_large : lambda_small);
    p.mu = std::max(Float{1.0}, p.lambda * mu_factor);
    return large() ? p.settings.sigma0
                   : p.settings.sigma0 * std::pow(10, -2 * dist(*p.generator));
}
} // namespace restart
