#include "parameters.hpp"

namespace parameters
{
Parameters::Parameters(const Settings &settings) :
    successful_adaptation(true),
    lambda(settings.lambda0),
    mu(settings.mu0),
    settings(settings),
    stats{},
    generator(settings.seed ? std::make_shared<rng::Generator>(*settings.seed) : rng::fork()),
    sampler(sampling::get(settings.dim, settings.modules, settings.lambda0, generator)),
    weights(settings, sampler->expected_length()),
    pop(settings.dim, settings.lambda0),
    old_pop(settings.dim, settings.lambda0),
    criteria(restart::Criteria::get(settings.modules, settings.lambda0)),
    adaptation(matrix_adaptation::get(settings.modules,
                                      settings.dim,
                                      settings.x0.value_or(Vector::Zero(settings.dim)),
                                      sampler->expected_length(),
                                      settings.sigma0)),
    mutation(mutation::get(
        settings.modules, static_cast<Float>(settings.dim), settings.sigma0, generator)),
    selection(std::make_shared<selection::Strategy>(settings.modules)),
    restart_strategy(restart::strategy::get(settings.modules,
                                            static_cast<Float>(settings.mu0),
                                            static_cast<Float>(settings.lambda0),
                                            settings.budget)),
    bounds(bounds::get(settings.modules.bound_correction, settings.dim, generator)),
    repelling(repelling::get(settings.modules)),
    center_placement(center::get(settings.modules.center_placement)),
    coordinate_mapping(mapping::get(settings.integer_variables, settings.dim, weights.mueff))
{
    criteria.reset(*this);
}

Parameters::Parameters(const size_t dim) :
    Parameters(Settings(dim, {}))
{
}

void Parameters::resize_population(const size_t new_mu, const size_t new_lambda)
{
    // TODO: Check validity
    mu = new_mu;
    lambda = new_lambda;

    sampler->reset(settings.modules, lambda);
    weights.init(mu, lambda, settings);
    pop = Population(settings.dim, lambda);

    // old_pop = Population(settings.dim, lambda);
}

void Parameters::perform_restart(FunctionType &objective, const std::optional<Float> &sigma)
{
    begin_restart(sigma);
    complete_restart(objective);
}

void Parameters::complete_restart(FunctionType &objective)
{
    {
        const Vector x = restart_center();
        tell_restart_center(objective(x), x);
    }
    while (repelling->has_query())
    {
        const Vector x = repelling->query_point();
        repelling->answer(objective(x), x, *this);
    }
    finish_restart();
}

void Parameters::begin_restart(const std::optional<Float> &sigma)
{
    if (settings.verbose)
    {
        std::cout << "performing restart. t(" << stats.t << ", " << stats.evaluations
                  << ") reason: " << criteria.reason() << std::endl;
        std::cout << "setting parameters (mu, lambda, sigma): (" << mu << ", " << lambda << ", "
                  << sigma.value_or(settings.sigma0) << ")\n";
    }
    restart_sigma = sigma;
    stats.solutions.push_back(stats.current_best);
    pending_restart_center = coordinate_mapping->transform(adaptation->m);
}

void Parameters::tell_restart_center(const Float y, Eigen::Ref<const Vector> x)
{
    stats.evaluations++;
    stats.centers.emplace_back(x, y, stats.t - 1, stats.evaluations);
    stats.update_best(stats.centers.back().x, stats.centers.back().y);
    stats.has_improved = false;
    repelling->begin_update(*this);
}

void Parameters::finish_restart()
{
    const std::optional<Float> sigma = restart_sigma;
    restart_sigma.reset();

    sampler->reset(settings.modules, lambda);
    weights.init(mu, lambda, settings);

    pop = Population(settings.dim, lambda);
    old_pop = Population(settings.dim, lambda);

    mutation = mutation::get(settings.modules,
                             static_cast<Float>(settings.dim),
                             sigma.value_or(settings.sigma0),
                             generator);
    adaptation->restart(settings, sigma.value_or(settings.sigma0));
    (*center_placement)(*this);
    criteria.reset(*this);
    stats.current_best = {};
}

void Parameters::adapt()
{
    adaptation->adapt_evolution_paths(pop, weights, stats, settings, mu, lambda);
    mutation->adapt(weights, adaptation, pop, old_pop, stats, lambda);

    if (constants::clip_sigma)
        mutation->sigma = std::min(std::max(mutation->sigma, restart::MinSigma::tolerance),
                                   restart::MaxSigma::tolerance);

    successful_adaptation =
        adaptation->adapt_matrix(weights, settings.modules, pop, mu, settings, stats);
    criteria.update(*this);
    stats.t++;
}

bool Parameters::begin_generation()
{
    // The previous generation becomes old_pop: a swap of the buffers (no copy). The next mutate
    // writes every active column of pop.
    std::swap(old_pop, pop);
    if (!criteria.any())
        return false;
    begin_restart(restart_strategy->update(*this));
    return true;
}

void Parameters::start(FunctionType &objective)
{
    if (begin_generation())
        complete_restart(objective);
}
} // namespace parameters

std::ostream &operator<<(std::ostream &os, const parameters::Stats &s)
{
    return os << "Stats" << " t=" << s.t << " e=" << s.evaluations << " best=" << s.global_best
              << " improved=" << std::boolalpha << s.has_improved;
}
