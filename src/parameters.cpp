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
    sampler(sampling::get(
        settings.dim, settings.modules, settings.lambda0, generator, settings.fast_sampling)),
    weights(settings, sampler->expected_length()),
    pop(settings.dim, settings.lambda0),
    old_pop(settings.dim, settings.lambda0),
    criteria(restart::Criteria::get(settings.modules, settings.lambda0)),
    adaptation(matrix_adaptation::get(
        settings.modules, settings.dim, settings.x0.value_or(settings.center))),
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
    coordinate_mapping(mapping::get(settings.integer_variables, settings.dim))
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

size_t Parameters::reserved_points() const
{
    // TPA's two points, with the population's first batch
    return mutation->n_extra_points();
}

void Parameters::inject(const Eigen::Index i, Eigen::Ref<const Vector> x)
{
    pop.X.col(i) = x;
    pop.X_transformed.col(i) = x;
    adaptation->invert_x_into(pop.X.col(i), pop.S.col(i), pop.Y.col(i));
    adaptation->invert_y_into(pop.Y.col(i), pop.Z.col(i));
    // A point far outside the distribution enters the update clipped (as pycma): X, from which the
    // mean is recombined, moves with y; X_transformed keeps the evaluated point for the statistics
    if (matrix_adaptation::clip_injected_step(pop.Y.col(i), pop.Z.col(i)))
        pop.X.col(i) = adaptation->m + pop.S.col(i).cwiseProduct(pop.Y.col(i));
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
    // The center belongs to the last generation of the run that ends (0 before any)
    stats.centers.emplace_back(x, y, stats.t > 0 ? stats.t - 1 : 0, stats.evaluations);
    stats.update_best(stats.centers.back().x, stats.centers.back().y);
    stats.has_improved = false;
    repelling->begin_update(*this);
}

void Parameters::finish_restart()
{
    const std::optional<Float> sigma = restart_sigma;
    restart_sigma.reset();
    fit_lambda_to_budget();

    sampler->reset(settings.modules, lambda);
    weights.init(mu, lambda, settings);

    pop = Population(settings.dim, lambda);
    old_pop = Population(settings.dim, lambda);

    mutation = mutation::get(settings.modules,
                             static_cast<Float>(settings.dim),
                             sigma.value_or(settings.sigma0),
                             generator);
    adaptation->restart(settings);
    (*center_placement)(*this);
    criteria.reset(*this);
    stats.current_best = {};
    stats.t_run = 0;
}

void Parameters::fit_lambda_to_budget()
{
    // The first generation after a restart takes at most the remaining budget (the center and the
    // repelling queries are evaluated by now; TPA adds its two points): a restart near the end of
    // the budget, where IPOP and BIPOP may also have grown lambda, would otherwise exceed it.
    // At least min_lambda_after_restart points; pairwise selection needs an even lambda.
    const size_t extra = reserved_points();
    const size_t used = stats.evaluations + extra;
    const size_t remaining = settings.budget > used ? settings.budget - used : 0;
    if (lambda <= remaining)
        return;
    size_t capped = std::max(min_lambda_after_restart, remaining);
    if (settings.modules.mirrored == parameters::Mirror::PAIRWISE)
        capped -= capped % 2;
    mu = std::clamp<size_t>(static_cast<size_t>(std::round(static_cast<Float>(mu) *
                                                           static_cast<Float>(capped) /
                                                           static_cast<Float>(lambda))),
                            1, capped);
    lambda = capped;
}

bool Parameters::restart_fits() const
{
    return stats.evaluations + 1 + min_lambda_after_restart + reserved_points() <= settings.budget;
}

void Parameters::recombine()
{
    auto &a = *adaptation;
    a.m_old = a.m;
    // m = m_old + (X_mu - m_old) w. Eigen evaluated this product (its left-hand side is an
    // expression, not a matrix) as dst = m_old; dst += w_k (x_k - m_old) for k = 0..mu-1; the same
    // loop, without the temporary, gives the same bits.
    a.m = a.m_old;
    if (weights.utility)
    {
        // UTILITY (NES/xNES): every evaluated point, with the utilities (which sum to zero)
        Vector u;
        weights.mean_weights_into(pop.n, u);
        for (Eigen::Index k = 0; k < u.size(); ++k)
            a.m.noalias() += u(k) * (pop.X.col(k) - a.m_old);
        return;
    }
    for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(mu); ++k)
        a.m.noalias() += (1.0 * weights.positive(k)) * (pop.X.col(k) - a.m_old);
}

void Parameters::adapt()
{
    adaptation->adapt_evolution_paths(pop, weights, stats, settings, mu, lambda);
    mutation->adapt(weights, adaptation, pop, old_pop, stats, lambda);

    if (constants::clip_sigma)
        mutation->sigma = std::min(std::max(mutation->sigma, restart::MinSigma::tolerance),
                                   restart::MaxSigma::tolerance);

    successful_adaptation =
        adaptation->adapt_matrix(weights, pop, mu, settings, stats);
    criteria.update(*this);
    stats.t++;
    stats.t_run++;
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

} // namespace parameters

std::ostream &operator<<(std::ostream &os, const parameters::Stats &s)
{
    return os << "Stats" << " t=" << s.t << " e=" << s.evaluations << " best=" << s.global_best
              << " improved=" << std::boolalpha << s.has_improved;
}
