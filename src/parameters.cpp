#include "parameters.hpp"

namespace parameters
{
namespace
{
//! The settings, after every check (Settings::require_valid): the setters check only their own
//! field, so the checks between fields (the box, #133; the population and the budget, #136) run
//! again here, before anything is built from them
const Settings &valid(const Settings &settings)
{
    settings.require_valid();
    return settings;
}
} // namespace

Parameters::Parameters(const Settings &settings) :
    successful_adaptation(true),
    // The first member built from the settings: they are checked here
    lambda(valid(settings).lambda0),
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
    restart_strategy(restart::strategy::get(settings.modules, settings.lambda0)),
    bounds(bounds::get(settings.modules.bound_correction, settings.dim, generator)),
    repelling(repelling::get(settings.modules)),
    center_placement(center::get(settings.modules.center_placement)),
    coordinate_mapping(mapping::get(settings.integer_variables, settings.dim))
{
    // Warm starting (#89): the matrix starts from Settings::C0, before the criteria read it. C0 is
    // checked again: C++ code can assign the field directly, and the modules can change after
    // set_C0 (to NONE)
    if (this->settings.C0)
    {
        this->settings.set_C0(this->settings.C0);
        adaptation->start_from(*this->settings.C0, this->settings);
    }
    criteria.reset(*this);
}

Parameters::Parameters(const size_t dim) :
    Parameters(Settings(dim, {}))
{
}

void Parameters::resize_population(const size_t new_mu, const size_t new_lambda)
{
    // The populations keep their contents: the next generation makes pop old_pop (elitism, MSR,
    // PSR and sequential selection compare with it) and reserves the columns it needs (#117)
    mu = new_mu;
    lambda = new_lambda;

    sampler->reset(settings.modules, lambda);
    weights.init(mu, lambda, settings);
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

void Parameters::begin_restart(const Float sigma)
{
    if (settings.verbose)
    {
        std::cout << "performing restart. t(" << stats.t << ", " << stats.evaluations
                  << ") reason: " << criteria.reason() << std::endl;
        std::cout << "setting parameters (mu, lambda, sigma): (" << mu << ", " << lambda << ", "
                  << sigma << ")\n";
    }
    restart_sigma = sigma;
    end_run();
}

Vector Parameters::center_of_mass() const
{
    // Clipped per coordinate and per side (an infinite bound clips nothing, #79 F20), then
    // rounded: the integer variables stay in the box (#79 F21). With integral bounds the order
    // does not matter.
    Vector x = adaptation->m;
    if (settings.modules.bound_correction != CorrectionMethod::NONE)
        x = x.cwiseMax(settings.lb).cwiseMin(settings.ub);
    return coordinate_mapping->transform(x, settings.lb, settings.ub);
}

void Parameters::end_run() { stats.solutions.push_back(stats.current_best); }

void Parameters::tell_center_of_mass(const Float y,
                                     Eigen::Ref<const Vector> x,
                                     std::vector<std::string> reasons,
                                     const bool restarting)
{
    // The record of the run that ends: the evaluated center (of the last generation of the run, 0
    // before any) with the distribution the run ended with. Nothing has changed sigma and the
    // matrix since the run ended (a restart resets them in finish_restart), and the restart
    // criteria read the same metric (the same factor_version, #123): for the adaptations that
    // compute it, at most one decomposition per run.
    stats.centers.emplace_back(x,
                               y,
                               stats.t > 0 ? stats.t - 1 : 0,
                               stats.evaluations,
                               mutation->sigma,
                               adaptation->metric().copy(),
                               std::move(reasons));
    stats.update_best(stats.centers.back().x, stats.centers.back().y);
    stats.has_improved = false;
    if (restarting)
        repelling->begin_update(*this);
}

void Parameters::finish_restart()
{
    fit_lambda_to_budget();

    sampler->reset(settings.modules, lambda);
    weights.init(mu, lambda, settings);

    pop = Population(settings.dim, lambda);
    old_pop = Population(settings.dim, lambda);

    mutation =
        mutation::get(settings.modules, static_cast<Float>(settings.dim), restart_sigma, generator);
    adaptation->restart(settings);
    // Every run starts from C0, as from sigma0 and (center_placement X0) x0: the initial
    // distribution of the settings (#89; open decision: or C = I after a restart)
    if (settings.C0)
        adaptation->start_from(*settings.C0, settings);
    (*center_placement)(*this);
    stats.current_best = {};
    stats.t_run = 0;
    // Last: the on_reset of a restart criterion may be Python code that throws; the new run is
    // complete by then
    criteria.reset(*this);
}

void Parameters::fit_lambda_to_budget()
{
    // The first generation after a restart takes at most the remaining budget (the center and the
    // repelling queries are evaluated by now; TPA adds its two points; the new run keeps one
    // evaluation for its own center of mass, #111): a restart near the end of the budget, where
    // IPOP and BIPOP may also have grown lambda, would otherwise exceed it.
    // At least min_lambda_after_restart points; pairwise selection needs an even lambda.
    const size_t extra = reserved_points() + 1;
    const size_t used = stats.evaluations + extra;
    const size_t remaining = settings.budget > used ? settings.budget - used : 0;
    if (lambda <= remaining)
        return;
    // Never more than lambda: past the budget (ask-tell that ignores break_conditions), a (1+1)-ES
    // stays one (#114)
    size_t capped = std::min(lambda, std::max(min_lambda_after_restart, remaining));
    if (settings.modules.mirrored == parameters::Mirror::PAIRWISE)
        capped -= capped % 2;
    // mu of the capped lambda by the rule of the restart strategies (#150); it was mu scaled by
    // capped / lambda, rounded
    mu = settings.mu_for(capped);
    lambda = capped;
}

bool Parameters::restart_fits() const
{
    return stats.evaluations + 1 + min_lambda_after_restart + reserved_points() + 1 <=
           settings.budget;
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
    adaptation->adapt_evolution_paths(pop, weights, stats, settings, mu);
    mutation->adapt(weights, adaptation, pop, old_pop, stats);

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
