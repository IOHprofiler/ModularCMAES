#include "parameters.hpp"

#include <stdexcept>

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
    adaptation(matrix_adaptation::get(settings.modules,
                                      settings.dim,
                                      settings.x0.value_or(settings.center),
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
    coordinate_mapping(mapping::get(settings.integer_variables, settings.dim, weights.mueff)),
    queue(settings.dim)
{
    criteria.reset(*this);

    // The built-in subscribers of the pipeline. They reach the modules through p, so a module that
    // a restart replaces (the step-size adaptation) is the current one.
    using evaluation::GenerationStage;
    pipeline.on(GenerationStage::RUN_ENDING, "restart-center", [](Parameters &p) {
        // The final mean of the run that ends; its consumer starts the update of the repelling
        // archive, which queues the hill-valley points
        p.queue.push(
            p.restart_center(),
            [](Parameters &q, const evaluation::Evaluations &e) {
                q.tell_restart_center(e.f(0), e.X.col(0));
            },
            "restart-center");
    });
    pipeline.on(
        GenerationStage::ALONGSIDE_POPULATION,
        "step-size-extra-points",
        [](Parameters &p) {
            const auto k = static_cast<Eigen::Index>(p.mutation->n_extra_points());
            if (k == 0)
                return;
            auto X = p.queue.push(
                k,
                [](Parameters &q, const evaluation::Evaluations &e) {
                    q.mutation->tell_extra_points(e.f, e.X, q);
                },
                "step-size-extra-points");
            p.mutation->extra_points(X, p);
        },
        [](const Parameters &p) { return p.mutation->n_extra_points(); });
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
    pipeline.fire(evaluation::GenerationStage::RUN_ENDING, *this);
    queue.drain(*this, objective);
    finish_restart();
    pipeline.fire(evaluation::GenerationStage::RUN_STARTING, *this);
    queue.drain(*this, objective);
}

size_t Parameters::budget_left() const
{
    const size_t used = stats.evaluations + static_cast<size_t>(queue.size());
    return settings.budget > used ? settings.budget - used : 0;
}

void Parameters::push_variable(Eigen::Ref<const Matrix> X,
                               evaluation::Consumer consume,
                               const char *source)
{
    if (static_cast<size_t>(X.cols()) > budget_left())
        throw std::logic_error("a request of " + std::to_string(X.cols()) + " points (" +
                               std::string(source) + ") exceeds the budget: " +
                               std::to_string(budget_left()) + " evaluations are left");
    queue.push(X, std::move(consume), source);
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
    stats.evaluations++;
    stats.centers.emplace_back(x, y, stats.t - 1, stats.evaluations);
    stats.update_best(stats.centers.back().x, stats.centers.back().y);
    stats.has_improved = false;
    repelling->begin_update(*this);
}

void Parameters::finish_restart()
{
    // The modules are replaced below: no request may still wait for its values (its consumer
    // could refer to a replaced module)
    if (!queue.empty())
        throw std::logic_error("finish_restart: evaluations are still pending");
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
    adaptation->restart(settings, sigma.value_or(settings.sigma0));
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
    const size_t extra = pipeline.fixed_points(*this);
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

bool Parameters::restart_fits(const size_t n) const
{
    return stats.evaluations + n + min_lambda_after_restart + pipeline.fixed_points(*this) <=
           settings.budget;
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
