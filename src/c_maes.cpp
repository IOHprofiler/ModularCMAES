#include "c_maes.hpp"

#include <stdexcept>

void ModularCMAES::recombine() const
{
    auto &a = *p->adaptation;
    a.m_old = a.m;
    // m = m_old + (X_mu - m_old) w. Eigen evaluated this product (its left-hand side is an
    // expression, not a matrix) as dst = m_old; dst += w_k (x_k - m_old) for k = 0..mu-1; the same
    // loop, without the temporary, gives the same bits.
    a.m = a.m_old;
    if (p->weights.utility)
    {
        // UTILITY (NES/xNES): every evaluated point, with the utilities (which sum to zero)
        Vector u;
        p->weights.mean_weights_into(p->pop.n, u);
        for (Eigen::Index k = 0; k < u.size(); ++k)
            a.m.noalias() += u(k) * (p->pop.X.col(k) - a.m_old);
        return;
    }
    for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(p->mu); ++k)
        a.m.noalias() += (1.0 * p->weights.positive(k)) * (p->pop.X.col(k) - a.m_old);
}

void ModularCMAES::select()
{
    p->selection->select(*p);
    enter(evaluation::GenerationStage::SELECTED);
    require_no_pending("select()");
}

void ModularCMAES::adapt()
{
    p->adapt();
    enter(evaluation::GenerationStage::UPDATED);
    require_no_pending("adapt()");
    stage.reset();
}

void ModularCMAES::require_no_pending(const char *step) const
{
    if (!p->queue.empty())
        throw std::logic_error(std::string(step) + " left points to evaluate (" +
                               p->queue.sources().front() +
                               "): use step(), run() or ask-tell with this configuration");
}

void ModularCMAES::mutate(FunctionType &objective)
{
    // Up to selection: the user calls select(), recombine() and adapt()
    stop_before_selection = true;
    while (true)
    {
        if (p->queue.empty() && (advance() || p->queue.empty()))
            break;
        evaluate(objective);
        p->queue.answer(*p, values.head(p->queue.size()));
    }
    stop_before_selection = false;
}

bool ModularCMAES::step(FunctionType &objective)
{
    ask();
    do
        evaluate(objective);
    while (!tell(values.head(p->queue.size())));
    return !break_conditions();
}

void ModularCMAES::operator()(FunctionType &objective)
{
    while (step(objective))
        ;
}

Vector ModularCMAES::xfavorite() const
{
    Vector x = p->coordinate_mapping->transform(p->adaptation->m);
    if (p->settings.has_bounds &&
        p->settings.modules.bound_correction != parameters::CorrectionMethod::NONE)
        x = x.cwiseMax(p->settings.lb).cwiseMin(p->settings.ub);
    return x;
}

bool ModularCMAES::break_conditions() const
{
    const auto target_reached =
        p->settings.target and p->settings.target.value() >= p->stats.global_best.y;
    // Stop when the next generation (lambda points, plus the fixed points of the pipeline: TPA's)
    // would exceed the budget; a generation that ends exactly at the budget is still run. A
    // generation that starts with a restart evaluates the center first (and hill-valley points of
    // repelling, while the budget allows) and takes at most the rest of the budget
    // (fit_lambda_to_budget).
    const auto restart_due =
        p->settings.modules.restart_strategy != parameters::RestartStrategyType::STOP &&
        p->criteria.any();
    const auto budget_used_up = restart_due ? !p->restart_fits(1)
                                            : (p->stats.evaluations + p->lambda +
                                               p->pipeline.fixed_points(*p)) > p->settings.budget;
    const auto exceed_gens =
        p->settings.max_generations and p->stats.t >= p->settings.max_generations;
    const auto restart_strategy_criteria =
        p->settings.modules.restart_strategy == parameters::RestartStrategyType::STOP and
        p->criteria.any();
    return exceed_gens or target_reached or budget_used_up or restart_strategy_criteria;
}

// -----------------------------------------------------------------------------
// Ask-tell: the pipeline of a generation over the evaluation queue (#84)
// -----------------------------------------------------------------------------

Eigen::Ref<const Matrix> ModularCMAES::ask()
{
    // Starts a generation (or continues one that mutate() stopped before selection)
    while (p->queue.empty())
        advance();
    return p->queue.batch();
}

bool ModularCMAES::tell(Eigen::Ref<const Vector> f) { return tell(f, nullptr); }

bool ModularCMAES::tell(Eigen::Ref<const Vector> f, Eigen::Ref<const Matrix> X)
{
    return tell(f, &X);
}

bool ModularCMAES::tell(Eigen::Ref<const Vector> f, const Eigen::Ref<const Matrix> *X)
{
    if (p->queue.empty())
        throw std::logic_error("tell() without pending points: call ask() first");
    p->queue.answer(*p, f, X);
    return advance();
}

void ModularCMAES::enter(const evaluation::GenerationStage next)
{
    stage = next;
    p->pipeline.fire(next, *p);
}

bool ModularCMAES::advance()
{
    using evaluation::GenerationStage;
    while (p->queue.empty())
    {
        if (!stage)
        {
            // A new generation: a restart first, when the criteria fired
            enter(p->begin_generation() ? GenerationStage::RUN_ENDING
                                        : GenerationStage::GENERATION_STARTING);
            continue;
        }
        switch (*stage)
        {
        case GenerationStage::RUN_ENDING:
            p->finish_restart();
            enter(GenerationStage::RUN_STARTING);
            break;
        case GenerationStage::RUN_STARTING:
            enter(GenerationStage::GENERATION_STARTING);
            break;
        case GenerationStage::GENERATION_STARTING:
            // Queues the population's first batch; ALONGSIDE_POPULATION fires from within
            stage = GenerationStage::POPULATION;
            p->mutation->begin_population(p->lambda, *p);
            break;
        case GenerationStage::POPULATION:
        case GenerationStage::ALONGSIDE_POPULATION:
            enter(GenerationStage::POPULATION_EVALUATED);
            break;
        case GenerationStage::POPULATION_EVALUATED:
            if (stop_before_selection)
                return false;
            p->selection->select(*p);
            enter(GenerationStage::SELECTED);
            break;
        case GenerationStage::SELECTED:
            recombine();
            p->adapt();
            enter(GenerationStage::UPDATED);
            break;
        case GenerationStage::UPDATED:
            stage.reset();
            return true;
        }
    }
    return false;
}

void ModularCMAES::evaluate(FunctionType &objective)
{
    const auto batch = p->queue.batch();
    if (values.size() < batch.cols())
        values.resize(batch.cols());
    for (Eigen::Index j = 0; j < batch.cols(); ++j)
        values(j) = objective(batch.col(j));
}
