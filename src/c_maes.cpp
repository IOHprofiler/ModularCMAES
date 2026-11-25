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

void ModularCMAES::select() const
{
    require_no_generation("select()");
    p->selection->select(*p);
}

void ModularCMAES::adapt() const
{
    require_no_generation("adapt()");
    p->adapt();
}

void ModularCMAES::require_no_generation(const char *step) const
{
    if (next != evaluation::Step::BEGIN || !p->queue.empty())
        throw std::logic_error(std::string(step) +
                               " while a generation started by ask() is in progress: tell() the "
                               "values of its points, or evaluate them with mutate(), first");
}

void ModularCMAES::mutate(FunctionType &objective)
{
    // Up to selection: a new generation, or the one that ask() started. The rest is the user's:
    // select(), recombine() and adapt() act on the state, and the next mutate() or ask() starts a
    // new generation, as in 1.x
    while (!p->queue.empty() || next != evaluation::Step::SELECT)
    {
        if (p->queue.empty())
            advance(true);
        else
        {
            evaluate(objective);
            p->queue.answer(*p, values.head(p->queue.size()));
        }
    }
    next = evaluation::Step::BEGIN;
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
    // Stop when the next generation (lambda points, plus the reserved points: TPA's) would exceed
    // the budget; a generation that ends exactly at the budget is still run. A generation that
    // starts with a restart evaluates the center first (and hill-valley points of repelling, while
    // the budget allows) and takes at most the rest of the budget (fit_lambda_to_budget).
    const auto restart_due =
        p->settings.modules.restart_strategy != parameters::RestartStrategyType::STOP &&
        p->criteria.any();
    const auto budget_used_up =
        restart_due ? !p->restart_fits(1)
                    : (p->stats.evaluations + p->lambda + p->reserved_points()) >
                          p->settings.budget;
    const auto exceed_gens =
        p->settings.max_generations and p->stats.t >= p->settings.max_generations;
    const auto restart_strategy_criteria =
        p->settings.modules.restart_strategy == parameters::RestartStrategyType::STOP and
        p->criteria.any();
    return exceed_gens or target_reached or budget_used_up or restart_strategy_criteria;
}

// -----------------------------------------------------------------------------
// Ask-tell: the steps of a generation over the evaluation queue (#84)
// -----------------------------------------------------------------------------

Eigen::Ref<const Matrix> ModularCMAES::ask()
{
    // Starts a generation when none is in progress
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

bool ModularCMAES::advance(const bool stop_before_selection)
{
    // The steps of a generation, in order. A step that queues points returns control: ask() hands
    // them out, and the next step runs once tell() has given all their values.
    while (p->queue.empty())
    {
        switch (next)
        {
        case evaluation::Step::BEGIN:
            // A restart first when the criteria fired: the final mean of the run that ends is
            // evaluated, and its value starts the hill-valley tests of the repelling archive
            if (p->begin_generation())
            {
                p->push_restart_center();
                next = evaluation::Step::FINISH_RESTART;
            }
            else
                next = evaluation::Step::SAMPLE;
            break;

        case evaluation::Step::FINISH_RESTART:
            // Every point of the restart has its value: the modules are reset for the new run
            p->finish_restart();
            next = evaluation::Step::SAMPLE;
            break;

        case evaluation::Step::SAMPLE:
            // The population's first batch (each batch queues the next), with the extra points
            // of the step-size adaptation (TPA's), which do not depend on the population's values
            p->mutation->begin_population(p->lambda, *p);
            p->mutation->push_extra_points(*p);
            next = evaluation::Step::SELECT;
            break;

        case evaluation::Step::SELECT:
            // Every point of the population has its value
            if (stop_before_selection)
                return false;
            p->selection->select(*p);
            next = evaluation::Step::UPDATE;
            break;

        case evaluation::Step::UPDATE:
            recombine();
            p->adapt();
            next = evaluation::Step::BEGIN;
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
