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
    for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(p->mu); ++k)
        a.m.noalias() += (1.0 * p->weights.positive(k)) * (p->pop.X.col(k) - a.m_old);
}

void ModularCMAES::select() const { p->selection->select(*p); }

void ModularCMAES::adapt() const { p->adapt(); }

void ModularCMAES::mutate(FunctionType &objective)
{
    if (phase == Phase::IDLE)
        start_generation();
    while (phase != Phase::DONE)
    {
        evaluate(objective);
        consume(values.head(n_pending), nullptr);
    }
    phase = Phase::IDLE;
}

bool ModularCMAES::step(FunctionType &objective)
{
    do
    {
        ask();
        evaluate(objective);
    } while (!tell(values.head(n_pending)));
    return !break_conditions();
}

void ModularCMAES::operator()(FunctionType &objective)
{
    while (step(objective))
        ;
}

bool ModularCMAES::break_conditions() const
{
    const auto target_reached =
        p->settings.target and p->settings.target.value() >= p->stats.global_best.y;
    const auto budget_used_up = (p->stats.evaluations + p->lambda) >= p->settings.budget;
    const auto exceed_gens =
        p->settings.max_generations and p->stats.t >= p->settings.max_generations;
    const auto restart_strategy_criteria =
        p->settings.modules.restart_strategy == parameters::RestartStrategyType::STOP and
        p->criteria.any();
    return exceed_gens or target_reached or budget_used_up or restart_strategy_criteria;
}

// -----------------------------------------------------------------------------
// Ask-tell
// -----------------------------------------------------------------------------

Eigen::Ref<const Matrix> ModularCMAES::ask()
{
    if (phase == Phase::IDLE)
        start_generation();
    return batch.leftCols(n_pending);
}

bool ModularCMAES::tell(Eigen::Ref<const Vector> f) { return tell(f, nullptr); }

bool ModularCMAES::tell(Eigen::Ref<const Vector> f, Eigen::Ref<const Matrix> X)
{
    return tell(f, &X);
}

bool ModularCMAES::tell(Eigen::Ref<const Vector> f, const Eigen::Ref<const Matrix> *X)
{
    if (phase == Phase::IDLE || phase == Phase::DONE)
        throw std::logic_error("tell() without pending points: call ask() first");
    if (f.size() != n_pending)
        throw std::invalid_argument("tell() expects " + std::to_string(n_pending) +
                                    " objective values (the points of ask()), got " +
                                    std::to_string(f.size()));
    if (X != nullptr && (X->rows() != batch.rows() || X->cols() != n_pending))
        throw std::invalid_argument("tell() expects the points as a " +
                                    std::to_string(batch.rows()) + " x " +
                                    std::to_string(n_pending) + " matrix, got " +
                                    std::to_string(X->rows()) + " x " + std::to_string(X->cols()));

    consume(f, X);
    if (phase != Phase::DONE)
        return false;

    select();
    recombine();
    adapt();
    phase = Phase::IDLE;
    return true;
}

void ModularCMAES::reserve_batch(const Eigen::Index k)
{
    const auto d = static_cast<Eigen::Index>(p->settings.dim);
    if (batch.rows() != d || batch.cols() < k)
        batch.resize(d, std::max(k, static_cast<Eigen::Index>(p->lambda) + 2));
    n_pending = k;
}

void ModularCMAES::start_generation()
{
    if (p->begin_generation())
    {
        reserve_batch(1);
        batch.col(0) = p->restart_center();
        phase = Phase::RESTART_CENTER;
        return;
    }
    start_population();
}

void ModularCMAES::start_population()
{
    p->mutation->begin_population(p->lambda, *p);
    next = 0;
    sample_batch();
}

void ModularCMAES::sample_batch()
{
    // Individuals next..last: sequential selection cannot stop the population before last.
    // Sampled in a loop that evaluates each individual, individual i would see next + i
    // evaluations (threshold convergence).
    const auto n = static_cast<Eigen::Index>(p->pop.n);
    const auto first_stop = p->mutation->first_stop(static_cast<size_t>(next), *p);
    const Eigen::Index last = first_stop >= static_cast<size_t>(n)
                                  ? n - 1
                                  : static_cast<Eigen::Index>(first_stop);
    const size_t evaluations = p->stats.evaluations;

    n_individuals = last - next + 1;
    p->mutation->sample_batch(next, n_individuals, evaluations, *p);

    // The extra points of the step-size adaptation go with the last individuals: they do not
    // depend on the population
    n_extra = last == n - 1 ? static_cast<Eigen::Index>(p->mutation->n_extra_points()) : 0;
    reserve_batch(n_individuals + n_extra);
    batch.leftCols(n_individuals) = p->pop.X_transformed.middleCols(next, n_individuals);
    if (n_extra != 0)
        p->mutation->extra_points(batch.middleCols(n_individuals, n_extra), *p);
    phase = Phase::POPULATION;
}

void ModularCMAES::next_restart_point()
{
    if (p->repelling->has_query())
    {
        reserve_batch(1);
        batch.col(0) = p->repelling->query_point();
        phase = Phase::RESTART_REPELLING;
        return;
    }
    p->finish_restart();
    start_population();
}

void ModularCMAES::inject(const Eigen::Index i, Eigen::Ref<const Vector> x) const
{
    auto &pop = p->pop;
    pop.X.col(i) = x;
    pop.X_transformed.col(i) = x;
    p->adaptation->invert_x_into(pop.X.col(i), pop.S.col(i), pop.Y.col(i));
    p->adaptation->invert_y_into(pop.Y.col(i), pop.Z.col(i));
}

void ModularCMAES::consume(Eigen::Ref<const Vector> f, const Eigen::Ref<const Matrix> *X)
{
    // The evaluated points: the given ones, or the asked ones
    const Eigen::Ref<const Matrix> points =
        X != nullptr ? *X : Eigen::Ref<const Matrix>(batch.leftCols(n_pending));

    switch (phase)
    {
    case Phase::RESTART_CENTER:
        p->tell_restart_center(f(0), points.col(0));
        next_restart_point();
        return;

    case Phase::RESTART_REPELLING:
        p->repelling->answer(f(0), points.col(0), *p);
        next_restart_point();
        return;

    case Phase::POPULATION:
    {
        bool complete = false;
        for (Eigen::Index j = 0; j < n_individuals && !complete; ++j)
        {
            if (X != nullptr && points.col(j) != batch.col(j))
                inject(next + j, points.col(j));
            complete = p->mutation->evaluated(next + j, f(j), *p);
        }
        if (!complete)
        {
            next += n_individuals;
            sample_batch();
            return;
        }

        p->mutation->end_population(*p);
        if (n_extra != 0)
        {
            p->mutation->tell_extra_points(
                f.segment(n_individuals, n_extra), points.middleCols(n_individuals, n_extra), *p);
            phase = Phase::DONE;
        }
        else if (const auto k = static_cast<Eigen::Index>(p->mutation->n_extra_points()); k != 0)
        {
            // Sequential selection stopped the population: the extra points on their own
            reserve_batch(k);
            p->mutation->extra_points(batch.leftCols(k), *p);
            n_individuals = 0;
            n_extra = k;
            phase = Phase::EXTRA;
        }
        else
            phase = Phase::DONE;
        return;
    }

    case Phase::EXTRA:
        p->mutation->tell_extra_points(f, points, *p);
        phase = Phase::DONE;
        return;

    default:
        throw std::logic_error("ModularCMAES: no pending points");
    }
}

void ModularCMAES::evaluate(FunctionType &objective)
{
    // Resized here, never in consume(): tell() reads the values from this buffer
    if (values.size() < batch.cols())
        values.resize(batch.cols());
    for (Eigen::Index j = 0; j < n_pending; ++j)
        values(j) = objective(batch.col(j));
}
