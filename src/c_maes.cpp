#include "c_maes.hpp"

#include <stdexcept>

void ModularCMAES::recombine() const
{
    if (steps_skipped)
        return;
    require_no_generation("recombine()");
    p->recombine();
}

void ModularCMAES::select() const
{
    if (steps_skipped)
        return;
    require_no_generation("select()");
    p->selection->select(*p);
}

void ModularCMAES::adapt() const
{
    if (steps_skipped)
        return;
    require_no_generation("adapt()");
    p->adapt();
}

void ModularCMAES::require_no_generation(const char *step) const
{
    if (next != Step::BEGIN || n_pending != 0)
        throw std::logic_error(std::string(step) +
                               " while the points of a generation wait for their values (after "
                               "ask(), or an interrupted mutate() or step()): tell() their "
                               "values, or evaluate them with mutate(), first");
}

void ModularCMAES::mutate(FunctionType &objective)
{
    // Up to selection: a new generation, or the one that ask() started. The rest is the user's:
    // select(), recombine() and adapt() act on the state, and the next mutate() or ask() starts a
    // new generation, as in 1.x
    steps_skipped = false;
    while (n_pending != 0 || next != Step::SELECT)
    {
        if (n_pending == 0)
        {
            // With stop_before_selection, advance() returns true only when the run has ended (its
            // final center of mass has its value): there is no generation for the steps after
            if (advance(true))
            {
                steps_skipped = true;
                return;
            }
        }
        else
        {
            evaluate(objective);
            answer(values.head(n_pending), nullptr);
        }
    }
    next = Step::BEGIN;
}

bool ModularCMAES::step(FunctionType &objective)
{
    ask();
    do
        evaluate(objective);
    while (!tell(values.head(n_pending)));
    return !break_conditions();
}

void ModularCMAES::operator()(FunctionType &objective)
{
    // The break conditions are checked before every generation, the first one too: a run never
    // exceeds the budget, and an optimizer that has stopped does not run again (#114)
    if (break_conditions())
        return;
    while (step(objective))
        ;
}

ModularCMAES::StopFlags ModularCMAES::stop_flags() const
{
    StopFlags flags;
    flags.target = p->settings.target and p->settings.target.value() >= p->stats.global_best.y;
    // Stop when the next generation (lambda points, plus the reserved points: TPA's) would leave no
    // evaluation for the final center of mass (#111); a generation that ends one evaluation before
    // the budget is still run. A generation that starts with a restart evaluates the center first
    // (and hill-valley points of repelling, while the budget allows) and takes at most the rest of
    // the budget, less the new run's center of mass (fit_lambda_to_budget).
    const auto restart_due =
        p->settings.modules.restart_strategy != parameters::RestartStrategyType::STOP &&
        p->criteria.any();
    flags.budget = restart_due ? !p->restart_fits()
                               : (p->stats.evaluations + p->lambda + p->reserved_points() + 1) >
                                     p->settings.budget;
    flags.generations = p->settings.max_generations and p->stats.t >= p->settings.max_generations;
    flags.criteria =
        p->settings.modules.restart_strategy == parameters::RestartStrategyType::STOP and
        p->criteria.any();
    return flags;
}

bool ModularCMAES::stop_conditions() const { return stop_flags().any(); }

bool ModularCMAES::break_conditions() const { return ended && stop_conditions(); }

std::vector<std::string> ModularCMAES::reasons(const StopFlags &flags) const
{
    std::vector<std::string> names;
    for (const auto &criterion : p->criteria.items)
        if (criterion->met)
            names.push_back(criterion->name);
    if (flags.budget)
        names.emplace_back("budget");
    if (flags.target)
        names.emplace_back("target");
    if (flags.generations)
        names.emplace_back("max_generations");
    return names;
}

// -----------------------------------------------------------------------------
// Ask-tell: the steps of a generation, and the points they queue (#84)
// -----------------------------------------------------------------------------

Eigen::Ref<const Matrix> ModularCMAES::ask()
{
    // Starts a generation when none is in progress
    while (n_pending == 0)
        advance();
    return batch.leftCols(n_pending);
}

bool ModularCMAES::tell(Eigen::Ref<const Vector> f) { return tell(f, nullptr); }

bool ModularCMAES::tell(Eigen::Ref<const Vector> f, Eigen::Ref<const Matrix> X)
{
    return tell(f, &X);
}

bool ModularCMAES::tell(Eigen::Ref<const Vector> f, const Eigen::Ref<const Matrix> *X)
{
    answer(f, X);
    return advance();
}

void ModularCMAES::answer(Eigen::Ref<const Vector> f, const Eigen::Ref<const Matrix> *X)
{
    const auto n = n_pending;
    if (n == 0)
        throw std::logic_error("tell() without pending points: call ask() first");
    if (f.size() != n)
        throw std::invalid_argument("tell() expects " + std::to_string(n) +
                                    " objective values (the points of ask()), got " +
                                    std::to_string(f.size()));
    if (X != nullptr && (X->rows() != batch.rows() || X->cols() != n))
        throw std::invalid_argument("tell() expects the points as a " +
                                    std::to_string(batch.rows()) + " x " +
                                    std::to_string(n) + " matrix, got " +
                                    std::to_string(X->rows()) + " x " + std::to_string(X->cols()));

    // Each request's values go to its module, in the order they were queued. Nothing is pending
    // afterwards, also when a module throws (a clear() keeps the capacity: no allocation)
    struct Clear
    {
        ModularCMAES &es;
        ~Clear()
        {
            es.n_pending = 0;
            es.requests.clear();
        }
    } clear{*this};
    const Eigen::Ref<const Matrix> evaluated =
        X != nullptr ? *X : Eigen::Ref<const Matrix>(batch.leftCols(n));
    for (const auto &r : requests)
    {
        // Every evaluation is counted here, before its module receives the values: no module
        // counts (Solution.e of a point is the count after its request)
        p->stats.evaluations += static_cast<size_t>(r.size);
        dispatch(r, f.segment(r.begin, r.size), evaluated.middleCols(r.begin, r.size), X != nullptr);
    }
}

void ModularCMAES::dispatch(const Request &request,
                            Eigen::Ref<const Vector> f,
                            Eigen::Ref<const Matrix> X,
                            const bool given)
{
    // The module in place now receives the values (one may have been replaced since the points
    // were queued)
    switch (request.source)
    {
    case Source::POPULATION:
        p->mutation->tell_batch(request.index, f, X, given, *p);
        return;
    case Source::EXTRA_POINTS:
        p->mutation->tell_extra_points(f, X, *p);
        return;
    case Source::CENTER_OF_MASS:
        // Queued by BEGIN, which set next: RESTART at a restart, END when the optimizer stops
        p->tell_center_of_mass(f(0), X.col(0), std::move(end_reasons), next == Step::RESTART);
        end_reasons.clear();
        return;
    case Source::HILL_VALLEY:
        p->repelling->answer(f(0), X.col(0), *p);
        return;
    }
}

bool ModularCMAES::advance(const bool stop_before_selection)
{
    // The steps of a generation, in order. This is the only place that queues points: a step that
    // queues some returns control (ask() hands them out), and runs again, or the next step, once
    // tell() has given their values.
    while (n_pending == 0)
    {
        switch (next)
        {
        case Step::BEGIN: {
            // Every run ends with the evaluation of its center of mass (#111): when the optimizer
            // stops, and when a restart is due (the restart criteria fired)
            const auto flags = stop_flags();
            if (!flags.any())
                ended = false; // the stop no longer holds (the budget was raised): continue
            if (flags.any() && !ended)
            {
                end_reasons = reasons(flags);
                p->end_run();
                queue(p->center_of_mass(), Source::CENTER_OF_MASS);
                next = Step::END;
            }
            else if (p->begin_generation())
            {
                end_reasons = reasons(StopFlags{});
                queue(p->center_of_mass(), Source::CENTER_OF_MASS);
                next = Step::RESTART;
            }
            else
                next = Step::SAMPLE;
            break;
        }

        case Step::END:
            // The final center of mass has its value: the optimizer has stopped. A later ask()
            // (ignoring break_conditions) continues with generations, as before the end
            ended = true;
            next = Step::BEGIN;
            return true;

        case Step::RESTART:
            // The center's value started the update of the repelling archive: its hill-valley
            // points, one at a time (each depends on the value of the one before); then the
            // modules are reset for the new run
            if (p->repelling->has_query())
                queue(p->repelling->query_point(), Source::HILL_VALLEY);
            else
            {
                // The next step is set first, as in UPDATE: when the reset throws (the on_reset of
                // a restart criterion written in Python), the next ask() does not reset again
                next = Step::SAMPLE;
                p->finish_restart();
            }
            break;

        case Step::SAMPLE:
            // A new population: its first batch, with the extra points of the step-size adaptation
            // (TPA's), which do not depend on the population's values
            steps_skipped = false;
            p->mutation->begin_population(p->lambda, *p);
            next_individual = 0;
            queue_population_batch();
            queue_extra_points();
            next = Step::POPULATION;
            break;

        case Step::POPULATION:
            // With sequential selection the population comes in batches, until every individual
            // has its value (a stop shortens pop.n to the evaluated individuals)
            if (next_individual >= static_cast<Eigen::Index>(p->pop.n))
                next = Step::SELECT;
            else
                queue_population_batch();
            break;

        case Step::SELECT:
            if (stop_before_selection)
                return false;
            p->selection->select(*p);
            next = Step::UPDATE;
            break;

        case Step::UPDATE:
            // The next step is set first: when the update throws (a restart criterion written in
            // Python), the next ask() starts a new generation instead of repeating the update
            next = Step::BEGIN;
            p->recombine();
            p->adapt();
            return true;
        }
    }
    return false;
}

void ModularCMAES::queue_population_batch()
{
    const Eigen::Index begin = next_individual;
    const Eigen::Index size = p->mutation->sample_batch_from(begin, *p);
    // Copied between the concrete matrices (not through queue(X)'s Ref): Eigen inlines this copy
    queue(size, Source::POPULATION, begin) =
        p->pop.X_transformed.middleCols(begin, size);
    next_individual += size;
}

void ModularCMAES::queue_extra_points()
{
    if (const auto k = static_cast<Eigen::Index>(p->mutation->n_extra_points()); k != 0)
        p->mutation->extra_points(queue(k, Source::EXTRA_POINTS), *p);
}

Eigen::Ref<Matrix> ModularCMAES::queue(const Eigen::Index k,
                                       const Source source,
                                       const Eigen::Index index)
{
    const Eigen::Index needed = n_pending + k;
    if (batch.rows() != static_cast<Eigen::Index>(p->settings.dim) || batch.cols() < needed)
    {
        // Grows rarely (a population of a new size), keeping the queued points; room for a whole
        // generation (the population and its reserved points)
        const auto generation = static_cast<Eigen::Index>(p->lambda + p->reserved_points());
        batch.conservativeResize(static_cast<Eigen::Index>(p->settings.dim),
                                 std::max(needed, generation));
    }
    requests.push_back(Request{source, n_pending, k, index});
    n_pending = needed;
    return batch.middleCols(needed - k, k);
}

void ModularCMAES::queue(Eigen::Ref<const Matrix> X, const Source source)
{
    queue(X.cols(), source) = X;
}

void ModularCMAES::evaluate(FunctionType &objective)
{
    if (values.size() < n_pending)
        values.resize(n_pending);
    for (Eigen::Index j = 0; j < n_pending; ++j)
        values(j) = objective(batch.col(j));
}
