#pragma once

#include <optional>

#include "parameters.hpp"

/**
 * The modular CMA-ES.
 *
 * Two ways to run it:
 *  - with an objective function: run() / operator() until a break condition holds, step() for one
 *    generation, or mutate(), select(), recombine(), adapt() for the steps of a generation;
 *  - ask-tell, for an objective that is evaluated outside: ask() returns the points to evaluate,
 *    tell() takes their objective values. step() and run() are loops over ask() and tell(), so both
 *    ways give the same results.
 *
 * A generation is a sequence of steps, in advance(): a restart when one is due, the population,
 * selection, then recombination and adaptation. advance() is the only place that queues points
 * (Parameters::queue); a step runs again, or the next one runs, once tell() has given their
 * values.
 * ask() returns everything that is queued: a whole population (with TPA's two points), or, with
 * sequential selection, the individuals up to the first one after which it can stop (with TPA's
 * points), then one (or a mirrored pair) at a time; a restart asks for the final mean, then for the
 * hill-valley points of the repelling archive, one at a time.
 */
struct ModularCMAES
{
    std::shared_ptr<parameters::Parameters> p;

    ModularCMAES(const std::shared_ptr<parameters::Parameters> p) :
        p(p)
    {
    }
    ModularCMAES(const size_t dim) :
        ModularCMAES(std::make_shared<parameters::Parameters>(dim))
    {
    }
    ModularCMAES(const parameters::Settings &settings) :
        ModularCMAES(std::make_shared<parameters::Parameters>(settings))
    {
    }

    /*
     * The steps of a generation (the 1.x API): mutate(), select(), recombine(), adapt(). As in
     * 1.x, mutate() always starts a new generation (or completes one that ask() started), and the
     * other steps act on the state as it is. select() and adapt() refuse to run while ask() has
     * points that wait for tell().
     */

    void recombine() const;

    void select() const;

    void adapt() const;

    //! Sample and evaluate a generation, up to selection (after a restart, when one is due)
    void mutate(FunctionType &objective);

    //! One generation; returns false when a break condition holds
    bool step(FunctionType &objective);

    //! Run until a break condition holds
    void operator()(FunctionType &objective);

    bool break_conditions() const;

    /**
     * The favorite solution: the mean of the search distribution as a point of the objective
     * (integer variables rounded; inside the bounds when a bound correction is used), as pycma's
     * xfavorite. Not evaluated. With a noisy objective the value of the best evaluated point is
     * biased low; the mean averages over the population and, with a large population, is often
     * closer to the optimum.
     */
    Vector xfavorite() const;

    /**
     * The points to evaluate next, the columns of a d x k matrix (k >= 1, in the space of the
     * objective). Starts a generation when none is in progress. Until tell(), it returns the same
     * points. The view is valid until the next tell().
     */
    Eigen::Ref<const Matrix> ask();

    /**
     * The objective values of the points of ask(), in the same order. Returns true when this
     * completed a generation (selection, recombination and adaptation are done; break_conditions()
     * tells whether to stop), false when ask() has more points of this generation.
     */
    bool tell(Eigen::Ref<const Vector> f);

    /**
     * The same, for the points X (d x k) instead of the asked ones: a column that differs from the
     * asked point replaces it (a repaired or injected solution). For an individual of the
     * population, its y and z are derived from X with its sigma, as for elitism.
     */
    bool tell(Eigen::Ref<const Vector> f, Eigen::Ref<const Matrix> X);

    //! Whether ask() has points that wait for tell()
    [[nodiscard]] bool has_pending() const { return !p->queue.empty(); }

  private:
    //! The step of the generation that runs once the queue is empty (evaluation::Step)
    evaluation::Step next = evaluation::Step::BEGIN;
    //! The first individual of the population's next batch
    Eigen::Index next_individual = 0;
    //! The objective values of the queued points in step() and mutate()
    Vector values;

    bool tell(Eigen::Ref<const Vector> f, const Eigen::Ref<const Matrix> *X);
    /**
     * Run the generation until points are queued (returns false) or it is complete (true). With
     * stop_before_selection (mutate()), it also returns false, with nothing queued, when every
     * point of the population has its value.
     */
    bool advance(bool stop_before_selection = false);
    //! Queue the next batch of the population (Strategy::sample_next_batch)
    void queue_population_batch();
    //! Queue the extra points of the step-size adaptation (TPA's)
    void queue_extra_points();
    //! The objective values of the queued points
    void evaluate(FunctionType &objective);
    //! select() and adapt() of the 1.x API: not while a generation waits for tell()
    void require_no_generation(const char *step) const;
};
