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
 * A generation is a pipeline of stages (evaluation::GenerationStage): a restart when one is due
 * (RUN_ENDING, RUN_STARTING), then GENERATION_STARTING, the POPULATION (with the points
 * ALONGSIDE_POPULATION, TPA's), POPULATION_EVALUATED, selection (SELECTED), recombination and
 * adaptation (UPDATED). At each stage its subscribers may queue points (Parameters::queue);
 * the pipeline waits for tell() whenever the queue is not empty. ask() returns everything that is
 * queued: a whole population (with TPA's two points), or, with sequential selection, the
 * individuals up to the first one after which it can stop, then one (or a mirrored pair) at a
 * time; a restart asks for the final mean, then for the hill-valley points of the repelling
 * archive, one at a time.
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

    void recombine() const;

    //! Selection, then the SELECTED stage (which may not queue points here: use step() or
    //! ask-tell for modules that evaluate after selection)
    void select();

    //! Adaptation, then the UPDATED stage (likewise); ends the generation
    void adapt();

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

    //! The stage of the generation in progress (none between generations)
    [[nodiscard]] std::optional<evaluation::GenerationStage> current_stage() const { return stage; }

  private:
    std::optional<evaluation::GenerationStage> stage;
    //! mutate(): the pipeline stops before selection
    bool stop_before_selection = false;
    //! The objective values of the queued points in step() and mutate()
    Vector values;

    bool tell(Eigen::Ref<const Vector> f, const Eigen::Ref<const Matrix> *X);
    //! Enter a stage: fire its subscribers
    void enter(evaluation::GenerationStage next);
    //! Run the stages until points are queued, or the generation is complete (returns true)
    bool advance();
    //! The objective values of the queued points
    void evaluate(FunctionType &objective);
    //! After select() or adapt(): no stage may wait for points outside ask-tell
    void require_no_pending(const char *step) const;
};
