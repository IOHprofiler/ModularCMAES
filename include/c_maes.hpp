#pragma once

#include <vector>

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
 * selection, then recombination and adaptation. advance() is the only place that queues points;
 * a step runs again, or the next one runs, once tell() has given their values.
 * ask() returns everything that is queued: a whole population (with TPA's two points), or, with
 * sequential selection, the individuals up to the first one after which it can stop (with TPA's
 * points), then one (or a mirrored pair) at a time. Every run ends with the evaluation of its
 * center of mass, one point (#111): when a restart is due, followed by the hill-valley points of
 * the repelling archive, one at a time; and when the optimizer stops, after which
 * break_conditions() holds.
 *
 * Gradient injection (#90): between the ask() that returns a new population and its tell(),
 * inject_gradient() replaces one of its points by a point along the negative gradient; run(),
 * step() and mutate() do this with a gradient function.
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
     * other steps act on the state as it is. select(), recombine() and adapt() refuse to run while
     * the points of a generation wait for their values (after ask(), or an interrupted mutate()).
     * When the optimizer stops, mutate() evaluates the final center of mass instead of a
     * generation, and select(), recombine() and adapt() then do nothing, so that the 1.x loop
     * ends as run() does.
     */

    void recombine() const;

    void select() const;

    void adapt() const;

    //! Sample and evaluate a generation, up to selection (after a restart, when one is due); when
    //! the optimizer stops, evaluate the final center of mass instead. With a gradient, every new
    //! population gets a gradient point (inject_gradient), with the gradient at center_of_mass()
    void mutate(FunctionType &objective, const GradientType &gradient = {});

    //! One generation (when the optimizer stops: the evaluation of the final center of mass);
    //! returns false when a break condition holds. With a gradient, as mutate()
    bool step(FunctionType &objective, const GradientType &gradient = {});

    //! Run generations while no break condition holds (checked before each generation). With a
    //! gradient, as mutate()
    void operator()(FunctionType &objective, const GradientType &gradient = {});

    /**
     * Whether the optimizer has stopped: a stop condition holds (stop_conditions()) and the run's
     * final center of mass has been evaluated. False again when the stop conditions no longer
     * hold (the budget was raised): the run continues.
     */
    bool break_conditions() const;

    /**
     * Whether the optimizer should stop: the target was reached, max_generations was reached, the
     * next generation would leave no evaluation for the final center of mass (or a restart is
     * due and does not fit), or a restart criterion holds with restart_strategy STOP.
     */
    bool stop_conditions() const;

    /**
     * The center of mass: the mean of the search distribution as a point of the objective
     * (integer variables rounded; inside the bounds when a bound correction is used), pycma's
     * xfavorite. Every run ends with its evaluation (stats.centers, #111). With a noisy objective
     * the value of the best evaluated point is biased low; the mean averages over the population
     * and, with a large population, is often closer to the optimum.
     */
    Vector center_of_mass() const { return p->center_of_mass(); }

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
    [[nodiscard]] bool has_pending() const { return n_pending != 0; }

    /**
     * Whether inject_gradient() can be called now: the points of ask() are the first batch of a
     * new population, and no gradient was injected into it. From the step that samples the
     * population until its first tell().
     */
    [[nodiscard]] bool accepts_gradient() const { return gradient_open; }

    /**
     * Gradient injection (#90): g, the gradient of the objective at center_of_mass(), replaces
     * individual 0 of the new population (with mirrored sampling also its partner, the mirror)
     * by the point along -g from the mean at the length of a typical sample
     * (Parameters::inject_gradient). The next ask() returns it. Only while accepts_gradient()
     * (std::logic_error otherwise); once per population. A zero gradient changes nothing.
     * std::invalid_argument when g does not have d finite entries (then a gradient is still
     * accepted).
     */
    void inject_gradient(Eigen::Ref<const Vector> g);

  private:
    friend struct serialization::Access;

    /**
     * The steps of a generation, in order (advance()). A step may queue points; it runs again, or
     * the next step runs, once they all have their values.
     */
    enum class Step
    {
        //! A new generation. When the optimizer stops or a restart is due, the center of mass of
        //! the run that ends is queued first
        BEGIN,
        //! The final center of mass has its value: the optimizer has stopped
        END,
        //! The restart: the hill-valley points of the repelling archive, one at a time; then the
        //! modules are reset for the new run
        RESTART,
        //! A new population: its first batch, with the points that do not depend on its values
        //! (TPA's two)
        SAMPLE,
        //! The next batch of the population, until it is complete (sequential selection)
        POPULATION,
        //! Every point of the population has its value: selection
        SELECT,
        //! Recombination and adaptation; the generation is complete
        UPDATE,
    };

    //! Who needs the values of queued points: dispatch() passes them on by this tag
    enum class Source
    {
        //! A batch of the population (Request::index: its first individual)
        POPULATION,
        //! The extra points of the step-size adaptation (TPA)
        EXTRA_POINTS,
        //! The center of mass of the run that ends
        CENTER_OF_MASS,
        //! A point of a hill-valley test of the repelling archive
        HILL_VALLEY,
    };

    //! Queued points: columns begin..begin+size-1 of batch, for a source
    struct Request
    {
        Source source;
        Eigen::Index begin;
        Eigen::Index size;
        //! A payload for the module (POPULATION: the first individual of the batch)
        Eigen::Index index;
    };

    //! The step of the generation that runs once the queued points have their values
    Step next = Step::BEGIN;
    //! The queued points (the first n_pending columns: what ask() returns), and the requests they
    //! belong to, in queue order
    Matrix batch;
    Eigen::Index n_pending = 0;
    std::vector<Request> requests;
    //! The first individual of the population's next batch
    Eigen::Index next_individual = 0;
    //! The objective values of the queued points in step() and mutate()
    Vector values;
    //! The run's final center of mass has its value (break_conditions())
    bool ended = false;
    //! The last mutate() evaluated the final center of mass: select(), recombine() and adapt() do
    //! nothing until the next generation
    bool steps_skipped = false;
    //! Why the run that ends now ends, for its record (queued with its center of mass)
    std::vector<std::string> end_reasons;
    //! The pending points are the first batch of a new population, without a gradient point yet
    //! (accepts_gradient)
    bool gradient_open = false;

    //! The stop conditions, each on its own
    struct StopFlags
    {
        bool target = false;
        bool budget = false;
        bool generations = false;
        bool criteria = false;
        [[nodiscard]] bool any() const { return target || budget || generations || criteria; }
    };
    [[nodiscard]] StopFlags stop_flags() const;
    //! The names of the restart criteria that are met, and the stop flags that hold
    [[nodiscard]] std::vector<std::string> reasons(const StopFlags &flags) const;

    bool tell(Eigen::Ref<const Vector> f, const Eigen::Ref<const Matrix> *X);
    //! The values of the queued points: each request's values to its module (dispatch)
    void answer(Eigen::Ref<const Vector> f, const Eigen::Ref<const Matrix> *X);
    //! Pass a request's values f (at the points X; given: by tell(f, X)) to the module that needs
    //! them, by the request's source
    void dispatch(const Request &request,
                  Eigen::Ref<const Vector> f,
                  Eigen::Ref<const Matrix> X,
                  bool given);
    /**
     * Run the generation until points are queued (returns false) or it is complete (true). With
     * stop_before_selection (mutate()), it also returns false, with nothing queued, when every
     * point of the population has its value.
     */
    bool advance(bool stop_before_selection = false);
    //! Queue k points for a source: returns their columns, to be filled before the next queue()
    Eigen::Ref<Matrix> queue(Eigen::Index k, Source source, Eigen::Index index = 0);
    //! Queue the points X (copied)
    void queue(Eigen::Ref<const Matrix> X, Source source);
    //! Queue the next batch of the population (Strategy::sample_batch_from)
    void queue_population_batch();
    //! Queue the extra points of the step-size adaptation (TPA's)
    void queue_extra_points();
    //! The objective values of the queued points
    void evaluate(FunctionType &objective);
    //! With a gradient, while accepts_gradient(): inject_gradient(gradient(center_of_mass()))
    void offer_gradient(const GradientType &gradient);
    //! select(), recombine() and adapt() of the 1.x API: not while points wait for their values
    void require_no_generation(const char *step) const;
};
