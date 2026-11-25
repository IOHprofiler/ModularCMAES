#pragma once

#include "bounds.hpp"
#include "center_placement.hpp"
#include "evaluation.hpp"
#include "mapping.hpp"
#include "matrix_adaptation.hpp"
#include "mutation.hpp"
#include "population.hpp"
#include "repelling.hpp"
#include "restart_criteria.hpp"
#include "restart_strategy.hpp"
#include "sampling.hpp"
#include "selection.hpp"
#include "stats.hpp"
#include "weights.hpp"

namespace parameters
{
struct Parameters
{
    //! The smallest first generation after a restart that fits the rest of the budget (lambda = 1
    //! is the (1+1)-ES): the budget check and the hill-valley tests of repelling keep this much
    static constexpr size_t min_lambda_after_restart = 2;

    bool successful_adaptation;

    size_t lambda;
    size_t mu;

    Settings settings;
    Stats stats;

    //! The random generator of this optimizer: every module draws from it
    std::shared_ptr<rng::Generator> generator;

    std::shared_ptr<sampling::Sampler> sampler;

    Weights weights;

    Population pop;
    Population old_pop;

    restart::Criteria criteria;
    std::shared_ptr<matrix_adaptation::Adaptation> adaptation;
    std::shared_ptr<mutation::Strategy> mutation;
    std::shared_ptr<selection::Strategy> selection;
    std::shared_ptr<restart::Strategy> restart_strategy;
    std::shared_ptr<bounds::BoundCorrection> bounds;
    std::shared_ptr<repelling::Repelling> repelling;
    std::shared_ptr<center::Placement> center_placement;
    std::shared_ptr<mapping::CoordinateMapping> coordinate_mapping;
    //! Every evaluation of the objective: the pending requests (#84)
    evaluation::EvaluationQueue queue;

    Parameters(const size_t dim);

    Parameters(const Settings &settings);

    //! The start of a generation, with an objective for a restart (begin_generation, and the
    //! restart evaluations when it starts one)
    void start(FunctionType &objective);

    void adapt();

    //! A restart with an objective function: begin_restart, the evaluations, finish_restart
    void perform_restart(FunctionType &objective, const std::optional<Float> &sigma = std::nullopt);

    /**
     * The points the modules queue in a step of every generation besides the population, fixed per
     * configuration: TPA's two, in SAMPLE. The budget reserves them (break_conditions,
     * restart_fits, fit_lambda_to_budget, budget_left).
     */
    [[nodiscard]] size_t reserved_points(evaluation::Step step) const;

    //! The points every generation evaluates besides the population: reserved_points of all steps
    [[nodiscard]] size_t reserved_points() const;

    /**
     * The evaluations that a request with a variable number of points may use, queued while `next`
     * is the step that runs once the queue is empty: the budget minus the evaluations, the queued
     * points, the reserved points of `next` and the steps after it, and, at a restart
     * (RESTART: the hill-valley tests), the smallest first generation of the next run
     * (min_lambda_after_restart). A module with a variable number of points clips its requests to
     * it.
     */
    [[nodiscard]] size_t budget_left(evaluation::Step next) const;

    /**
     * Individual i of the population was evaluated at x instead of its sampled point (tell(f, X)):
     * x replaces it, its y and z are derived from x (clipped when far outside the distribution)
     */
    void inject(const Eigen::Index i, Eigen::Ref<const Vector> x);

    /*
     * The same, in steps (ask-tell). begin_generation makes the previous population old_pop and
     * decides whether to restart. ModularCMAES::advance then queues the center (restart_center());
     * its value (tell_restart_center) starts the update of the repelling archive, whose hill-valley
     * points advance queues one at a time (repelling->query_point()); once they are evaluated,
     * finish_restart resets the modules.
     */

    //! Returns whether a restart started (then restart_center() waits for its objective value)
    bool begin_generation();

    void begin_restart(const std::optional<Float> &sigma = std::nullopt);

    //! The point at which a restart evaluates the objective (the center, in objective space)
    [[nodiscard]] const Vector &restart_center() const { return pending_restart_center; }

    //! The objective value y of the center x; starts the repelling archive update
    void tell_restart_center(const Float y, Eigen::Ref<const Vector> x);

    void finish_restart();
    //! Caps lambda (and mu, in proportion) at the remaining budget; called by finish_restart
    void fit_lambda_to_budget();
    //! Whether a restart that starts now still fits: n more evaluations (the center, or a
    //! hill-valley point), then the smallest first generation and its reserved points (TPA's)
    [[nodiscard]] bool restart_fits(size_t n) const;

    void resize_population(const size_t new_mu, const size_t new_lambda);

  private:
    std::optional<Float> restart_sigma;
    Vector pending_restart_center;

    //! The evaluations of a started restart with an objective function, then finish_restart
    void complete_restart(FunctionType &objective);
};

} // namespace parameters

std::ostream &operator<<(std::ostream &os, const parameters::Stats &dt);