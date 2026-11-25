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
    //! The stages of a generation and their subscribers (restart center, TPA, ...)
    evaluation::Pipeline pipeline;

    Parameters(const size_t dim);

    Parameters(const Settings &settings);

    //! The start of a generation, with an objective for a restart (begin_generation, and the
    //! restart evaluations when it starts one)
    void start(FunctionType &objective);

    void adapt();

    //! A restart with an objective function: begin_restart, the evaluations, finish_restart
    void perform_restart(FunctionType &objective, const std::optional<Float> &sigma = std::nullopt);

    //! The evaluations left in the budget, minus the points already queued
    [[nodiscard]] size_t budget_left() const;

    /**
     * Queue a request whose number of points varies (re-evaluations, hill-valley tests): it must fit
     * the budget (budget_left()); a request that does not is refused (std::logic_error). The fixed
     * points of a generation (the population, Pipeline::fixed_points) are checked by
     * break_conditions before the generation starts.
     */
    void push_variable(Eigen::Ref<const Matrix> X, evaluation::Consumer consume, const char *source);

    /**
     * Individual i of the population was evaluated at x instead of its sampled point (tell(f, X)):
     * x replaces it, its y and z are derived from x (clipped when far outside the distribution)
     */
    void inject(const Eigen::Index i, Eigen::Ref<const Vector> x);

    /*
     * The same, in stages (ask-tell). begin_generation makes the previous population old_pop and
     * decides whether to restart. A restart then fires RUN_ENDING: the center is queued
     * (restart_center(), tell_restart_center), and the repelling archive update queues its
     * hill-valley points; once they are evaluated, finish_restart resets the modules.
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
    //! Whether a restart that starts now still fits: n more evaluations (the center, or a hill-valley
    //! point), then the smallest first generation (and the fixed points of the pipeline, TPA's)
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