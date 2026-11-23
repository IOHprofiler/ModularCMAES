#pragma once

#include "bounds.hpp"
#include "center_placement.hpp"
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
    bool successful_adaptation;

    size_t lambda;
    size_t mu;

    Settings settings;
    Stats stats;
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

    Parameters(const size_t dim);

    Parameters(const Settings &settings);

    //! The start of a generation, with an objective for a restart (begin_generation, and the
    //! restart evaluations when it starts one)
    void start(FunctionType &objective);

    void adapt();

    //! A restart with an objective function: begin_restart, the evaluations, finish_restart
    void perform_restart(FunctionType &objective, const std::optional<Float> &sigma = std::nullopt);

    /*
     * The same, resumable (ask-tell). begin_generation makes the previous population old_pop and
     * decides whether to restart. A restart then evaluates the center (restart_center(),
     * tell_restart_center), and the hill-valley points of the repelling archive update
     * (repelling->has_query()), and finish_restart resets the modules.
     */

    //! Returns whether a restart started (then restart_center() waits for its objective value)
    bool begin_generation();

    void begin_restart(const std::optional<Float> &sigma = std::nullopt);

    //! The point at which a restart evaluates the objective (the center, in objective space)
    [[nodiscard]] const Vector &restart_center() const { return pending_restart_center; }

    //! The objective value y of the center x; starts the repelling archive update
    void tell_restart_center(const Float y, Eigen::Ref<const Vector> x);

    void finish_restart();

    void resize_population(const size_t new_mu, const size_t new_lambda);

  private:
    std::optional<Float> restart_sigma;
    Vector pending_restart_center;

    //! The evaluations of a started restart with an objective function, then finish_restart
    void complete_restart(FunctionType &objective);
};

} // namespace parameters

std::ostream &operator<<(std::ostream &os, const parameters::Stats &dt);