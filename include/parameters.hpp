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

    Parameters(const size_t dim);

    Parameters(const Settings &settings);

    //! The new mean: the weighted recombination of the selected points (all of them, with the
    //! utilities of NES/xNES); the old one becomes adaptation->m_old
    void recombine();

    void adapt();

    /**
     * The points a generation evaluates besides the population, at most; fixed per configuration:
     * TPA's two. The budget reserves them in every generation (break_conditions, restart_fits,
     * fit_lambda_to_budget), also in one that does not evaluate them (TPA after a generation whose
     * elitist selection kept every parent, #179).
     */
    [[nodiscard]] size_t reserved_points() const;

    /**
     * Individual i of the population was evaluated at x instead of its sampled point (tell(f, X)):
     * x replaces it, its y and z are derived from x (clipped when far outside the distribution)
     */
    void inject(const Eigen::Index i, Eigen::Ref<const Vector> x);

    /**
     * Gradient injection (#90): individual 0 of the sampled population becomes the point along
     * the negative gradient g from the mean, at the length of a typical sample:
     * z = -c A^T g / ||A^T g|| with c = E||z|| (weights.expected_length_z; E||N(0, I)|| for a
     * Gaussian sampler), y = A z, x = m + s o y with the individual's step size s (sigma, unless
     * step sizes are sampled or integer variables raise it). With s = sigma this is
     * x = m - sigma c C g / sqrt(g^T C g), Hansen (2011), Eq. (14), with c for his sqrt(d).
     * With mirrored sampling individual 1 becomes its mirror (-z), so that the pair stays a pair.
     * Each point is moved into the box as TPA's points (BoundCorrection::correct_point), its
     * integer variables are rounded, and it enters the update as an injected solution: y and z
     * derived from x, clipped (inject). The samples are drawn as without injection.
     *
     * Returns the number of individuals replaced: 0 when A^T g is zero, or when the repelling
     * archive rejects a point (the samples stay). Throws std::invalid_argument when g does not
     * have d finite entries.
     */
    Eigen::Index inject_gradient(Eigen::Ref<const Vector> g);

    /**
     * The center of mass: the mean of the search distribution as a point of the objective
     * (integer variables rounded; clipped to the bounds when a bound correction is used), pycma's
     * xfavorite. Every run ends with its evaluation (#111).
     */
    [[nodiscard]] Vector center_of_mass() const;

    /*
     * The end of a run, and a restart, in steps (ModularCMAES::advance). Every run ends with the
     * evaluation of its center of mass, when it stops and when a restart is due (#111): end_run
     * (the run's best joins stats.solutions), then advance queues center_of_mass(); its value
     * (tell_center_of_mass) records the run in stats.centers. At a restart, begin_generation
     * decides it, begin_restart calls end_run, and the value of the center starts the update of
     * the repelling archive, whose hill-valley points advance queues one at a time
     * (repelling->query_point()); once they are evaluated, finish_restart resets the modules.
     */

    //! Returns whether a restart started (then the center of mass waits for its value)
    bool begin_generation();

    //! The run ends: its best joins stats.solutions
    void end_run();

    //! A restart, with step size sigma for the new run: the run ends (end_run)
    void begin_restart(Float sigma);

    /**
     * The value y of the run's center of mass, evaluated at x: the run's record joins
     * stats.centers, with the distribution it ended with (sigma, metric) and reasons, and x may
     * become the best solution. At a restart, it starts the update of the repelling archive.
     */
    void tell_center_of_mass(Float y,
                             Eigen::Ref<const Vector> x,
                             std::vector<std::string> reasons,
                             bool restarting);

    void finish_restart();
    //! Caps lambda at the remaining budget, less the evaluation of the new run's center of mass,
    //! with mu = Settings::mu_for(lambda); called by finish_restart
    void fit_lambda_to_budget();
    //! Whether a restart that starts now still fits: one more evaluation (the center, or a
    //! hill-valley point), then the smallest first generation and its reserved points (TPA's), and
    //! the evaluation of the new run's center of mass
    [[nodiscard]] bool restart_fits() const;

    void resize_population(const size_t new_mu, const size_t new_lambda);

  private:
    friend struct serialization::Access;

    //! The step size of the new run, from begin_restart to finish_restart
    Float restart_sigma = 0;

    //! Parameters without modules, for the deserialization (serialization.cpp), which sets every
    //! member
    Parameters(const Settings &settings, const Weights &weights);
};

} // namespace parameters

std::ostream &operator<<(std::ostream &os, const parameters::Stats &dt);