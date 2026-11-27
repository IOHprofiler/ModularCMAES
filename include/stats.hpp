#pragma once

#include "common.hpp"

namespace parameters
{
/**
 * The record of a finished run (#111): its center of mass, evaluated (x, y; t is the last
 * generation of the run, e the evaluations after it), with the distribution the run ended with,
 * N(x, sigma^2 C), and why it ended.
 */
struct Center : Solution
{
    //! The step size at the end of the run
    Float sigma = 0;
    //! The shape at the end of the run (Adaptation::metric())
    MetricCopy metric;
    //! The names of the restart criteria that were met, and "budget", "target" or
    //! "max_generations" when one of those ended the run
    std::vector<std::string> reasons;

    Center() = default;

    Center(Eigen::Ref<const Vector> x,
           const Float y,
           const size_t t,
           const size_t e,
           const Float sigma,
           MetricCopy metric,
           std::vector<std::string> reasons) :
        Solution(x, y, t, e),
        sigma(sigma),
        metric(std::move(metric)),
        reasons(std::move(reasons))
    {
    }

    //! sigma max(d): the longest semi-axis of the one-sigma ellipsoid (NaN without a metric)
    [[nodiscard]] Float max_std() const
    {
        return metric.d.size() == 0 ? std::numeric_limits<Float>::quiet_NaN()
                                    : sigma * metric.d.maxCoeff();
    }
};

struct Stats
{
    size_t t = 0;
    //! The generations of the current run (reset at a restart)
    size_t t_run = 0;
    size_t evaluations = 0;

    Float current_avg = std::numeric_limits<Float>::infinity();
    //! The best solution of every finished run
    std::vector<Solution> solutions = {};
    //! The record of every finished run: its evaluated center of mass and final distribution
    std::vector<Center> centers = {};
    Solution current_best = {};
    Solution global_best = {};
    bool has_improved = false;
    Float success_ratio = 2.0 / 11.0;
    Float cp = 1.0 / 12.0;
    size_t last_update = 0;
    size_t n_updates = 0;

    /**
     * An evaluated point (x by reference, a column of the population: copied only when it
     * improves). Every point can become current_best, the best of the run, which the success rules
     * and the restarts read. Only a feasible one (Settings::in_domain: inside [lb, ub] with a bound
     * correction) can become global_best (#168): RESAMPLE evaluates a point outside the box after
     * 5 d rejected tries, and the best point that run() and fmin return must lie in the box.
     */
    void update_best(Eigen::Ref<const Vector> x, const Float y, const bool feasible = true)
    {
        has_improved = false;
        if (y < current_best.y)
        {
            current_best = Solution(x, y, t, evaluations);
            has_improved = true;
        }
        if (feasible)
            update_global_best(x, y);
    }

    //! A feasible point that is not the best of its generation: it can only become global_best
    void update_global_best(Eigen::Ref<const Vector> x, const Float y)
    {
        if (y < global_best.y)
            global_best = Solution(x, y, t, evaluations);
    }

    //! Once per generation (selection): a success when the best of the generation beats the best of
    //! the run so far (with elitism, the parent)
    void update_success(const bool success)
    {
        success_ratio = (1 - cp) * success_ratio + (cp * static_cast<Float>(success));
    }
};
} // namespace parameters
