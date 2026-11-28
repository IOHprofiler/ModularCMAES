#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <modules.hpp>
#include <optional>
#include <vector>

#include "common.hpp"

namespace parameters
{
struct Parameters;
}

namespace repelling
{

namespace distance
{
Float manhattan(const Vector &u, const Vector &v);
Float euclidian(const Vector &u, const Vector &v);
Float mahanolobis(const Vector &u, const Vector &v, const Matrix &C_inv);
} // namespace distance

/**
 * @brief The metric that RepellingMetric::ARCHIVED stores with a new taboo point: A^-1 of the
 * current run (from Adaptation::invert_y, so for every matrix adaptation), scaled to unit
 * determinant. Empty when another metric is selected.
 */
Matrix archived_metric(parameters::Parameters &p);

struct TabooPoint
{
    Solution solution;
    Float radius;
    Float shrinkage;
    int n_rep;
    Float criticality;

    Float min_radius = 0.0;
    Float max_radius = std::numeric_limits<Float>::infinity();

    Float duplicate_evaluations = 0.0;

    int rejected_count = 0;
    int checked_count = 0;
    Float last_rejection_rate = 0.0;

    //! For RepellingMetric::ARCHIVED: A^-1 of the run that found the point, scaled to unit
    //! determinant (only the shape of its basin). Empty for the other metrics.
    Matrix metric;

    // Adaptive constructor: radius bounds are used by adaptive repelling.
    TabooPoint(const Solution &s,
               const Float radius,
               const Float min_radius,
               const Float max_radius,
               const Matrix &metric = Matrix()) :
        solution(s),
        radius(radius),
        shrinkage(std::pow(0.95, 1.0 / static_cast<Float>(s.x.size()))),
        n_rep(1),
        criticality(0.0),
        min_radius(min_radius),
        max_radius(max_radius),
        duplicate_evaluations(0.0),
        rejected_count(0),
        checked_count(0),
        last_rejection_rate(0.0),
        metric(metric)
    {
    }

    // Old-style/default constructor: useful for coverage-based repelling.
    TabooPoint(const Solution &s, const Float radius, const Matrix &metric = Matrix()) :
        TabooPoint(s, radius, 0.0, std::numeric_limits<Float>::infinity(), metric)
    {
    }

    //! The distance from x to this point in the metric of p.settings.modules.repelling_metric,
    //! divided by the current sigma
    Float distance(Eigen::Ref<const Vector> x, const parameters::Parameters &p) const;

    bool rejects(Eigen::Ref<const Vector> xi, const parameters::Parameters &p, const int attempts) const;

    void calculate_criticality(const parameters::Parameters &p);

    void clamp_radius() { radius = std::max(min_radius, std::min(radius, max_radius)); }
};

struct Repelling
{
    std::vector<TabooPoint> archive;
    int attempts = 0;

    // Kept here because both old and adaptive implementations may expose it,
    // but adaptive repelling does not have to use it.
    Float coverage = 2.0;

    //! The number of points of a hill-valley test between a candidate and a taboo point
    static constexpr size_t hill_valley_points = 10;

    virtual ~Repelling() = default;

    virtual bool is_rejected(Eigen::Ref<const Vector> xi, parameters::Parameters &p);
    virtual void prepare_sampling(parameters::Parameters &p);
    virtual void adapt_radii_after_sampling(parameters::Parameters &p);

    //! Start the archive update after a restart (the last center and solution are in p.stats):
    //! the first hill-valley point is query_point(); each answer() sets the next, until has_query()
    //! is false
    void begin_update(parameters::Parameters &p);

    //! Whether the update waits for the objective value of query_point()
    [[nodiscard]] bool has_query() const { return query.active; }

    //! The next point of a hill-valley test (in the space of the objective)
    [[nodiscard]] const Vector &query_point() const { return query.x; }

    //! The objective value y of the point x (query_point(), or a point that replaces it); the
    //! update continues until the next point is needed or it is complete
    void answer(const Float y, Eigen::Ref<const Vector> x, parameters::Parameters &p);

  protected:
    void sort_archive_by_criticality(const parameters::Parameters &p);

    //! Whether begin_update tests the candidate against the archive (false: no archive update)
    [[nodiscard]] virtual bool updates_archive() const { return false; }

    //! Called by begin_update at every restart, before the candidate is chosen (also when the run
    //! left no finite candidate and the archive is not updated)
    virtual void on_restart(parameters::Parameters &p) {}

    //! The candidate shares the basin of point; returns whether to test the remaining points
    virtual bool on_shared_basin(TabooPoint &point,
                                 const Solution &candidate,
                                 parameters::Parameters &p)
    {
        return true;
    }

    //! Called after the last hill-valley test; accepted: the candidate is a new basin
    virtual void on_update_end(const Solution &candidate, bool accepted, parameters::Parameters &p)
    {
    }

  private:
    friend struct serialization::Access;
    //! The state of the resumable update: hill-valley test of point j, step k
    struct Query
    {
        bool active = false;
        std::optional<Solution> candidate;
        size_t j = 0;
        size_t k = 1;
        bool accept = true;
        Vector x;
    } query;

    //! Set the query to the next point to evaluate, or complete the update
    void next_query(parameters::Parameters &p);
};

struct AdaptiveRepelling : Repelling
{
    Float initial_radius_factor = 0.5;
    Float min_radius_factor = 0.05;
    Float max_radius_factor = 2.0;

    Float shrink_factor = 0.90;

    Float target_rejection_rate = 0.20;
    Float max_rejection_rate = 0.50;

    //! The evaluation count at the last restart
    size_t last_restart_evaluations = 0;
    //! The running average of the evaluations per run: 0 until the first restart, which sets it
    //! to the first run's length (#121)
    Float average_restart_evaluations = 0.0;
    Float restart_eval_alpha = 0.20;
    Float grow_eta = 0.15;
    Float max_waste_ratio = 3.0;

    Float local_target_rejection_rate = 0.20;
    Float local_max_rejection_rate = 0.50;
    int min_checks_before_local_shrink = 5;

    Float initial_radius(const parameters::Parameters &p) const;
    Float min_radius(const parameters::Parameters &p) const;
    Float max_radius(const parameters::Parameters &p) const;

    void adapt_radii_after_sampling(parameters::Parameters &p) override;

  protected:
    bool updates_archive() const override { return true; }
    void on_restart(parameters::Parameters &p) override;
    bool on_shared_basin(TabooPoint &point,
                         const Solution &candidate,
                         parameters::Parameters &p) override;
    void on_update_end(const Solution &candidate, bool accepted, parameters::Parameters &p) override;

  private:
    friend struct serialization::Access;
    //! The evaluations since the previous restart, set by on_restart
    Float restart_evals = 0.0;
};

struct CoverageRepelling : Repelling
{
    void adapt_radii_after_sampling(parameters::Parameters &p) override {}

  protected:
    bool updates_archive() const override { return true; }
    bool on_shared_basin(TabooPoint &point,
                         const Solution &candidate,
                         parameters::Parameters &p) override;
    void on_update_end(const Solution &candidate, bool accepted, parameters::Parameters &p) override;
};

struct NoRepelling final : Repelling
{
    bool is_rejected(Eigen::Ref<const Vector>, parameters::Parameters &) override { return false; }

    void prepare_sampling(parameters::Parameters &p) override {}
    void adapt_radii_after_sampling(parameters::Parameters &p) override {}
};

inline std::shared_ptr<Repelling> get(const parameters::Modules &m)
{
    switch (m.repelling_type)
    {
    case parameters::RepellingType::COVERAGE:
        return std::make_shared<CoverageRepelling>();

    case parameters::RepellingType::ADAPTIVE:
        return std::make_shared<AdaptiveRepelling>();

    case parameters::RepellingType::NONE:
    default:
        return std::make_shared<NoRepelling>();
    }
}

} // namespace repelling