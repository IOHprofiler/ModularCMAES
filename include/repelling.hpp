#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <modules.hpp>
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

bool hill_valley_test(const Solution &u, const Solution &v, FunctionType &f, const size_t n_evals);

bool hill_valley_test_p(const Solution &u,
                        const Solution &v,
                        FunctionType &f,
                        const size_t n_evals,
                        parameters::Parameters &p);
} // namespace distance

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

    // Old-style/default constructor: useful for coverage-based repelling.
    TabooPoint(const Solution &s, const Float radius) :
        solution(s),
        radius(radius),
        shrinkage(std::pow(0.95, 1.0 / static_cast<Float>(s.x.size()))),
        n_rep(1),
        criticality(0.0)
    {
    }

    // Adaptive constructor: radius bounds are used by adaptive repelling.
    TabooPoint(const Solution &s,
               const Float radius,
               const Float min_radius,
               const Float max_radius) :
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
        last_rejection_rate(0.0)
    {
    }

    bool rejects(const Vector &xi, const parameters::Parameters &p, const int attempts) const;

    bool
    shares_basin(FunctionType &objective, const Solution &sol, parameters::Parameters &p) const;

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

    virtual ~Repelling() = default;

    virtual bool is_rejected(const Vector &xi, parameters::Parameters &p);
    virtual void update_archive(FunctionType &objective, parameters::Parameters &p);
    virtual void prepare_sampling(parameters::Parameters &p);
    virtual void adapt_radii_after_sampling(parameters::Parameters &p);

  protected:
    void sort_archive_by_criticality(const parameters::Parameters &p);
};

struct AdaptiveRepelling : Repelling
{
    Float initial_radius_factor = 0.5;
    Float min_radius_factor = 0.05;
    Float max_radius_factor = 2.0;

    Float shrink_factor = 0.90;

    Float target_rejection_rate = 0.20;
    Float max_rejection_rate = 0.50;

    int last_restart_evaluations = 0;
    Float average_restart_evaluations = 1.0;
    Float restart_eval_alpha = 0.20;
    Float grow_eta = 0.15;
    Float max_waste_ratio = 3.0;

    Float local_target_rejection_rate = 0.20;
    Float local_max_rejection_rate = 0.50;
    int min_checks_before_local_shrink = 5;

    Float initial_radius(const parameters::Parameters &p) const;
    Float min_radius(const parameters::Parameters &p) const;
    Float max_radius(const parameters::Parameters &p) const;

    void update_archive(FunctionType &objective, parameters::Parameters &p) override;
    void adapt_radii_after_sampling(parameters::Parameters &p) override;
    void prepare_sampling(parameters::Parameters &p) override;
};

struct CoverageRepelling : Repelling
{
    void update_archive(FunctionType &objective, parameters::Parameters &p) override;
    void adapt_radii_after_sampling(parameters::Parameters &p) override {}
};

struct NoRepelling final : Repelling
{
    bool is_rejected(const Vector &xi, parameters::Parameters &p) override { return false; }

    void update_archive(FunctionType &objective, parameters::Parameters &p) override {}
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