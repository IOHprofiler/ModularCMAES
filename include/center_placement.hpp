#pragma once

#include "common.hpp"
#include "modules.hpp"

namespace parameters
{
struct Parameters;
struct Settings;
} // namespace parameters

namespace center
{
//! The half-width, in units of sigma0, of the interval in which UNIFORM, NOVELTY_WEIGHTED and
//! MAXIMIN_TABOO place a center in a coordinate with an infinite bound (#79 F19): x0 +- 2 sigma0,
//! moved inside the finite bound when there is one. pycma's advice for sigma0 is that the
//! optimum lie "within about x0 +- 3*sigma0" (cma.fmin2, pycma 4.6.0).
inline constexpr Float infinite_box_half_width = 2.0;

//! The box in which UNIFORM, NOVELTY_WEIGHTED and MAXIMIN_TABOO sample their candidates, as
//! lower + width * u with u uniform in (0, 1)^d: [lb, ub] where both bounds are finite; in a
//! coordinate with an infinite bound, an interval of width 2 h, h = infinite_box_half_width *
//! sigma0, around x0 (Settings.x0, or Settings.center): [x0 - h, x0 + h], shifted to start at
//! lb when x0 - h < lb, or to end at ub when x0 + h > ub. The widths also scale the distances to
//! the taboo points.
struct Region
{
    Vector lower;
    Vector width;

    explicit Region(const parameters::Settings &settings);

    //! The point lower + width * u
    [[nodiscard]] Vector point(const Vector &u) const { return lower + u.cwiseProduct(width); }
};

struct Placement
{
    virtual ~Placement() = default;

    virtual void operator()(parameters::Parameters &p) = 0;
};

struct X0 : Placement
{
    void operator()(parameters::Parameters &p) override;
};

struct Uniform : Placement
{
    void operator()(parameters::Parameters &p) override;
};

struct Zero : Placement
{
    void operator()(parameters::Parameters &p) override;
};

struct Center : Placement
{
    void operator()(parameters::Parameters &p) override;
};

struct NoveltyWeighted : Placement
{
    void operator()(parameters::Parameters &p) override;
};

struct MaximinTaboo : Placement
{
    void operator()(parameters::Parameters &p) override;
};

inline std::shared_ptr<Placement> get(const parameters::CenterPlacement &p)
{
    using namespace parameters;
    switch (p)
    {
    case CenterPlacement::UNIFORM:
        return std::make_shared<Uniform>();
    case CenterPlacement::ZERO:
        return std::make_shared<Zero>();
    case CenterPlacement::CENTER:
        return std::make_shared<Center>();
    case CenterPlacement::NOVELTY_WEIGHTED:
        return std::make_shared<NoveltyWeighted>();
    case CenterPlacement::MAXIMIN_TABOO:
        return std::make_shared<MaximinTaboo>();
    default:
    case CenterPlacement::X0:
        return std::make_shared<X0>();
    }
}
} // namespace center