#include "bounds.hpp"
#include "parameters.hpp"
#include "population.hpp"

//! x mod 2 in [0, 2), for the integer x = floor(y) of MIRROR, in floating point: as an int, a point
//! more than 2^31 widths outside the box was undefined behaviour (exact for |x| < 2^53, where the
//! integers are exact; the larger doubles are even)
static Float modulo2(const Float x) { return x - 2.0 * std::floor(x / 2.0); }

namespace bounds
{

Mask is_out_of_bounds(Eigen::Ref<const Vector> xi, const Vector &lb, const Vector &ub)
{
    return xi.array() < lb.array() || xi.array() > ub.array();
}

bool any_out_of_bounds(Eigen::Ref<const Vector> xi, const Vector &lb, const Vector &ub)
{
    return ((xi.array() < lb.array()) || (xi.array() > ub.array())).any();
}

Mask BoundCorrection::is_out_of_bounds(const Vector &xi, const parameters::Settings &settings) const
{
    return bounds::is_out_of_bounds(xi, settings.lb, settings.ub);
}

Vector BoundCorrection::delta_out_of_bounds(const Vector &xi,
                                            const Mask &oob,
                                            const parameters::Settings &settings) const
{
    return (oob).select((xi - settings.lb).cwiseQuotient(settings.db), xi);
}

void BoundCorrection::correct(const Eigen::Index i, parameters::Parameters &p)
{
    // Per coordinate and per side (#79 F20): an infinite bound is never crossed, the finite
    // bounds of a box that is not finite are corrected. The common case, inside the bounds, needs
    // no mask.
    if (!bounds::any_out_of_bounds(p.pop.X.col(i), p.settings.lb, p.settings.ub))
        return;

    n_out_of_bounds++;
    if (p.settings.modules.bound_correction == parameters::CorrectionMethod::NONE)
        return;

    const Mask oob = bounds::is_out_of_bounds(p.pop.X.col(i), p.settings.lb, p.settings.ub);
    p.pop.X.col(i) = correct_x(p.pop.X.col(i), oob, p.mutation->sigma, p.settings);
    p.pop.X_transformed.col(i) = p.pop.X.col(i);
    p.adaptation->invert_x_into(p.pop.X.col(i), p.pop.S.col(i), p.pop.Y.col(i));
    p.adaptation->invert_y_into(p.pop.Y.col(i), p.pop.Z.col(i));
}

Vector BoundCorrection::correct_point(const Vector &xi,
                                      const Float sigma,
                                      const parameters::Settings &settings)
{
    if (!bounds::any_out_of_bounds(xi, settings.lb, settings.ub))
        return xi;
    switch (settings.modules.bound_correction)
    {
    case parameters::CorrectionMethod::NONE:
        return xi;
    case parameters::CorrectionMethod::RESAMPLE:
        // A point that is not a sample cannot be sampled again: clipped, as the center of mass
        // (Parameters::center_of_mass)
        return xi.cwiseMax(settings.lb).cwiseMin(settings.ub);
    default:
        return correct_x(
            xi, bounds::is_out_of_bounds(xi, settings.lb, settings.ub), sigma, settings);
    }
}

Vector BoundCorrection::correct_x(const Vector &xi,
                                  const Mask &oob,
                                  const Float sigma,
                                  const parameters::Settings &settings)
{
    // A coordinate with a finite, positive width is corrected by the method. A zero-width
    // coordinate (lb_i == ub_i, a fixed variable) would divide by its width (#133).
    const Mask usable = settings.db.array().isFinite() && settings.db.array() > 0;
    if (usable.all())
        return correct_finite(xi, oob, sigma, settings);

    // A coordinate with an infinite bound has no width: SATURATE on its finite side (#79 F20).
    // A fixed coordinate has none either: SATURATE puts it at lb_i = ub_i (#133).
    // The method draws its random numbers (COTN, UNIFORM_RESAMPLE) for every coordinate, as with
    // a finite box.
    const Mask by_method = oob && usable;
    const Mask saturated = oob && !usable;
    const Vector x = correct_finite(xi, by_method, sigma, settings);
    return saturated.select(xi.cwiseMax(settings.lb).cwiseMin(settings.ub), x);
}

Vector COTN::correct_finite(const Vector &xi,
                            const Mask &oob,
                            const Float sigma,
                            const parameters::Settings &settings)
{
    // The distance from the violated bound, in units of the domain width: sigma |n| with n normal
    // (standard deviation 1/3), redrawn where it would leave the box on the other side (Caraffini
    // et al. 2019: the new coordinate is inside the search space)
    Vector r = sigma * sampler().array().abs();
    for (int attempt = 0; attempt < 10 && (r.array() > 1.0).any(); ++attempt)
    {
        const Vector fresh = sigma * sampler().array().abs();
        r = (r.array() > 1.0).select(fresh, r);
    }
    r = r.cwiseMin(1.0);
    const Vector y = delta_out_of_bounds(xi, oob, settings);
    return (oob).select(
        settings.lb.array() +
            settings.db.array() * ((y.array() > 0).cast<Float>() - r.array()).abs(),
        y);
}

Vector Mirror::correct_finite(const Vector &xi,
                              const Mask &oob,
                              const Float sigma,
                              const parameters::Settings &settings)
{
    const Vector y = delta_out_of_bounds(xi, oob, settings);
    return (oob).select(
        settings.lb.array() +
            settings.db.array() *
                (y.array() - y.array().floor() - y.array().floor().unaryExpr(&modulo2)).abs(),
        y);
}

Vector UniformResample::correct_finite(const Vector &xi,
                                       const Mask &oob,
                                       const Float sigma,
                                       const parameters::Settings &settings)
{
    return (oob).select(settings.lb + sampler().cwiseProduct(settings.db), xi);
}

Vector Saturate::correct_finite(const Vector &xi,
                                const Mask &oob,
                                const Float sigma,
                                const parameters::Settings &settings)
{
    const Vector y = delta_out_of_bounds(xi, oob, settings);
    return (oob).select(settings.lb.array() + settings.db.array() * (y.array() > 0).cast<Float>(),
                        y);
}

Vector Toroidal::correct_finite(const Vector &xi,
                                const Mask &oob,
                                const Float sigma,
                                const parameters::Settings &settings)
{
    const Vector y = delta_out_of_bounds(xi, oob, settings);
    return (oob).select(
        settings.lb.array() + settings.db.array() * (y.array() - y.array().floor()).abs(), y);
}
} // namespace bounds
