#pragma once

#include "common.hpp"
#include "modules.hpp"
#include "sampling.hpp"

namespace parameters
{
struct Parameters;
struct Settings;
} // namespace parameters

namespace bounds
{
Mask is_out_of_bounds(Eigen::Ref<const Vector> xi, const Vector &lb, const Vector &ub);
//! Whether any coordinate of xi is outside [lb, ub] (without allocating a mask)
bool any_out_of_bounds(Eigen::Ref<const Vector> xi, const Vector &lb, const Vector &ub);

struct BoundCorrection
{
    size_t n_out_of_bounds = 0;

    virtual ~BoundCorrection() = default;

    void correct(const Eigen::Index i, parameters::Parameters &p);

    /**
     * A point that is not sampled from the search distribution, moved into [lb, ub] as correct()
     * moves an individual: by the method, per coordinate and per side. TPA uses it for its two
     * points (#98) when the mean itself is outside the box; otherwise it moves them toward the
     * mean, symmetrically, until both are inside.
     * RESAMPLE cannot sample such a point again: it sets the coordinates outside the box on the
     * bound they crossed (as SATURATE). NONE leaves the point as it is. Not counted in
     * n_out_of_bounds (which counts the population).
     */
    [[nodiscard]] Vector
    correct_point(const Vector &xi, const Float sigma, const parameters::Settings &settings);

    //! xi with the coordinates in oob moved into [lb, ub], per coordinate and per side (#79 F20).
    //! The method (correct_finite) corrects the coordinates with two finite bounds. A coordinate
    //! with one infinite bound is set to the finite bound it crossed, as SATURATE does: every
    //! method but SATURATE needs the width ub - lb (MIRROR's period, TOROIDAL's wrap, COTN's
    //! distance, UNIFORM_RESAMPLE's interval).
    virtual Vector correct_x(const Vector &xi,
                             const Mask &oob,
                             const Float sigma,
                             const parameters::Settings &settings);

    //! The method itself: xi with the coordinates in oob moved into [lb, ub]; every coordinate in
    //! oob has two finite bounds
    virtual Vector correct_finite(const Vector &xi,
                                  const Mask &oob,
                                  const Float sigma,
                                  const parameters::Settings &settings) = 0;

    [[nodiscard]] Mask is_out_of_bounds(const Vector &xi,
                                        const parameters::Settings &settings) const;

    [[nodiscard]] Vector delta_out_of_bounds(const Vector &xi,
                                             const Mask &oob,
                                             const parameters::Settings &settings) const;
};

struct NoCorrection : BoundCorrection
{
    //! Nothing is moved, on either side (NONE; RESAMPLE samples again instead)
    Vector correct_x(const Vector &xi,
                     const Mask &oob,
                     const Float sigma,
                     const parameters::Settings &settings) override
    {
        return xi;
    }

    Vector correct_finite(const Vector &xi,
                          const Mask &oob,
                          const Float sigma,
                          const parameters::Settings &settings) override
    {
        return xi;
    }
};

struct Resample final : NoCorrection
{
};

struct COTN final : BoundCorrection
{
    sampling::Gaussian sampler;

    COTN(const size_t d, const std::shared_ptr<rng::Generator> &generator = rng::global()) :
        sampler(d, rng::normal<Float>(0, 1.0 / 3.), generator)
    {
    }

    Vector correct_finite(const Vector &xi,
                          const Mask &oob,
                          const Float sigma,
                          const parameters::Settings &settings) override;
};

struct Mirror final : BoundCorrection
{
    Vector correct_finite(const Vector &xi,
                          const Mask &oob,
                          const Float sigma,
                          const parameters::Settings &settings) override;
};

struct UniformResample final : BoundCorrection
{
    sampling::Uniform sampler;

    UniformResample(const size_t d,
                    const std::shared_ptr<rng::Generator> &generator = rng::global()) :
        sampler(d, generator)
    {
    }

    Vector correct_finite(const Vector &xi,
                          const Mask &oob,
                          const Float sigma,
                          const parameters::Settings &settings) override;
};

struct Saturate final : BoundCorrection
{
    Vector correct_finite(const Vector &xi,
                          const Mask &oob,
                          const Float sigma,
                          const parameters::Settings &settings) override;
};

struct Toroidal final : BoundCorrection
{
    Vector correct_finite(const Vector &xi,
                          const Mask &oob,
                          const Float sigma,
                          const parameters::Settings &settings) override;
};

//! The bound correction; COTN and UNIFORM_RESAMPLE draw from generator
inline std::shared_ptr<BoundCorrection>
get(const parameters::CorrectionMethod &m,
    const size_t d,
    const std::shared_ptr<rng::Generator> &generator = rng::global())
{
    using namespace parameters;
    switch (m)
    {
    case CorrectionMethod::MIRROR:
        return std::make_shared<Mirror>();
    case CorrectionMethod::COTN:
        return std::make_shared<COTN>(d, generator);
    case CorrectionMethod::UNIFORM_RESAMPLE:
        return std::make_shared<UniformResample>(d, generator);
    case CorrectionMethod::SATURATE:
        return std::make_shared<Saturate>();
    case CorrectionMethod::TOROIDAL:
        return std::make_shared<Toroidal>();
    case CorrectionMethod::RESAMPLE:
        return std::make_shared<Resample>();
    default:
    case CorrectionMethod::NONE:
        return std::make_shared<NoCorrection>();
    }
};
} // namespace bounds
