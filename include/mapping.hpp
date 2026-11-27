#pragma once

#include "common.hpp"

namespace mapping
{

//! Maps a sample to the point that is evaluated: the integer variables are rounded
struct CoordinateMapping
{
    std::vector<Eigen::Index> integer_idx;

    CoordinateMapping(const Indices &integer_variables, const size_t d)
    {
        for (const auto idx : integer_variables)
        {
            if (idx < 0 || static_cast<size_t>(idx) >= d)
                throw std::invalid_argument("integer variable index " + std::to_string(idx) +
                                            " is out of range for dimension " + std::to_string(d));
            integer_idx.push_back(idx);
        }
    }

    //! out = x with the integer variables rounded (no allocation). A coordinate inside [lb, ub]
    //! is rounded to the nearest integer in [ceil(lb), floor(ub)], so it stays inside (#79 F21:
    //! with lb = 0.3, 0.4 was rounded to 0); Settings reject an integer variable without an
    //! integer in [lb, ub]. A coordinate outside the box (bound_correction NONE, the last sample
    //! of RESAMPLE, TPA's points) is rounded to the nearest integer and stays outside. With
    //! integral bounds this is plain rounding.
    void transform_into(Eigen::Ref<const Vector> x,
                        Eigen::Ref<Vector> out,
                        const Vector &lb,
                        const Vector &ub) const
    {
        out = x;
        for (const auto idx : integer_idx)
        {
            const Float xi = x(idx);
            Float r = std::round(xi);
            if (r < lb(idx) && xi >= lb(idx))
                r = std::ceil(lb(idx));
            else if (r > ub(idx) && xi <= ub(idx))
                r = std::floor(ub(idx));
            out(idx) = r;
        }
    }

    [[nodiscard]] Vector transform(const Vector &x, const Vector &lb, const Vector &ub) const
    {
        Vector y(x.size());
        transform_into(x, y, lb, ub);
        return y;
    }
};

inline std::shared_ptr<CoordinateMapping>
get(const Indices &integer_idx, const size_t d)
{
    return std::make_shared<CoordinateMapping>(integer_idx, d);
}
} // namespace mapping