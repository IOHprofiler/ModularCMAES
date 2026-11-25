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

    //! out = x with the integer variables rounded (no allocation)
    void transform_into(Eigen::Ref<const Vector> x, Eigen::Ref<Vector> out) const
    {
        out = x;
        for (const auto idx : integer_idx)
            out(idx) = std::round(out(idx));
    }

    [[nodiscard]] Vector transform(const Vector &x) const
    {
        Vector y(x.size());
        transform_into(x, y);
        return y;
    }
};

inline std::shared_ptr<CoordinateMapping>
get(const Indices &integer_idx, const size_t d)
{
    return std::make_shared<CoordinateMapping>(integer_idx, d);
}
} // namespace mapping