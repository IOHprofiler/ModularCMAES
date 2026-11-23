#pragma once

#include "common.hpp"

namespace parameters
{
struct Parameters;
}

namespace mapping
{

// struct IntegerHandling
// {
//     Float lb_sigma;

//     IntegerHandling(const size_t d, const Float mueff)
//         : lb_sigma(std::min(0.2, mueff / static_cast<Float>(d)))
//     {
//     }

//     virtual Array get_effective_sigma(const parameters::Parameters &p, const size_t idx);

//     virtual Vector round_to_integer(const Vector &x, const Indices iidx)
//     {
//         auto x_rounded = x;
//         for (const auto &idx : iidx)
//             x_rounded[idx] = std::round(x[idx]);
//         return x_rounded;
//     }
// };

// struct NoIntegerHandling : IntegerHandling
// {
//     using IntegerHandling::IntegerHandling;

//     // virtual void get_effective_sigma(const parameters::Parameters& p) override {}
//     // void round_to_integer(Eigen::Ref<Vector>  x, const Indices iidx) override {}
// };
//  Array IntegerHandling::get_effective_sigma(const parameters::Parameters& p, const size_t idx)
// {
//     Array effective_sigma = Array::Constant(p.settings.dim, p.pop.S(idx));

//     const Array& Cdiag = p.adaptation->coordinate_wise_variances;
//     for (const auto& iidx: p.settings.integer_variables)
//     {
//         const Float Cii = std::max(Cdiag[iidx], Float(1e-16));
//         const Float sigma_from_lb = lb_sigma / std::sqrt(Cii);
//         effective_sigma[iidx] = std::max(sigma_from_lb, effective_sigma[iidx]);
//     }

//     return effective_sigma;
// }

//! Maps a sample to the point that is evaluated: the integer variables are rounded
struct CoordinateMapping
{
    size_t d;
    std::vector<Eigen::Index> integer_idx;

    CoordinateMapping(const Indices &integer_variables, const size_t d, const Float mueff) :
        d(d)
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
get(const Indices &integer_idx, const size_t d, const Float mueff)
{
    return std::make_shared<CoordinateMapping>(integer_idx, d, mueff);
}
} // namespace mapping