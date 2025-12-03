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

    using CoordinateTransformType = std::function<Float(Float)>;

    inline Float identity(Float x) { return x; }
    inline Float iround(Float x) { return std::round(x); }

    struct CoordinateMapping
    {
        size_t d;
        std::vector<CoordinateTransformType> transformations;

        CoordinateMapping(const Indices &integer_idx, const size_t d, const Float mueff)
            : d(d), transformations(d, identity)
        {
            for (int i = 0; i < integer_idx.size(); i++)
                transformations[i] = iround;
        }

        Vector transform(const Vector &x)
        {
            Vector y = x;            
            for (size_t i = 0; i < d; i++)
                y(i) = transformations[i](x(i));
            return y;
        }
    };

    inline std::shared_ptr<CoordinateMapping> get(const Indices &integer_idx, const size_t d, const Float mueff)
    {
        return std::make_shared<CoordinateMapping>(integer_idx, d, mueff);
    }
}