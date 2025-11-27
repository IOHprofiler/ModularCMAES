#pragma once

#include "common.hpp"

namespace parameters
{
	struct Parameters;
}


namespace integer
{

    struct IntegerHandling
    {
        Float lb_sigma;

        IntegerHandling(const size_t d, const Float mueff) 
            : 
            lb_sigma(std::min(0.2, mueff / static_cast<Float>(d)))
        {
        }

        virtual Array get_effective_sigma(const parameters::Parameters& p, const size_t idx); 

        virtual Vector round_to_integer(const Vector& x, const Indices iidx) 
        {
            auto x_rounded = x;
            for (const auto& idx: iidx)
                x_rounded[idx] = std::round(x[idx]);
            return x_rounded;
        }
    };

    struct NoIntegerHandling : IntegerHandling
    {
        using IntegerHandling::IntegerHandling;

        // virtual void get_effective_sigma(const parameters::Parameters& p) override {}
        // void round_to_integer(Eigen::Ref<Vector>  x, const Indices iidx) override {}
    };

    inline std::shared_ptr<IntegerHandling> get(const Indices &idx, const size_t d, const Float mueff)
    {
        if (idx.size() == 0)
            return std::make_shared<NoIntegerHandling>(d, mueff);

        return std::make_shared<IntegerHandling>(d, mueff);
    }
}