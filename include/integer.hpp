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
        Vector ones;
        Vector effective_y;

        IntegerHandling(const size_t d, const Float mueff) 
            : 
            lb_sigma(std::min(0.2, mueff / static_cast<Float>(d))), 
            ones(Vector::Ones(d)), 
            effective_y(Vector::Ones(d))
        {
        }

        virtual void update_diagonal(const parameters::Parameters& p);

        virtual Array get_effective_sigma(const parameters::Parameters& p, const size_t idx); 

        virtual void round_to_integer(Eigen::Ref<Vector> x, const Indices iidx) 
        {
            for (const auto& idx: iidx)
                x[idx] = std::round(x[idx]);
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