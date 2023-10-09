#pragma once

#include "common.hpp"
#include "modules.hpp"
#include "settings.hpp"

namespace parameters
{
    struct Weights 
    {
        Vector weights;
        Vector positive;
        Vector negative;

        double mueff, mueff_neg;
        double c1, cmu, cc;

        Weights(const size_t dim, const size_t mu, const size_t lambda, const Settings &settings);

        void weights_default(const size_t lambda);

        void weights_equal(const size_t mu);

        void weights_half_power_lambda(const size_t mu, const size_t lambda);

        Vector clipped() const;
    };

}