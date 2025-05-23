#pragma once


#include "settings.hpp"

namespace parameters
{
    struct Weights 
    {
        Vector weights;
        Vector positive;
        Vector negative;

        Float mueff, mueff_neg;
        Float c1, cmu, cc;
        Float lazy_update_interval;
        Float sigma_path_scale;

        Weights(const size_t dim, const size_t mu, const size_t lambda, const Settings &settings);

        void weights_default(const size_t lambda);

        void weights_equal(const size_t mu);

        void weights_half_power_lambda(const size_t mu, const size_t lambda);

        Vector clipped() const;
    };

}