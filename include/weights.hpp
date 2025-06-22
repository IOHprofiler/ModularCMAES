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
        Float c1, cmu, cc, cs;
        Float damps, acov;
        Float sqrt_cc_mueff, sqrt_cs_mueff;
        Float lazy_update_interval;
        Float expected_length_z;
        Float expected_length_ps;
        Float beta;

        Weights(const size_t dim, const size_t mu, const size_t lambda, const Settings &settings, 
            const Float expected_length_z);

        void weights_default(const size_t mu, const size_t lambda);

        void weights_equal(const size_t mu);

        void weights_exponential(const size_t mu, const size_t lambda);

        Vector clipped() const;
    };

}