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
    Float int_lb_sigma;

    Weights(const Settings &settings, const Float expected_length_z);

    void weights_default(const size_t mu, const size_t lambda);

    void weights_equal(const size_t mu);

    void weights_exponential(const size_t mu, const size_t lambda);

    Vector clipped() const;

    /**
     * @brief The weights for a population of n evaluated individuals, mu < n <= lambda (when
     * sequential selection stopped a generation early). The positive weights are unchanged;
     * the negative weights of ranks mu..n-1 are scaled so that their sum equals the sum of all
     * lambda - mu negative weights. For n >= lambda these are the weights themselves.
     */
    Vector for_population(const size_t n) const;

    void init_weights(const Settings &settings);

    void init_vectors(const size_t mu, const size_t lambda, const Settings &settings);

    void init(const size_t mu, const size_t lambda, const Settings &settings);
};

} // namespace parameters