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
    //! The variance effective selection mass of the weights the mean is recombined with,
    //! 1 / sum(w_i^2): mueff, except for UTILITY, where the mean uses all lambda utilities
    Float mueff_mean;
    //! RecombinationWeights::UTILITY: the utilities of NES/xNES, which sum to zero, recombine the
    //! mean from all evaluated points and are the weights of every matrix update
    bool utility = false;
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

    /**
     * The utilities of NES/xNES for n ranked points (Wierstra et al. 2014; Glasmachers et al.
     * 2010): u_k = max(0, ln(n/2 + 1) - ln k) / sum_j max(0, ln(n/2 + 1) - ln j) - 1/n, k = 1..n.
     * They decrease with the rank and sum to zero.
     */
    static void utilities_into(const size_t n, Vector &out);

    //! The weights the mean is recombined with for n evaluated points: the mu positive weights, or
    //! with UTILITY the utilities of all n points
    void mean_weights_into(const size_t n, Vector &out) const;

    Vector clipped() const;

    /**
     * @brief The weights for a population of n evaluated individuals, mu < n <= lambda (when
     * sequential selection stopped a generation early). The positive weights are unchanged;
     * the negative weights of ranks mu..n-1 are scaled so that their sum equals the sum of all
     * lambda - mu negative weights. For n >= lambda these are the weights themselves.
     */
    Vector for_population(const size_t n) const;

    //! for_population into out (no allocation once out has the capacity)
    void for_population_into(const size_t n, Vector &out) const;

    void init_weights(const Settings &settings);

    void init_vectors(const size_t mu, const size_t lambda, const Settings &settings);

    void init(const size_t mu, const size_t lambda, const Settings &settings);
};

} // namespace parameters