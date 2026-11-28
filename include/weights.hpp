#pragma once

#include "settings.hpp"

namespace parameters
{
struct Weights
{
    Vector weights;
    Vector positive;
    Vector negative;
    //! The unscaled negative weights that the scaling of `negative` and mueff_neg are computed
    //! from: `negative` before the scaling, except with DEFAULT weights and lambda < 2 mu: the mu
    //! negative weights of a population of 2 mu, of which negative has the last lambda - mu (as
    //! pycma, #73)
    Vector negative_reference;
    Float mueff, mueff_neg;
    //! The variance effective selection mass of the weights the mean is recombined with,
    //! 1 / sum(w_i^2): mueff, except for UTILITY, where the mean uses all lambda utilities
    Float mueff_mean;
    //! RecombinationWeights::UTILITY: the utilities of NES/xNES, which sum to zero, recombine the
    //! mean from all evaluated points and are the weights of every matrix update
    bool utility = false;
    //! Mirror::PAIRWISE: the losers of the mirrored pairs get no negative weights
    bool pairwise = false;
    Float c1, cmu, cc, cs;
    //! The learning rate of the path p_sigma: min(cs, 1). LPXNES's cs exceeds 1 for large mueff,
    //! and as the rate of the path made it NaN (#72)
    Float cs_path;
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

    /**
     * @brief The weights for a population of n evaluated individuals, mu < n <= lambda (when
     * sequential selection stopped a generation early). The positive weights are unchanged;
     * the negative weights of ranks mu..n-1 are scaled so that their sum equals the sum of all
     * lambda - mu negative weights. For n >= lambda these are the weights themselves.
     */
    Vector for_population(const size_t n) const;

    /**
     * @brief for_population into out (no allocation once out has the capacity). With pairwise
     * selection only the first n_ranked points (the winners of the mirrored pairs and, with
     * elitism, the parents; the losers keep their real f and are sorted after them,
     * selection::Pairwise, #75) can get a negative weight: as pycma, mirrored points that lost
     * their pair get no negative weights.
     */
    void for_population_into(const size_t n,
                             Vector &out,
                             const size_t n_ranked = std::numeric_limits<size_t>::max()) const;

    void init_weights(const Settings &settings);

    void init_vectors(const size_t mu, const size_t lambda, const Settings &settings);

    void init(const size_t mu, const size_t lambda, const Settings &settings);
};

} // namespace parameters