#include "weights.hpp"

#include <Eigen/Dense>
#include <cmath>

using std::log;
using std::max;
using std::min;
using std::sqrt;

namespace parameters
{
//! xNES's learning rate of sigma and of the shape, eta_sigma = eta_B = 3 (3 + ln d) / (5 d sqrt(d))
//! (Glasmachers et al., Exponential Natural Evolution Strategies, GECCO 2010, Table 1)
static Float xnes_learning_rate(const Float d)
{
    return (9.0 + 3.0 * std::log(d)) / (5.0 * d * std::sqrt(d));
}


static Float get_default_cs(const Settings &settings, const Float mueff, const Float d)
{
    // TODO: check whether the value of cs needs to be increased when CMA is NONE

    switch (settings.modules.ssa)
    {
    case StepSizeAdaptation::XNES:
        // With NATURAL_GRADIENT the rule is xNES's own, sigma *= exp(eta_sigma / 2 tr(G) / d),
        // with xNES's eta_sigma (Glasmachers et al. 2010, Table 1). Only XNES gets it: the other
        // rules keep their own rates with NATURAL_GRADIENT (#72, F3)
        if (settings.modules.matrix_adaptation == MatrixAdaptationType::NATURAL_GRADIENT)
            return xnes_learning_rate(d);
        // Otherwise the xNES SA of Krause, Glasmachers & Igel, Qualitative and Quantitative
        // Assessment of Step Size Adaptation Rules, FOGA 2017, Sec. 4.2 and Table 3: sigma *=
        // exp(c_sigma / sqrt(d) sum_i w_i (||z_i||^2 - d)) with c_sigma = mueff / (2 ln(d)
        // sqrt(d)). ln(max(2, d)) keeps it finite at d = 1, as 1.x (branch legacy-python,
        // modcma/parameters.py). It was the constant 0.01 (#72, F27)
        return mueff / (2.0 * std::log(std::max(Float{2.0}, d)) * std::sqrt(d));
    case StepSizeAdaptation::MXNES:
        // mean-xNES SA: c_sigma = 1 (Krause et al. 2017, Sec. 4.3, Table 3), also with
        // NATURAL_GRADIENT, which gave it xNES's eta_sigma (#72, F3)
        return 1.0;
    case StepSizeAdaptation::LPXNES:
        // prior-xNES SA: c_sigma = 9 mueff / (10 sqrt(d)) (Krause et al. 2017, Sec. 4.4, Table 3)
        return (9.0 * mueff) / (10.0 * sqrt(d));
    case StepSizeAdaptation::PSR:
        // Loshchilov (2014), Alg. 6 (the Python implementation used 0.4, the C++ one 0.9)
        return 0.3;
    case StepSizeAdaptation::SR:
        return 2.0 / (d + 2.0);
    case StepSizeAdaptation::CSA:
        // c_sigma = (mueff + 2) / (n + mueff + 5), without a cap: Hansen, The CMA Evolution
        // Strategy: A Tutorial, arXiv:1604.00772v2 (2023), Table 1, Eq. (55), and its MATLAB code
        // (line 30); `cmaes` 0.13 has the same. pycma 4.6 uses n + mueff + 3 and caps cs at 1/2
        // (sigma_adaptation.py:235, 422); modcma follows the tutorial (#138)
        return (mueff + 2.0) / (d + mueff + 5.0);
    default:
        return 0.3;
    }
}

static Float
get_default_damps(const Settings &settings, const Float mueff, const Float d, const Float cs)
{
    switch (settings.modules.ssa)
    {
    case StepSizeAdaptation::SR:
        return 1.0 + (d / 2.0);
    case StepSizeAdaptation::MSR:
        // 2 (d - 1) / d (Ait ElHara et al. 2013, pycma); 1 for d = 1, where it would be 0
        return std::max(Float(1.0), 2.0 - (2.0 / d));
    case StepSizeAdaptation::PSR:
        // d_sigma = 1 (Loshchilov 2014, Alg. 6)
        return 1.0;
    case StepSizeAdaptation::CSA: {
        // d_sigma = 1 + 2 max(0, sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma (Hansen,
        // arXiv:1604.00772v2, Table 1, Eq. (55))
        const Float rhs = std::sqrt((mueff - Float(1.0)) / (d + 1)) - 1;
        return 1.0 + (2.0 * std::max(Float(0.0), rhs) + cs);
    }
    default:
        return 1.0;
    }
}

static Float
get_default_c1(const Settings &settings, const Float d, const Float mueff, const Float acov)
{
    if (settings.one_plus_one)
        return 2.0 / (pow(d, 2) + 6.0);

    // SEPARABLE has this c1 too, without the (d + 2) / 3 of its cmu: an open decision of #73 (Ros &
    // Hansen 2008 or pycma's CMA_diagonal, which has its own rates c1_sep and cmu_sep)
    return acov / (pow(d + 1.3, 2) + mueff);
}

static Float get_default_cc(const Settings &settings, const Float d, const Float mueff)
{
    // NATURAL_GRADIENT: cc is the rate t of A <- A expm(t G), xNES's eta_B / 2 (Glasmachers et al.
    // 2010, Table 1: B <- B expm(eta_B / 2 G_B)) with every step-size rule. It was cs / 2, the same
    // value while NATURAL_GRADIENT gave every rule cs = eta_sigma = eta_B (#72, F3)
    if (settings.modules.matrix_adaptation == MatrixAdaptationType::NATURAL_GRADIENT)
        return !settings.one_plus_one ? 0.5 * xnes_learning_rate(d) : (1.0 / (4.0 * pow(d, 1.5)));

    if (settings.one_plus_one)
        return 2.0 / (d + 2.0);

    return (4.0 + (mueff / d)) / (d + 4.0 + (2.0 * mueff / d));
}

static Float get_default_cmu(const Settings &settings,
                             const Float d,
                             const Float mueff,
                             const Float c1,
                             const Float acov)
{
    Float cmu_default =
        acov * (mueff - 2.0 + (1.0 / mueff)) / (pow(d + 2.0, 2) + (acov * mueff / 2));

    // SEPARABLE: multiplied by (d + 2) / 3 (Ros & Hansen 2008) before the cap c1 + cmu <= 1. Capped
    // first, cmu exceeded 1 with large populations (IPOP), and the decay of C became negative
    // (#73, F4). pycma also caps its separable rate last (options_parameters.py:1100)
    if (settings.modules.matrix_adaptation == MatrixAdaptationType::SEPARABLE)
        cmu_default *= ((d + 2.0) / 3.0);
    cmu_default = std::min(1.0 - c1, cmu_default);

    if (settings.lambda0 == 1)
    {
        cmu_default = 2 / (pow(d, 2) + 6.0);
    }
    return cmu_default;
}

Weights::Weights(const Settings &settings, const Float expected_length_z) :
    expected_length_z(expected_length_z)
{
    init(settings.mu0, settings.lambda0, settings);
}

void Weights::init(const size_t mu, const size_t lambda, const Settings &settings)
{
    init_vectors(mu, lambda, settings);
    init_weights(settings);
}

void Weights::init_vectors(const size_t mu, const size_t lambda, const Settings &settings)
{
    weights.resize(lambda);
    positive.resize(mu);
    negative.resize(lambda - mu);
    negative_reference.resize(0);

    switch (settings.modules.weights)
    {
    case RecombinationWeights::EQUAL:
        weights_equal(mu);
        break;
    case RecombinationWeights::EXPONENTIAL:
        weights_exponential(mu, lambda);
        break;
    case RecombinationWeights::DEFAULT:
        weights_default(mu, lambda);
        break;
    case RecombinationWeights::UTILITY:
        // weights: the utilities themselves; positive: their non-negative part among the mu best
        // (normalized in init_weights), for the modules that need convex weights (SA, and XNES
        // without NATURAL_GRADIENT and LPXNES, whose source uses convex weights: Krause et al. 2017)
        utilities_into(lambda, weights);
        positive = weights.head(mu).cwiseMax(0.0);
        negative = weights.tail(lambda - mu);
        break;
    }
    if (negative_reference.size() == 0)
        negative_reference = negative;
    utility = settings.modules.weights == RecombinationWeights::UTILITY;
    pairwise = settings.modules.mirrored == Mirror::PAIRWISE;
}

void Weights::utilities_into(const size_t n, Vector &out)
{
    out.resize(static_cast<Eigen::Index>(n));
    const Float base = std::log1p(static_cast<Float>(n) / 2.0);
    for (Eigen::Index k = 0; k < out.size(); ++k)
        out(k) = std::max(Float{0.0}, base - std::log(static_cast<Float>(k + 1)));
    out /= out.sum();
    out.array() -= 1.0 / static_cast<Float>(n);
}

void Weights::mean_weights_into(const size_t n, Vector &out) const
{
    if (utility)
        utilities_into(n, out);
    else
        out = positive;
}

void Weights::init_weights(const Settings &settings)
{
    const Float d = static_cast<Float>(settings.dim);
    positive /= positive.sum();
    mueff = 1.0 / positive.dot(positive);
    mueff_neg = std::pow(negative_reference.sum(), 2) / negative_reference.dot(negative_reference);
    // The mean of UTILITY is recombined from all lambda utilities (which sum to zero): the paths are
    // normalized with their selection mass, so that they stay N(0, I) under random selection
    mueff_mean = utility ? 1.0 / weights.dot(weights) : mueff;

    acov = settings.acov.value_or(2.0);
    c1 = settings.c1.value_or(get_default_c1(settings, d, mueff, acov));
    cmu = settings.cmu.value_or(get_default_cmu(settings, d, mueff, c1, acov));
    cs = settings.cs.value_or(get_default_cs(settings, mueff, d));
    cc = settings.cc.value_or(get_default_cc(settings, d, mueff));

    damps = settings.damps.value_or(get_default_damps(settings, mueff, d, cs));

    // The rate of the path p_sigma: cs, at most 1. The cs of LPXNES, 9 mueff / (10 sqrt(d)),
    // exceeds 2 for mueff > 2.22 sqrt(d) (and XNES's for large populations): as the rate of the
    // path it made (1 - cs) negative and sqrt(cs (2 - cs)) NaN (#72). At 1 the path is the current
    // step, as with MXNES's cs = 1. (pycma gives the path of h_sigma CSA's rate with every
    // step-size rule, CMAAdaptSigmaBase.initialize_base, sigma_adaptation.py:51; with MATRIX, whose
    // rank-one update reads the path, that was worse on BBOB with MXNES: an open decision of #72)
    cs_path = std::min(cs, Float{1.0});
    sqrt_cs_mueff = std::sqrt(cs_path * (2.0 - cs_path) * mueff_mean);
    sqrt_cc_mueff = std::sqrt(cc * (2.0 - cc) * mueff_mean);

    // The sum of the absolute negative weights: min(1 + c1 / cmu, 1 + 2 mueff^- / (mueff + 2),
    // (1 - c1 - cmu) / (d cmu)) (Hansen, The CMA Evolution Strategy: A Tutorial, 2016, Eq. (51) to
    // (53); pycma's RecombinationWeights.finalize_negative_weights). At least 0, so that the
    // negative weights stay non-positive when c1 + cmu > 1 (given learning rates)
    const Float amu_neg = 1.0 + (c1 / cmu);
    const Float amueff_neg = 1.0 + ((2.0 * mueff_neg) / (mueff + 2.0));
    const Float aposdef_neg = (1.0 - c1 - cmu) / (d * cmu);
    const Float neg_scaler =
        std::max(Float{0.0}, std::min(amu_neg, std::min(amueff_neg, aposdef_neg)));
    if (!utility)
    {
        // UTILITY keeps the utilities as they are (they sum to zero, as in NES)
        negative *= (neg_scaler / negative_reference.cwiseAbs().sum());
        weights << positive, negative;
    }

    lazy_update_interval = 1.0 / (c1 + cmu + 1e-23) / d / 10.0;
    expected_length_ps = (1.4 + (2.0 / (d + 1.0))) * expected_length_z;

    beta = 1.0 / std::sqrt(2.0 * mueff);
    // SA is CMSA's operator: tau = 1 / sqrt(2 d) for its one step size per individual (Spettel,
    // Beyer & Hellwig 2018, Table I, after Beyer & Sendhoff 2008; #71). 1 / sqrt(2 mueff), chosen
    // when the step sizes were drawn per coordinate, stays for sample_sigma (no source)
    if (settings.modules.ssa == StepSizeAdaptation::SA)
        beta = 1.0 / std::sqrt(2.0 * d);
    // prior-xNES SA: beta = ln 2 / (sqrt(d) ln d) (Krause et al. 2017, Table 3); ln(max(d, 2))
    // keeps it finite at d = 1, where ln d = 0 made it inf (#72, F16)
    if (settings.modules.ssa == StepSizeAdaptation::LPXNES)
        beta = std::log(2.0) / (std::sqrt(d) * std::log(std::max(d, Float{2.0})));

    int_lb_sigma = std::min(0.2, mueff / d);
}

void Weights::weights_default(const size_t mu, const size_t lambda)
{
    // The raw weights of a population of n (Hansen, The CMA Evolution Strategy: A Tutorial, 2016,
    // Eq. (49); pycma's RecombinationWeights(n)): w'_i = ln((n + 1) / 2) - ln i, positive exactly
    // for i <= n / 2. With the default mu = lambda / 2, n = lambda. With another mu (given, or the
    // ratio of IPOP), n = 2 mu, as pycma with CMA_mu (options_parameters.py:1010-1013): then the
    // first mu are the positive weights, and the other mu the negative ones, for the worst ranks
    // (pycma's RecombinationWeights.__call__, recombination_weights.py:191-199). With lambda > 2 mu
    // the ranks in between get 0; with lambda < 2 mu the worst lambda - mu ranks get the last
    // (most negative) lambda - mu of them, and the scaling of init_weights is that of all mu.
    // Computed from the raw weights of n = lambda, the weights between ranks lambda / 2 and mu
    // were positive in the negative part (#73, F8).
    const bool default_mu = mu == lambda / 2;
    const auto n = static_cast<Float>(default_mu ? lambda : 2 * mu);
    const Float base = std::log((n + 1.) / 2.);
    const auto raw = [base](const Eigen::Index rank) {
        return base - std::log(static_cast<Float>(rank + 1));
    };

    const auto n_pos = positive.size();
    for (Eigen::Index i = 0; i < n_pos; ++i)
        positive(i) = raw(i);

    const auto n_neg = negative.size();
    if (default_mu)
    {
        for (Eigen::Index i = 0; i < n_neg; ++i)
            negative(i) = raw(i + n_pos);
        return;
    }

    // The mu negative weights of the population of 2 mu, for the worst ranks
    negative_reference.resize(n_pos);
    for (Eigen::Index i = 0; i < n_pos; ++i)
        negative_reference(i) = raw(i + n_pos);
    if (n_neg >= n_pos)
    {
        negative.head(n_neg - n_pos).setZero();
        negative.tail(n_pos) = negative_reference;
        negative_reference.resize(0); // the same sum and mueff^- as negative
    }
    else
        negative = negative_reference.tail(n_neg);
}

void Weights::weights_equal(const size_t mu)
{
    const Float wi = 1. / static_cast<Float>(mu);
    positive.setConstant(wi);
    negative.setConstant(-wi);
}

void Weights::weights_exponential(const size_t mu, const size_t lambda)
{
    const Float dmu = static_cast<Float>(mu);
    const Float base = (1.0 / pow(2.0, dmu)) / dmu;
    const Float delta = static_cast<Float>(lambda - mu);
    const Float base2 = (1.0 / pow(2.0, delta)) / delta;

    // w_i = 2^-i + 1 / (mu 2^mu), i = 1..mu, which sum to 1 (de Nobel et al., The Modular CMA-ES,
    // 2026, Sec. 4.2); the negative weights are the same for lambda - mu, reversed, with a minus sign
    for (auto i = 0; i < positive.size(); ++i)
        positive(i) = 1.0 / pow(2.0, static_cast<Float>(i + 1)) + base;

    for (auto i = 1; i < (negative.size() + 1); ++i)
        negative(negative.size() - i) = (1.0 / pow(2.0, static_cast<Float>(i)) + base2) * -1.0;
}

Vector Weights::for_population(const size_t n) const
{
    Vector w;
    for_population_into(n, w);
    return w;
}

void Weights::for_population_into(const size_t n, Vector &out, const size_t n_ranked) const
{
    if (utility)
    {
        // The utilities of the n evaluated points (sequential selection), again summing to zero
        utilities_into(std::min<size_t>(n, static_cast<size_t>(weights.size())), out);
    }
    else
    {
        const auto total = weights.size();
        const auto k = std::min<Eigen::Index>(static_cast<Eigen::Index>(n), total);
        out = weights.head(k);
        const auto mu = positive.size();
        if (k < total && k > mu)
        {
            const Float used = out.tail(k - mu).sum();
            if (used != 0.0)
                out.tail(k - mu) *= negative.sum() / used;
        }
    }
    if (pairwise)
    {
        const auto first = static_cast<Eigen::Index>(
            std::min<size_t>(n_ranked, static_cast<size_t>(out.size())));
        for (Eigen::Index i = first; i < out.size(); ++i)
            out(i) = std::max(out(i), Float(0.0));
    }
}
} // namespace parameters
