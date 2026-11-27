#include "weights.hpp"

#include <Eigen/Dense>
#include <cmath>

using std::log;
using std::max;
using std::min;
using std::sqrt;

namespace parameters
{
static Float get_default_cs(const Settings &settings, const Float mueff, const Float d)
{
    // TODO: check whether the value of cs needs to be increased when CMA is NONE

    if (settings.modules.matrix_adaptation == MatrixAdaptationType::NATURAL_GRADIENT)
        // the learning rate of xNES, eta_sigma = eta_B (Glasmachers et al. 2010)
        return (9.0 + 3.0 * std::log(d)) / (5.0 * d * std::sqrt(d));

    switch (settings.modules.ssa)
    {
    case StepSizeAdaptation::XNES:
        // (the Python implementation used mueff / (2 ln(max(2, d)) sqrt(d)); TASKS.md F27)
        return 0.01;
    case StepSizeAdaptation::MXNES:
        return 1.0;
    case StepSizeAdaptation::LPXNES:
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

    return acov / (pow(d + 1.3, 2) + mueff);
}

static Float
get_default_cc(const Settings &settings, const Float d, const Float mueff, const Float cs)
{
    if (settings.modules.matrix_adaptation == MatrixAdaptationType::NATURAL_GRADIENT)
        return !settings.one_plus_one ? 0.5 * cs : (1.0 / (4.0 * pow(d, 1.5)));

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
    Float cmu_default = std::min(
        1.0 - c1, acov * (mueff - 2.0 + (1.0 / mueff)) / (pow(d + 2.0, 2) + (acov * mueff / 2)));

    if (settings.modules.matrix_adaptation == MatrixAdaptationType::SEPARABLE)
        cmu_default *= ((d + 2.0) / 3.0);

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
        // (normalized in init_weights), for the modules that need convex weights (SA, LPXNES)
        utilities_into(lambda, weights);
        positive = weights.head(mu).cwiseMax(0.0);
        negative = weights.tail(lambda - mu);
        break;
    }
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
    mueff_neg = std::pow(negative.sum(), 2) / negative.dot(negative);
    // The mean of UTILITY is recombined from all lambda utilities (which sum to zero): the paths are
    // normalized with their selection mass, so that they stay N(0, I) under random selection
    mueff_mean = utility ? 1.0 / weights.dot(weights) : mueff;

    acov = settings.acov.value_or(2.0);
    c1 = settings.c1.value_or(get_default_c1(settings, d, mueff, acov));
    cmu = settings.cmu.value_or(get_default_cmu(settings, d, mueff, c1, acov));
    cs = settings.cs.value_or(get_default_cs(settings, mueff, d));
    cc = settings.cc.value_or(get_default_cc(settings, d, mueff, cs));

    damps = settings.damps.value_or(get_default_damps(settings, mueff, d, cs));

    sqrt_cs_mueff = std::sqrt(cs * (2.0 - cs) * mueff_mean);
    sqrt_cc_mueff = std::sqrt(cc * (2.0 - cc) * mueff_mean);

    // alpha_mu^- = 1 + c1 / cmu (Hansen 2016; pycma)
    const Float amu_neg = 1.0 + (c1 / cmu);
    const Float amueff_neg = 1.0 + ((2.0 * mueff_neg) / (mueff + 2.0));
    const Float aposdef_neg = (1.0 - c1 - cmu) / (d * cmu);
    const Float neg_scaler = std::min(amu_neg, std::min(amueff_neg, aposdef_neg));
    if (!utility)
    {
        // UTILITY keeps the utilities as they are (they sum to zero, as in NES)
        negative *= (neg_scaler / negative.cwiseAbs().sum());
        weights << positive, negative;
    }

    lazy_update_interval = 1.0 / (c1 + cmu + 1e-23) / d / 10.0;
    expected_length_ps = (1.4 + (2.0 / (d + 1.0))) * expected_length_z;

    beta = 1.0 / std::sqrt(2.0 * mueff);
    if (settings.modules.ssa == StepSizeAdaptation::LPXNES)
        beta = std::log(2.0) / (std::sqrt(d) * std::log(d));

    int_lb_sigma = std::min(0.2, mueff / d);
}

void Weights::weights_default(const size_t mu, const size_t lambda)
{
    const Float ratio = static_cast<Float>(lambda) / static_cast<Float>(mu);
    const Float base = std::log((static_cast<Float>(lambda) + 1.) / std::floor(ratio));

    for (auto i = 0; i < positive.size(); ++i)
        positive(i) = base - std::log(static_cast<Float>(i + 1));

    for (auto i = 0; i < negative.size(); ++i)
        negative(i) = base - std::log(static_cast<Float>(i + 1 + positive.size()));
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

Vector Weights::clipped() const
{
    return (weights.array() > 0).select(weights, Vector::Zero(weights.size()));
}
} // namespace parameters
