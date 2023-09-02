#include "parameters.hpp"

namespace parameters
{

    Weights::Weights(const size_t dim, const size_t mu, const size_t lambda, const Settings &settings)
        : weights(lambda), positive(mu), negative(lambda - mu)
    {
        const double d = static_cast<double>(dim);
        using namespace mutation;
        switch (settings.modules.weights)
        {
        case RecombinationWeights::EQUAL:
            weights_equal(mu);
            break;
        case RecombinationWeights::HALF_POWER_LAMBDA:
            weights_half_power_lambda(mu, lambda);
            break;
        case RecombinationWeights::DEFAULT:
            weights_default(lambda);
            break;
        }

        mueff = std::pow(positive.sum(), 2) / positive.dot(positive);
        mueff_neg = std::pow(negative.sum(), 2) / negative.dot(negative);
        positive /= positive.sum();

        c1 = settings.c1.value_or(2.0 / (pow(d + 1.3, 2) + mueff));
        cmu = settings.cmu.value_or(
            std::min(1.0 - c1, 2.0 * ((mueff - 2.0 + (1.0 / mueff)) / (pow(d + 2.0, 2) + (2.0 * mueff / 2))))
        );
        cc = settings.cmu.value_or(
            (4.0 + (mueff / d)) / (d + 4.0 + (2.0 * mueff / d))
        );
        
        const double amu_neg = 1.0 + (c1 / static_cast<double>(mu));
        const double amueff_neg = 1.0 + ((2.0 * mueff_neg) / (mueff + 2.0));
        const double aposdef_neg = (1.0 - c1 - cmu) / (d * cmu);

        const double neg_scaler = std::min(amu_neg, std::min(amueff_neg, aposdef_neg));
        negative *= neg_scaler / negative.cwiseAbs().sum();
        weights << positive, negative;
    }

    void Weights::weights_default(const size_t lambda)
    {
        const double base = std::log((static_cast<double>(lambda) + 1.) / 2.0);
        for (auto i = 0; i < positive.size(); ++i)
            positive(i) = base - std::log(static_cast<double>(i + 1));

        for (auto i = 0; i < negative.size(); ++i)
            negative(i) = base - std::log(static_cast<double>(i + 1 + positive.size()));
    }

    void Weights::weights_equal(const size_t mu)
    {
        const double wi = 1. / static_cast<double>(mu);
        positive.setConstant(wi);
        negative.setConstant(-wi);
    }

    void Weights::weights_half_power_lambda(const size_t mu, const size_t lambda)
    {
        const double dmu = static_cast<double>(mu);
        const double base = (1.0 / pow(2.0, dmu)) / dmu;
        const double delta = static_cast<double>(lambda - mu);
        const double base2 = (1.0 / pow(2.0, delta)) / delta;

        for (auto i = 0; i < positive.size(); ++i)
            positive(i) = dmu / pow(2.0, static_cast<double>(i + 1)) + base;

        for (auto i = 0; i < negative.size(); ++i)
            negative(negative.size() - i) = 1.0 / pow(2.0, static_cast<double>(i + 1)) + base2;
    }


    Vector Weights::clipped() const {
        return (weights.array() > 0).select(weights, Vector::Zero(weights.size()));
    }

}