#include "weights.hpp"



#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <cassert>

using Eigen::VectorXd;
using std::log;
using std::sqrt;
using std::max;
using std::min;

namespace parameters
{
	static Float get_default_cs(const Settings& settings, const Float mueff, const Float d)
	{
		// TODO: check whether the value of cs needs to be increased when CMA is NONE
		
		if (settings.modules.matrix_adaptation == MatrixAdaptationType::NATURAL_GRADIENT)
			return (9.0 + 3.0 + std::log(d)) / (5.0 * d * std::sqrt(d));
		
		switch (settings.modules.ssa)
		{
			case StepSizeAdaptation::XNES:
				//return mueff / (2.0 * std::log(std::max(Float{ 2. }, d)) * sqrt(d));
				return 0.01;
			case StepSizeAdaptation::MXNES:
				return 1.0;
			case StepSizeAdaptation::LPXNES:
				return (9.0 * mueff) / (10.0 * sqrt(d));
			case StepSizeAdaptation::PSR:
				return 0.9;
			case StepSizeAdaptation::SR:
				return 2.0 / (d + 2.0);
			case StepSizeAdaptation::CSA:
				//return (mueff + 2.0) / (d + mueff + 5.0);
				return (mueff + 2) / (d + (mueff + 3.0));
			default:
				return 0.3;
		}
	}

	
	static Float get_default_damps(const Settings& settings, const Float mueff, const Float d, const Float cs)
	{
		switch (settings.modules.ssa)
		{
			case StepSizeAdaptation::SR:
				return 1.0 + (d / 2.0);
			case StepSizeAdaptation::CSA:
			{
				const Float rhs = std::sqrt((mueff - Float(1.0)) / (d + 1)) - 1;
				return 1.0 + (2.0 * std::max(Float(0.0), rhs) + cs);
			}
			default:
				return 1.0;
		}
	}


	static Float get_default_c1(const Settings& settings, const Float d, const Float mueff, const Float acov)
	{
		if (settings.one_plus_one)
			return  2.0 / (pow(d, 2) + 6.0);

		return acov / (pow(d + 1.3, 2) + mueff);
	}

	static Float get_default_cc(const Settings& settings, const Float d, const Float mueff, const Float cs)
	{
		if (settings.modules.matrix_adaptation == MatrixAdaptationType::NATURAL_GRADIENT)
			return !settings.one_plus_one ? 0.5 * cs : (1.0 / (4.0 * pow(d, 1.5)));
		
		
		if (settings.one_plus_one)
			return 2.0 / (d + 2.0);
		
		return (4.0 + (mueff / d)) / (d + 4.0 + (2.0 * mueff / d));
	}

	static Float get_default_cmu(
		const Settings& settings, 
		const Float d, 
		const Float mueff, 
		const Float c1, 
		const Float acov
	)
	{
		Float cmu_default = std::min(1.0 - c1, 
			acov *
				(mueff - 2.0 + (1.0 / mueff)) 
					/ (pow(d + 2.0, 2) + (acov * mueff / 2)));

		//Float cmu_default = std::min(1.0 - c1,
		//	acov * 
		//	(0.25 + mueff + 1.0 / mueff - 2.0) / 
		//	(pow(d + 2., 2.0) + acov * mueff / 2.0));
			
		if (settings.modules.matrix_adaptation == MatrixAdaptationType::SEPARABLE)
			cmu_default *= ((d + 2.0) / 3.0);

		if (settings.lambda0 == 1)
		{
			cmu_default = 2 / (pow(d, 2) + 6.0);
		}
		return cmu_default;
	}

	Weights::Weights(
		const size_t dim,
		const size_t mu,
		const size_t lambda,
		const Settings& settings,
		const Float expected_length_z
	)
		: weights(lambda), positive(mu), negative(lambda - mu), expected_length_z(expected_length_z)
	{
		const Float d = static_cast<Float>(dim);
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
		}

		positive /= positive.sum();
		mueff = 1.0 / positive.dot(positive);
		mueff_neg = std::pow(negative.sum(), 2) / negative.dot(negative);

		acov = settings.acov.value_or(2.0);
		c1 = settings.c1.value_or(get_default_c1(settings, d, mueff, acov));
		cmu = settings.cmu.value_or(get_default_cmu(settings, d, mueff, c1, acov));
		cs = settings.cs.value_or(get_default_cs(settings, mueff, d));
		cc = settings.cc.value_or(get_default_cc(settings, d, mueff, cs));
		
		damps = settings.damps.value_or(get_default_damps(settings, mueff, d, cs));

		sqrt_cs_mueff = std::sqrt(cs * (2.0 - cs) * mueff);
		sqrt_cc_mueff = std::sqrt(cc * (2.0 - cc) * mueff);
			
		//const Float amu_neg = 1.0 + (c1 / cmu);
		const Float amu_neg = 1.0 + (c1 / static_cast<Float>(mu));
		const Float amueff_neg = 1.0 + ((2.0 * mueff_neg) / (mueff + 2.0));
		const Float aposdef_neg = (1.0 - c1 - cmu) / (d * cmu);
		const Float neg_scaler = std::min(amu_neg, std::min(amueff_neg, aposdef_neg));
		negative *= (neg_scaler / negative.cwiseAbs().sum());
		weights << positive, negative;		


		lazy_update_interval = 1.0 / (c1 + cmu + 1e-23) / d / 10.0;
		expected_length_ps = (1.4 + (2.0 / (d + 1.0))) * expected_length_z;

		beta = 1.0 / std::sqrt(2.0 * mueff);
		if (settings.modules.ssa == StepSizeAdaptation::LPXNES)
			beta = std::log(2.0) / (std::sqrt(d) * std::log(d));
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

		for (auto i = 0; i < positive.size(); ++i)
			positive(i) = dmu / pow(2.0, static_cast<Float>(i + 1)) + base;

		for (auto i = 1; i < (negative.size() + 1); ++i)
			negative(negative.size() - i) = (1.0 / pow(2.0, static_cast<Float>(i)) + base2) * -1.0;
	}


	Vector Weights::clipped() const
	{
		return (weights.array() > 0).select(weights, Vector::Zero(weights.size()));
	}
}
