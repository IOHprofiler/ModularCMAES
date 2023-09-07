#pragma once

#include "common.hpp"
#include "sampling.hpp"

struct Population;

namespace bounds
{

	struct BoundCorrection
	{
		Vector lb, ub, db;
		double diameter;
		size_t n_out_of_bounds = 0;

		BoundCorrection(const Vector &lb, const Vector &ub) : lb(lb), ub(ub), db(ub - lb),
															  diameter((ub - lb).norm()) {}

		virtual void correct(Population &pop, const Vector &m) = 0;
	};

	struct NoCorrection : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		void correct(Population &pop, const Vector &m) override {}
	};

	struct CountOutOfBounds : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		void correct(Population &pop, const Vector &m) override;
	};

	struct COTN : BoundCorrection
	{
		sampling::Gaussian sampler;

		COTN(Eigen::Ref<const Vector> lb, Eigen::Ref<const Vector> ub) : BoundCorrection(lb, ub), sampler(static_cast<size_t>(lb.size()), rng::normal<double>(0, 1.0 / 3.)) {}

		void correct(Population &pop, const Vector &m) override;
	};

	struct Mirror : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		void correct(Population &pop, const Vector &m) override;
	};

	struct UniformResample : BoundCorrection
	{
		sampling::Random<std::uniform_real_distribution<>> sampler;

		UniformResample(Eigen::Ref<const Vector> lb, Eigen::Ref<const Vector> ub) : BoundCorrection(lb, ub), sampler(static_cast<size_t>(lb.size())) {}

		void correct(Population &pop, const Vector &m) override;
	};

	struct Saturate : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		void correct(Population &pop, const Vector &m) override;
	};

	struct Toroidal : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		void correct(Population &pop, const Vector &m) override;
	};

	enum class CorrectionMethod
	{
		NONE,
		COUNT,
		MIRROR,
		COTN,
		UNIFORM_RESAMPLE,
		SATURATE,
		TOROIDAL
	};

	inline std::shared_ptr<BoundCorrection> get(const CorrectionMethod &m, const Vector &lb, const Vector &ub)
	{
		switch (m)
		{
		case CorrectionMethod::COUNT:
			return std::make_shared<CountOutOfBounds>(lb, ub);
		case CorrectionMethod::MIRROR:
			return std::make_shared<Mirror>(lb, ub);
		case CorrectionMethod::COTN:
			return std::make_shared<COTN>(lb, ub);
		case CorrectionMethod::UNIFORM_RESAMPLE:
			return std::make_shared<UniformResample>(lb, ub);
		case CorrectionMethod::SATURATE:
			return std::make_shared<Saturate>(lb, ub);
		case CorrectionMethod::TOROIDAL:
			return std::make_shared<Toroidal>(lb, ub);

		default:
		case CorrectionMethod::NONE:
			return std::make_shared<NoCorrection>(lb, ub);
		}
	};
}
