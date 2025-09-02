#pragma once

#include "common.hpp"
#include "sampling.hpp"
#include "modules.hpp"

struct Population;

namespace parameters
{
	struct Parameters;
}

namespace bounds
{
	using Mask = Eigen::Array<bool, Eigen::Dynamic, 1>;

	Mask is_out_of_bounds(const Vector& xi, const Vector& lb, const Vector& ub);
	bool any_out_of_bounds(const Vector& xi, const Vector& lb, const Vector& ub);

	struct BoundCorrection
	{
		virtual ~BoundCorrection() = default;
		Vector lb, ub, db;
		Float diameter;
		size_t n_out_of_bounds = 0;
		bool has_bounds;

		BoundCorrection(const Vector& lb, const Vector& ub) : lb(lb), ub(ub), db(ub - lb),
			diameter((ub - lb).norm()),
			has_bounds(true)
		{
			//! find a better way
			if (!std::isfinite(diameter))
			{
				diameter = 10;
				has_bounds = false;
			}
		}

		void correct(const Eigen::Index i, parameters::Parameters& p);

		virtual Vector correct_x(const Vector& xi, const Mask& oob, const Float sigma) = 0;

		[[nodiscard]] Mask is_out_of_bounds(const Vector& xi) const;

		[[nodiscard]] Vector delta_out_of_bounds(const Vector& xi, const Mask& oob) const;

		[[nodiscard]] bool any_out_of_bounds() const
		{
			return n_out_of_bounds > 0;
		}
	};

	struct NoCorrection : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		Vector correct_x(const Vector& xi, const Mask& oob, const Float sigma) override
		{
			return xi;
		}
	};

	struct Resample final : NoCorrection
	{
		using NoCorrection::NoCorrection;
	};

	struct COTN final : BoundCorrection
	{
		sampling::Gaussian sampler;

		COTN(Eigen::Ref<const Vector> lb, Eigen::Ref<const Vector> ub) : BoundCorrection(lb, ub), sampler(static_cast<size_t>(lb.size()), rng::normal<Float>(0, 1.0 / 3.)) {}

		Vector correct_x(const Vector& xi, const Mask& oob, const Float sigma) override;
	};

	struct Mirror final : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		Vector correct_x(const Vector& xi, const Mask& oob, const Float sigma) override;
	};

	struct UniformResample final : BoundCorrection
	{
		sampling::Uniform sampler;

		UniformResample(Eigen::Ref<const Vector> lb, Eigen::Ref<const Vector> ub) : BoundCorrection(lb, ub), sampler(static_cast<size_t>(lb.size())) {}

		Vector correct_x(const Vector& xi, const Mask& oob, const Float sigma) override;
	};

	struct Saturate final : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		Vector correct_x(const Vector& xi, const Mask& oob, const Float sigma) override;
	};

	struct Toroidal final : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		Vector correct_x(const Vector& xi, const Mask& oob, const Float sigma) override;
	};

	inline std::shared_ptr<BoundCorrection> get(const parameters::CorrectionMethod& m, const Vector& lb, const Vector& ub)
	{
		using namespace parameters;
		switch (m)
		{
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
		case CorrectionMethod::RESAMPLE:
			return std::make_shared<Resample>(lb, ub);
		default:
		case CorrectionMethod::NONE:
			return std::make_shared<NoCorrection>(lb, ub);
		}
	};
}
