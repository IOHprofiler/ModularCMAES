#pragma once

#include "common.hpp"
#include "sampling.hpp"
#include "modules.hpp"

struct Population;

namespace parameters
{
	struct Parameters;
	struct Settings;
}

namespace bounds
{
	Mask is_out_of_bounds(const Vector &xi, const Vector &lb, const Vector &ub);
	bool any_out_of_bounds(const Vector &xi, const Vector &lb, const Vector &ub);

	struct BoundCorrection
	{
		size_t n_out_of_bounds = 0;

		virtual ~BoundCorrection() = default;
		BoundCorrection() = default; 

		void correct(const Eigen::Index i, parameters::Parameters &p);

		virtual Vector correct_x(
			const Vector &xi,
			const Mask &oob,
			const Float sigma,
			const parameters::Settings &settings) = 0;

		[[nodiscard]] Mask is_out_of_bounds(
			const Vector &xi,
			const parameters::Settings &settings) const;

		[[nodiscard]] Vector delta_out_of_bounds(
			const Vector &xi,
			const Mask &oob,
			const parameters::Settings &settings) const;

		[[nodiscard]] bool any_out_of_bounds() const
		{
			return n_out_of_bounds > 0;
		}
	};

	struct NoCorrection : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		Vector correct_x(const Vector &xi, const Mask &oob, const Float sigma, const parameters::Settings &settings) override
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

		COTN(const size_t d) : BoundCorrection(), sampler(d, rng::normal<Float>(0, 1.0 / 3.)) {}

		Vector correct_x(const Vector &xi, const Mask &oob, const Float sigma, const parameters::Settings &settings) override;
	};

	struct Mirror final : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		Vector correct_x(const Vector &xi, const Mask &oob, const Float sigma, const parameters::Settings &settings) override;
	};

	struct UniformResample final : BoundCorrection
	{
		sampling::Uniform sampler;

		UniformResample(const size_t d) : BoundCorrection(), sampler(d) {}

		Vector correct_x(const Vector &xi, const Mask &oob, const Float sigma, const parameters::Settings &settings) override;
	};

	struct Saturate final : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		Vector correct_x(const Vector &xi, const Mask &oob, const Float sigma, const parameters::Settings &settings) override;
	};

	struct Toroidal final : BoundCorrection
	{
		using BoundCorrection::BoundCorrection;

		Vector correct_x(const Vector &xi, const Mask &oob, const Float sigma, const parameters::Settings &settings) override;
	};

	inline std::shared_ptr<BoundCorrection> get(const parameters::CorrectionMethod &m, const size_t d)
	{
		using namespace parameters;
		switch (m)
		{
		case CorrectionMethod::MIRROR:
			return std::make_shared<Mirror>();
		case CorrectionMethod::COTN:
			return std::make_shared<COTN>(d);
		case CorrectionMethod::UNIFORM_RESAMPLE:
			return std::make_shared<UniformResample>(d);
		case CorrectionMethod::SATURATE:
			return std::make_shared<Saturate>();
		case CorrectionMethod::TOROIDAL:
			return std::make_shared<Toroidal>();
		case CorrectionMethod::RESAMPLE:
			return std::make_shared<Resample>();
		default:
		case CorrectionMethod::NONE:
			return std::make_shared<NoCorrection>();
		}
	};
}
