#pragma once

#include "mutation.hpp"
#include "population.hpp"
#include "settings.hpp"
#include "stats.hpp"
#include "weights.hpp"

namespace matrix_adaptation
{
	struct Adaptation
	{
		Vector m, m_old, dm, ps;
		Float dd;
		Float expected_length_z;
		Matrix inv_C;

		Adaptation(const size_t dim, const Vector& x0, const Vector& ps, const Float expected_length_z) :
			m(x0), m_old(dim), dm(Vector::Zero(dim)),
			ps(ps), dd(static_cast<Float>(dim)),
			expected_length_z(expected_length_z),
			inv_C(Matrix::Identity(dim, dim))
		{
		}

		virtual void adapt_evolution_paths(const Population& pop, const parameters::Weights& w,
			const std::shared_ptr<mutation::Strategy>& mutation,
			const parameters::Stats& stats, size_t mu, size_t lambda) = 0;

		virtual bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			size_t mu, const parameters::Settings& settings, const parameters::Stats& stats) = 0;

		virtual void restart(const parameters::Settings& settings) = 0;

		virtual Vector compute_y(const Vector&) = 0;

		virtual Vector invert_x(const Vector&, Float sigma);

		virtual Vector invert_y(const Vector&) = 0;

	};

	struct None final : Adaptation
	{
		None(const size_t dim, const Vector& x0, const Float expected_length_z) : Adaptation(dim, x0, Vector::Ones(dim), expected_length_z)
		{
		}

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			const size_t mu, const parameters::Settings& settings, const parameters::Stats& stats) override
		{
			return true;
		}

		void adapt_evolution_paths(const Population& pop, const parameters::Weights& w,
			const std::shared_ptr<mutation::Strategy>& mutation, const parameters::Stats& stats,
			size_t mu, size_t lambda) override;

		void restart(const parameters::Settings& settings) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};

	struct CovarianceAdaptation : Adaptation
	{
		Vector pc, d;
		Matrix B, C;
		Matrix inv_root_C;

		bool hs = true;

		CovarianceAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z) : Adaptation(dim, x0, Vector::Zero(dim), expected_length_z),
			pc(Vector::Zero(dim)),
			d(Vector::Ones(dim)),
			B(Matrix::Identity(dim, dim)),
			C(Matrix::Identity(dim, dim)),
			inv_root_C(Matrix::Identity(dim, dim))
		{
		}

		void adapt_covariance_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			size_t mu);

		virtual bool perform_eigendecomposition(const parameters::Settings& settings);

		void adapt_evolution_paths(const Population& pop, const parameters::Weights& w,
			const std::shared_ptr<mutation::Strategy>& mutation, const parameters::Stats& stats,
			size_t mu, size_t lambda) override;

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, const parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};

	struct SeperableAdaptation : CovarianceAdaptation
	{
		using CovarianceAdaptation::CovarianceAdaptation;

		bool perform_eigendecomposition(const parameters::Settings& settings) override;
	};


	struct OnePlusOneAdaptation: CovarianceAdaptation
	{
		constexpr static Float max_success_ratio = 0.44;

		using CovarianceAdaptation::CovarianceAdaptation;

		void adapt_evolution_paths(const Population& pop, const parameters::Weights& w,
			const std::shared_ptr<mutation::Strategy>& mutation, const parameters::Stats& stats,
			size_t mu, size_t lambda) override;

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, const parameters::Stats& stats) override;

	};


	struct MatrixAdaptation final : Adaptation
	{
		Matrix M;
		Matrix M_inv;

		MatrixAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z) : Adaptation(dim, x0, Vector::Ones(dim), expected_length_z),
			M(Matrix::Identity(dim, dim)),
			M_inv(Matrix::Identity(dim, dim))
		{
		}

		void adapt_evolution_paths(const Population& pop, const parameters::Weights& w,
			const std::shared_ptr<mutation::Strategy>& mutation, const parameters::Stats& stats,
			size_t mu, size_t lambda) override;

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, const parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};

	inline std::shared_ptr<Adaptation> get(const parameters::Modules& m, const size_t dim, const Vector& x0, const Float expected_z)
	{
		using namespace parameters;
		switch (m.matrix_adaptation)
		{
		case MatrixAdaptationType::MATRIX:
			return std::make_shared<MatrixAdaptation>(dim, x0, expected_z);
		case MatrixAdaptationType::NONE:
			return std::make_shared<None>(dim, x0, expected_z);
		case MatrixAdaptationType::SEPERABLE:
			return std::make_shared<SeperableAdaptation>(dim, x0, expected_z);
		case MatrixAdaptationType::ONEPLUSONE:
			return std::make_shared<OnePlusOneAdaptation>(dim, x0, expected_z);
		default:
		case MatrixAdaptationType::COVARIANCE:
			return std::make_shared<CovarianceAdaptation>(dim, x0, expected_z);
		}
	}
} // namespace parameters
