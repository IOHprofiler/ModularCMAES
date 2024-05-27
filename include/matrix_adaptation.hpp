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
		double dd;
		double chiN;
		Matrix inv_C;

		Adaptation(const size_t dim, const Vector& x0, const Vector& ps) :
			m(x0), m_old(dim), dm(Vector::Zero(dim)),
			ps(ps), dd(static_cast<double>(dim)),
			chiN(sqrt(dd)* (1.0 - 1.0 / (4.0 * dd) + 1.0
				/ (21.0 * pow(dd, 2.0)))),
			inv_C(Matrix::Identity(dim, dim))
		{
		}

		virtual void adapt_evolution_paths(const Population& pop, const parameters::Weights& w,
			const std::shared_ptr<mutation::Strategy>& mutation,
			const parameters::Stats& stats, size_t mu, size_t lambda) = 0;

		virtual bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			size_t mu, const parameters::Settings& settings) = 0;

		virtual void restart(const parameters::Settings& settings) = 0;

		virtual Vector compute_y(const Vector&) = 0;

		virtual Vector invert_x(const Vector&, double sigma);

		virtual Vector invert_y(const Vector&) = 0;

	};

	struct None final : Adaptation
	{
		None(const size_t dim, const Vector& x0) : Adaptation(dim, x0, Vector::Ones(dim))
		{
		}

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			const size_t mu, const parameters::Settings& settings) override
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

		CovarianceAdaptation(const size_t dim, const Vector& x0) : Adaptation(dim, x0, Vector::Zero(dim)),
			pc(Vector::Zero(dim)),
			d(Vector::Ones(dim)),
			B(Matrix::Identity(dim, dim)),
			C(Matrix::Identity(dim, dim)),
			inv_root_C(Matrix::Identity(dim, dim))
		{
		}

		virtual bool perform_eigendecomposition(const parameters::Settings& settings);

		void adapt_evolution_paths(const Population& pop, const parameters::Weights& w,
			const std::shared_ptr<mutation::Strategy>& mutation, const parameters::Stats& stats,
			size_t mu, size_t lambda) override;

		void adapt_covariance_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			size_t mu);

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings) override;

		void restart(const parameters::Settings& settings) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};

	struct SeperableAdaptation : CovarianceAdaptation
	{
		using CovarianceAdaptation::CovarianceAdaptation;

		bool perform_eigendecomposition(const parameters::Settings& settings) override;
	};

	struct MatrixAdaptation final : Adaptation
	{
		Matrix M;
		Matrix M_inv;

		MatrixAdaptation(const size_t dim, const Vector& x0) : Adaptation(dim, x0, Vector::Ones(dim)),
			M(Matrix::Identity(dim, dim)),
			M_inv(Matrix::Identity(dim, dim))
		{
		}

		void adapt_evolution_paths(const Population& pop, const parameters::Weights& w,
			const std::shared_ptr<mutation::Strategy>& mutation, const parameters::Stats& stats,
			size_t mu, size_t lambda) override;

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings) override;

		void restart(const parameters::Settings& settings) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};

	inline std::shared_ptr<Adaptation> get(const parameters::Modules& m, const size_t dim, const Vector& x0)
	{
		using namespace parameters;
		switch (m.matrix_adaptation)
		{
		case MatrixAdaptationType::MATRIX:
			return std::make_shared<MatrixAdaptation>(dim, x0);
		case MatrixAdaptationType::NONE:
			return std::make_shared<None>(dim, x0);
		case MatrixAdaptationType::SEPERABLE:
			return std::make_shared<SeperableAdaptation>(dim, x0);
		default:
		case MatrixAdaptationType::COVARIANCE:
			return std::make_shared<CovarianceAdaptation>(dim, x0);
		}
	}
} // namespace parameters
