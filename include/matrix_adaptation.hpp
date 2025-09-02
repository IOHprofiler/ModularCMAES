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
		Vector m, m_old, dm, ps, dz;
		Float dd;
		Float expected_length_z;

		Adaptation(const size_t dim, const Vector& x0, const Vector& ps, const Float expected_length_z) :
			m(x0), m_old(dim), dm(Vector::Zero(dim)),
			ps(ps), dd(static_cast<Float>(dim)),
			expected_length_z(expected_length_z)
		{
		}

		virtual void adapt_ps(const parameters::Weights& w);
		 
		void adapt_evolution_paths(
			const Population& pop, 
			const parameters::Weights& w,
			const parameters::Stats& stats, 
			const parameters::Settings& settings, 
			size_t lambda, size_t mu
		);

		virtual void adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w,
			const parameters::Stats& stats, const parameters::Settings& settings, size_t mu, size_t lambda) = 0;

		bool adapt_matrix(
			const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			size_t mu, const parameters::Settings& settings, parameters::Stats& stats)
		{
			if (settings.one_plus_one and !stats.has_improved)
				return true;
			return adapt_matrix_inner(w, m, pop, mu, settings, stats);

		}

		virtual bool adapt_matrix_inner(
			const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			size_t mu, const parameters::Settings& settings, parameters::Stats& stats) = 0;

		virtual Vector compute_y(const Vector&) = 0;

		virtual Vector invert_x(const Vector&, Float sigma);

		virtual Vector invert_y(const Vector&) = 0;

		virtual void restart(const parameters::Settings& settings, const Float sigma)
		{
			m = settings.x0.value_or(Vector::Zero(settings.dim));
			m_old.setZero();
			dm.setZero();
			ps.setZero();
			dz.setZero();
		}

		Float distance(const Vector u, const Vector& v)
		{
			const auto& delta = u - v;
			return invert_y(delta).norm();
		}

		Float distance_from_center(const Vector& xi) 		{
			return distance(m, xi);
		}
	};

	struct None final : Adaptation
	{
		None(const size_t dim, const Vector& x0, const Float expected_length_z) : Adaptation(dim, x0, Vector::Ones(dim), expected_length_z)
		{
		}

		bool adapt_matrix_inner(const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			const size_t mu, const parameters::Settings& settings, parameters::Stats& stats) override
		{
			return true;
		}

		void adapt_evolution_paths_inner(const Population& pop,
			const parameters::Weights& w,
			const parameters::Stats& stats, const parameters::Settings& settings, size_t mu, size_t lambda) override;


		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};

	struct CovarianceAdaptation : Adaptation
	{
		Vector pc, d;
		Matrix B, C;
		Matrix A;
		Matrix inv_root_C;
		bool hs = true;

		CovarianceAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z) : Adaptation(dim, x0, Vector::Zero(dim), expected_length_z),
			pc(Vector::Zero(dim)),
			d(Vector::Ones(dim)),
			B(Matrix::Identity(dim, dim)),
			C(Matrix::Identity(dim, dim)),
			A(Matrix::Identity(dim, dim)),
			inv_root_C(Matrix::Identity(dim, dim))
		{
		}

		void adapt_covariance_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			size_t mu);

		virtual bool perform_eigendecomposition(const parameters::Settings& settings);

		void adapt_ps(const parameters::Weights& w) override;

		void adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w,
			const parameters::Stats& stats, const parameters::Settings& settings, size_t mu, size_t lambda) override;

		bool adapt_matrix_inner(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings, const Float sigma) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};

	struct SeparableAdaptation : Adaptation
	{
		Vector pc, d, c;
		bool hs;

		SeparableAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z) : Adaptation(dim, x0, Vector::Zero(dim), expected_length_z),
			pc(Vector::Zero(dim)),
			d(Vector::Ones(dim)),
			c(Vector::Ones(dim)),
			hs(true)
		{
		}

		void adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w,
			const parameters::Stats& stats, const parameters::Settings& settings, size_t mu, size_t lambda) override;

		bool adapt_matrix_inner(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings, const Float sigma) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};


	struct MatrixAdaptation final : Adaptation
	{
		Matrix M;
		Matrix M_inv;

		MatrixAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z) : Adaptation(dim, x0, Vector::Ones(dim), expected_length_z),
			M(Matrix::Identity(dim, dim)),
			M_inv(Matrix::Identity(dim, dim)),
			outdated_M_inv(false)
		{
		}

		void adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w,
			const parameters::Stats& stats, const parameters::Settings& settings, size_t mu, size_t lambda) override;

		bool adapt_matrix_inner(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings, const Float sigma) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;

	private:
		bool outdated_M_inv;
	};

	struct CholeskyAdaptation final : Adaptation
	{
		Matrix A;
		Vector pc;
		/*
		First, as only triangular matrices have to be stored, the storage complexity is optimal.
		Second, the diagonal elements of a triangular Cholesky factor are the square roots of the eigenvalues
		of the factorized matrix, that is, we get the eigenvalues of the covariance matrix for free*/

		CholeskyAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z)
			: Adaptation(dim, x0, Vector::Ones(dim), expected_length_z),
			A(Matrix::Identity(dim, dim)),
			pc(Vector::Zero(dim))
		{
		}

		void adapt_evolution_paths_inner(const Population& pop,
			const parameters::Weights& w,
			const parameters::Stats& stats, const parameters::Settings& settings, size_t mu, size_t lambda) override;

		bool adapt_matrix_inner(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings, const Float sigma) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};

	struct SelfAdaptation final : Adaptation
	{
		Matrix A;
		Matrix C;

		SelfAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z)
			: Adaptation(dim, x0, Vector::Ones(dim), expected_length_z),
			A(Matrix::Identity(dim, dim)),
			C(Matrix::Identity(dim, dim))
		{}

		void adapt_evolution_paths_inner(const Population& pop,
			const parameters::Weights& w,
			const parameters::Stats& stats, const parameters::Settings& settings, size_t mu, size_t lambda) override;

		bool adapt_matrix_inner(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings, const Float sigma) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;

	};

	struct CovarianceNoEigvAdaptation : CovarianceAdaptation
	{
		using CovarianceAdaptation::CovarianceAdaptation;

		void adapt_ps(const parameters::Weights& w) override;

		bool perform_eigendecomposition(const parameters::Settings& settings) override;

		Vector invert_y(const Vector&) override;
	};

	struct NaturalGradientAdaptation final : Adaptation
	{
		Matrix A;
		Matrix G;
		Matrix A_inv;
		Float sigma_g;

		NaturalGradientAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z, const Float sigma0)
			: Adaptation(dim, x0, Vector::Ones(dim), expected_length_z),
			A(Matrix::Identity(dim, dim) / sigma0),
			G(Matrix::Zero(dim, dim)),
			A_inv(Matrix::Identity(dim, dim)),
			sigma_g(0),
			outdated_A_inv(false)
		{}

		void compute_gradients(
			const Population& pop, 
			const parameters::Weights& w, 
			const parameters::Stats& stats, 
			const parameters::Settings& settings,
			size_t mu, 
			size_t lambda
		);

		void adapt_evolution_paths_inner(const Population& pop,
			const parameters::Weights& w,
			const parameters::Stats& stats, const parameters::Settings& settings, size_t mu, size_t lambda) override;

		bool adapt_matrix_inner(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings, const Float sigma) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;

	private:
		bool outdated_A_inv;
	};

	inline std::shared_ptr<Adaptation> get(const parameters::Modules& m, const size_t dim, const Vector& x0, const Float expected_z, const Float sigma0)
	{
		using namespace parameters;
		switch (m.matrix_adaptation)
		{
		case MatrixAdaptationType::MATRIX:
			return std::make_shared<MatrixAdaptation>(dim, x0, expected_z);
		case MatrixAdaptationType::NONE:
			return std::make_shared<None>(dim, x0, expected_z);
		case MatrixAdaptationType::SEPARABLE:
			return std::make_shared<SeparableAdaptation>(dim, x0, expected_z);
		case MatrixAdaptationType::CHOLESKY:
			return std::make_shared<CholeskyAdaptation>(dim, x0, expected_z);
		case MatrixAdaptationType::CMSA:
			return std::make_shared<SelfAdaptation>(dim, x0, expected_z);
		case MatrixAdaptationType::COVARIANCE_NO_EIGV:
			return std::make_shared<CovarianceNoEigvAdaptation>(dim, x0, expected_z);
		case MatrixAdaptationType::NATURAL_GRADIENT:
			return std::make_shared<NaturalGradientAdaptation>(dim, x0, expected_z, sigma0);
		default:
		case MatrixAdaptationType::COVARIANCE:
			return std::make_shared<CovarianceAdaptation>(dim, x0, expected_z);
		}
	}
} // namespace parameters
