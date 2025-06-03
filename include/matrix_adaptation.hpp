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

		void adapt_evolution_paths(const Population& pop, const parameters::Weights& w,
			const parameters::Stats& stats, size_t mu, size_t lambda);

		virtual void adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w,
			const parameters::Stats& stats, size_t mu, size_t lambda) = 0;

		virtual bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			size_t mu, const parameters::Settings& settings, parameters::Stats& stats) = 0;

		virtual Vector compute_y(const Vector&) = 0;

		virtual Vector invert_x(const Vector&, Float sigma);

		virtual Vector invert_y(const Vector&) = 0;

		virtual void restart(const parameters::Settings& settings)
		{
			m = settings.x0.value_or(Vector::Zero(settings.dim));
			m_old.setZero();
			dm.setZero();
			ps.setZero();
			dz.setZero();
		}

	};

	struct None final : Adaptation
	{
		None(const size_t dim, const Vector& x0, const Float expected_length_z) : Adaptation(dim, x0, Vector::Ones(dim), expected_length_z)
		{
		}

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop,
			const size_t mu, const parameters::Settings& settings, parameters::Stats& stats) override
		{
			return true;
		}

		void adapt_evolution_paths_inner(
			const Population& pop, 
			const parameters::Weights& w,
		    const parameters::Stats& stats,
			size_t mu, size_t lambda) override;


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

		virtual void adapt_ps(const parameters::Weights& w);
		
		void adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w,
			const parameters::Stats& stats,
			size_t mu, size_t lambda) override;

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};

	struct SeperableAdaptation : Adaptation
	{
		Vector pc, d, c;
		bool hs;
		
		SeperableAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z) : Adaptation(dim, x0, Vector::Zero(dim), expected_length_z),
			pc(Vector::Zero(dim)),
			d(Vector::Ones(dim)),
			c(Vector::Ones(dim)),
			hs(true)
		{
		}

		void adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w,
			const parameters::Stats& stats,
			size_t mu, size_t lambda) override;

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};


	struct OnePlusOneAdaptation: CovarianceAdaptation
	{
		constexpr static Float max_success_ratio = 0.44;

		using CovarianceAdaptation::CovarianceAdaptation;

		void adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w,
			const parameters::Stats& stats,
			size_t mu, size_t lambda) override;

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

	};


	struct MatrixAdaptation final : Adaptation
	{
		Matrix M;
		Matrix M_inv;

		MatrixAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z) : Adaptation(dim, x0, Vector::Ones(dim), expected_length_z),
			M(Matrix::Identity(dim, dim)),
			M_inv(Matrix::Identity(dim, dim)),
			/*ZwI(Matrix::Identity(dim, dim)),
			ssI(Matrix::Identity(dim, dim)),
			I(Matrix::Identity(dim, dim)), */
			outdated_M_inv(false)
		{
		}

		void adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w,
			 const parameters::Stats& stats,
			size_t mu, size_t lambda) override;

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;

	private:
		//Matrix ZwI, ssI, I;
		bool outdated_M_inv;
	}; 




	struct CholeskyAdaptation final : Adaptation
	{
		Matrix A;
		Vector pc;

		CholeskyAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z) 
			: Adaptation(dim, x0, Vector::Ones(dim), expected_length_z),
			A(Matrix::Identity(dim, dim)),
			pc(Vector::Zero(dim))
		{
		}

		void adapt_evolution_paths_inner(
			const Population& pop, 
			const parameters::Weights& w,
			const parameters::Stats& stats,
			size_t mu, size_t lambda
		) override;

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings) override;

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

		void adapt_evolution_paths_inner(
			const Population& pop,
			const parameters::Weights& w,
			const parameters::Stats& stats,
			size_t mu, size_t lambda
		) override;

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;
	};

	struct CovarainceNoEigvAdaptation final : CovarianceAdaptation
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

		NaturalGradientAdaptation(const size_t dim, const Vector& x0, const Float expected_length_z)
			: Adaptation(dim, x0, Vector::Ones(dim), expected_length_z),
			A(Matrix::Identity(dim, dim)),
			G(Matrix::Zero(dim, dim)),
			I(Matrix::Identity(dim, dim))
		{}

		void adapt_evolution_paths_inner(
			const Population& pop,
			const parameters::Weights& w,
			const parameters::Stats& stats,
			size_t mu, size_t lambda
		) override;

		bool adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
			const parameters::Settings& settings, parameters::Stats& stats) override;

		void restart(const parameters::Settings& settings) override;

		Vector compute_y(const Vector&) override;

		Vector invert_y(const Vector&) override;

	private:
		const Matrix I;
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
		case MatrixAdaptationType::CHOLESKY:
			return std::make_shared<CholeskyAdaptation>(dim, x0, expected_z);
		case MatrixAdaptationType::CMSA:
			return std::make_shared<SelfAdaptation>(dim, x0, expected_z);
		case MatrixAdaptationType::COVARIANCE_NO_EIGV:
			return std::make_shared<CovarainceNoEigvAdaptation>(dim, x0, expected_z);
		case MatrixAdaptationType::NATURAL_GRADIENT:
			return std::make_shared<NaturalGradientAdaptation>(dim, x0, expected_z);
		default:
		case MatrixAdaptationType::COVARIANCE:
			return std::make_shared<CovarianceAdaptation>(dim, x0, expected_z);
		}
	}
} // namespace parameters
