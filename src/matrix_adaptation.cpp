#include "matrix_adaptation.hpp"

namespace matrix_adaptation
{
	using namespace parameters;


	Vector Adaptation::invert_x(const Vector& xi, const Float sigma)
	{
		return (xi - m) / sigma;
	}

	void Adaptation::adapt_evolution_paths(const Population& pop, const Weights& w,
		const Stats& stats, const size_t mu, const size_t lambda)
	{
		const auto sigma = pop.s.mean();
		dm = (m - m_old) / sigma;
		dz = pop.Z.leftCols(mu) * w.positive.head(mu);
		adapt_evolution_paths_inner(pop, w, stats, mu, lambda);
	}

	void CovarianceAdaptation::adapt_ps(const Weights& w)
	{
		ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * inv_root_C * dm);
	}


	void CovarianceAdaptation::adapt_evolution_paths_inner(const Population& pop, const Weights& w,
		const Stats& stats, const size_t mu, const size_t lambda)
	{
		adapt_ps(w);
		const Float actual_ps_length = ps.norm() / sqrt(
			1.0 - pow(1.0 - w.cs, 2.0 * (stats.evaluations / lambda)));

		hs = actual_ps_length < w.expected_length_ps;
		pc = (1.0 - w.cc) * pc + (hs * w.sqrt_cc_mueff) * dm;
	}

	void CovarianceAdaptation::adapt_covariance_matrix(const Weights& w, const Modules& m, const Population& pop,
		const size_t mu)
	{
		const auto dhs = (1 - hs) * w.cc * (2.0 - w.cc);
		const auto& rank_one = w.c1 * pc * pc.transpose();

		const auto& weights = m.active ? w.weights.topRows(pop.Y.cols()) : w.positive;
		const auto& popY = m.active ? pop.Y : pop.Y.leftCols(mu);
		const auto& old_c = (1 - (w.c1 * dhs) - w.c1 - (w.cmu * weights.sum())) * C;
		const auto& rank_mu = w.cmu * (popY * weights.asDiagonal() * popY.transpose());
		C = old_c + rank_one + rank_mu;
		C = 0.5 * (C + C.transpose().eval());
	}

	bool CovarianceAdaptation::perform_eigendecomposition(const Settings& settings)
	{
		const Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver(C);
		if (eigen_solver.info() != Eigen::Success)
		{
			if (settings.verbose)
			{
				std::cout << "Eigenvalue solver failed, we need to restart reason:"
					<< eigen_solver.info() << '\n';
			}
			return false;
		}
		d = eigen_solver.eigenvalues();
		B = eigen_solver.eigenvectors();

		if (d.minCoeff() < 0.0)
		{
			if (settings.verbose)
			{
				std::cout << "Negative eigenvalues after decomposition, we need to restart.\n";
			}
			return false;
		}


		d.noalias() = d.cwiseSqrt();
		inv_root_C.noalias() = eigen_solver.operatorInverseSqrt();
		A.noalias() = B * d.asDiagonal();
		return true;
	}

	bool CovarianceAdaptation::adapt_matrix(const Weights& w, const Modules& m, const Population& pop, const size_t mu,
		const Settings& settings, parameters::Stats& stats)
	{

		if (static_cast<Float>(stats.t) >= static_cast<Float>(stats.last_update) + w.lazy_update_interval)
		{
			stats.last_update = stats.t;
			stats.n_updates++;
			adapt_covariance_matrix(w, m, pop, mu);
			return perform_eigendecomposition(settings);
		}
		return true;

	}

	void CovarianceAdaptation::restart(const Settings& settings)
	{
		Adaptation::restart(settings);
		B = Matrix::Identity(settings.dim, settings.dim);
		C = Matrix::Identity(settings.dim, settings.dim);
		A = Matrix::Identity(settings.dim, settings.dim);
		inv_root_C = Matrix::Identity(settings.dim, settings.dim);
		d.setOnes();
		pc.setZero();
	}

	Vector CovarianceAdaptation::compute_y(const Vector& zi)
	{
		return A * zi;
	}

	Vector CovarianceAdaptation::invert_y(const Vector& yi)
	{
		return (B.transpose() * yi).cwiseQuotient(d);
	}


	void SeperableAdaptation::adapt_evolution_paths_inner(
		const Population& pop,
		const parameters::Weights& w,
		const parameters::Stats& stats,
		size_t mu, size_t lambda)
	{
		ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * dz);

		const Float actual_ps_length = ps.norm() / sqrt(
			1.0 - pow(1.0 - w.cs, 2.0 * (stats.evaluations / lambda)));

		hs = actual_ps_length < w.expected_length_ps;

		pc = (1.0 - w.cc) * pc + (hs * w.sqrt_cc_mueff) * dm;
	}

	bool SeperableAdaptation::adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
		const parameters::Settings& settings, parameters::Stats& stats)
	{

		stats.last_update = stats.t;
		stats.n_updates++;

		const auto dhs = (1 - hs) * w.cc * (2.0 - w.cc);

		const auto& weights = m.active ? w.weights.topRows(pop.Y.cols()) : w.positive;
		const auto& popY = m.active ? pop.Y : pop.Y.leftCols(mu);
		const auto decay_c = (1 - (w.c1 * dhs) - w.c1 - (w.cmu * weights.sum()));

		for (auto j = 0; j < settings.dim; j++)
		{
			const auto rank_mu = (popY.row(j).array().pow(2) * weights.transpose().array()).sum();
			c(j) = (decay_c * c(j)) + (w.c1 * pow(pc(j), 2)) + (w.cmu * rank_mu);
			c(j) = std::max(c(j), 1e-12);
			d(j) = std::sqrt(c(j));
		}

		return true;
	}

	void SeperableAdaptation::restart(const parameters::Settings& settings)
	{
		Adaptation::restart(settings);
		c.setOnes();
		d.setOnes();
		pc.setZero();
	}

	Vector SeperableAdaptation::compute_y(const Vector& zi)
	{
		return d.array() * zi.array();
	}

	Vector SeperableAdaptation::invert_y(const Vector& yi)
	{
		return yi.array() / d.array();
	}


	void OnePlusOneAdaptation::adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w,
		const parameters::Stats& stats,
		size_t mu, size_t lambda)
	{
		if (!stats.has_improved)
			return;

		pc = (1.0 - w.cc) * pc;
		if (stats.success_ratio < max_success_ratio)
			pc += w.sqrt_cc_mueff * pop.Y.col(0);
	}

	bool OnePlusOneAdaptation::adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
		const parameters::Settings& settings, parameters::Stats& stats)
	{
		if (!stats.has_improved)
		{
			return true;
		}

		stats.last_update = stats.t;
		stats.n_updates++;
		return CovarianceAdaptation::adapt_matrix(w, m, pop, mu, settings, stats);
	}



	void MatrixAdaptation::adapt_evolution_paths_inner(const Population& pop, const Weights& w,
		const Stats& stats, const size_t mu, const size_t lambda)
	{
		ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * dz);
	}

	bool MatrixAdaptation::adapt_matrix(const Weights& w, const Modules& m, const Population& pop, const size_t mu,
		const Settings& settings, parameters::Stats& stats)
	{
		constexpr Float epsilon = 1e-10;

		stats.last_update = stats.t;
		stats.n_updates++;

		const auto& weights = m.active ? w.weights.topRows(pop.Y.cols()) : w.positive;
		const auto& popZ = m.active ? pop.Z : pop.Z.leftCols(mu);
		const auto& Z = popZ * weights.asDiagonal() * popZ.transpose();

		ZwI.noalias() = (w.cmu / 2.0) * (Z - I);
		ssI.noalias() = (w.c1 / 2.0) * (ps * ps.transpose() - I);

		M = M * (I + ssI + ZwI);
		M_inv = (I - ssI - ZwI + epsilon * I) * M_inv;
		return true;
	}

	void MatrixAdaptation::restart(const Settings& settings)
	{
		Adaptation::restart(settings);
		M = Matrix::Identity(settings.dim, settings.dim);
		M_inv = Matrix::Identity(settings.dim, settings.dim);
	}

	Vector MatrixAdaptation::compute_y(const Vector& zi)
	{
		return M * zi;
	}

	Vector MatrixAdaptation::invert_y(const Vector& yi)
	{
		return M_inv * yi;
	}


	void None::adapt_evolution_paths_inner(const Population& pop, const Weights& w,
		const Stats& stats, const size_t mu, const size_t lambda)
	{
		ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * dz);
	}

	Vector None::compute_y(const Vector& zi)
	{
		return zi;
	}


	Vector None::invert_y(const Vector& yi)
	{
		return yi;
	}

	void CholeskyAdaptation::adapt_evolution_paths_inner(
		const Population& pop,
		const parameters::Weights& w,
		const parameters::Stats& stats,
		size_t mu, size_t lambda
	)
	{
		pc = (1.0 - w.cc) * pc + (w.sqrt_cc_mueff) * dm;
		ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * A.triangularView<Eigen::Lower>().solve(dm));
	}

	bool CholeskyAdaptation::adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
		const parameters::Settings& settings, parameters::Stats& stats)
	{

		stats.last_update = stats.t;
		stats.n_updates++;

		A *= std::sqrt(1 - w.c1 - w.cmu);

		Eigen::internal::llt_rank_update_lower(A, pc, w.c1);
		for (auto i = 0; i < mu; i++)
			Eigen::internal::llt_rank_update_lower(A, pop.Y.col(i), w.cmu * w.positive(i));

		if (m.active)
			for (auto i = 0; i < pop.Y.cols() - mu; i++)
				Eigen::internal::llt_rank_update_lower(A, pop.Y.col(mu + i), w.cmu * w.negative(i));


		return true;
	}

	void CholeskyAdaptation::restart(const parameters::Settings& settings)
	{
		Adaptation::restart(settings);
		A = Matrix::Identity(settings.dim, settings.dim);
		pc.setZero();
	}

	Vector CholeskyAdaptation::compute_y(const Vector& zi)
	{
		return A * zi;
	}

	Vector CholeskyAdaptation::invert_y(const Vector& yi)
	{
		return A.triangularView<Eigen::Lower>().solve(yi);
	}

	void SelfAdaptation::adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w, const parameters::Stats& stats, size_t mu, size_t lambda)
	{
		ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * A.triangularView<Eigen::Lower>().solve(dm));
	}

	bool SelfAdaptation::adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu, const parameters::Settings& settings, parameters::Stats& stats)
	{
		stats.last_update = stats.t;
		stats.n_updates++;

		const Float tc = 1.0 + (dd * (dd + 1)) / (2.0 * w.mueff);
		const Float tc_inv = 1.0 / tc;

		const auto& weights = m.active ? w.weights.topRows(pop.Y.cols()) : w.positive;
		const auto& popY = m.active ? pop.Y : pop.Y.leftCols(mu);
		const auto& Y = popY * weights.asDiagonal() * popY.transpose();

		C = (1.0 - tc_inv) * C + (tc_inv * Y);
		C = 0.5 * (C + C.transpose().eval());

		const Eigen::LLT<Matrix> chol(C);
		if (chol.info() != Eigen::Success)
		{
			if (settings.verbose)
				std::cout << "t: " << stats.t << "Cholesky solver failed, we need to restart reason:"
				<< chol.info() << '\n';
			return false;
		}
		A = chol.matrixL();

		return true;
	}

	void SelfAdaptation::restart(const parameters::Settings& settings)
	{
		A = Matrix::Identity(settings.dim, settings.dim);
	}

	Vector SelfAdaptation::compute_y(const Vector& zi)
	{
		return A * zi;
	}

	Vector SelfAdaptation::invert_y(const Vector& yi)
	{
		return A.triangularView<Eigen::Lower>().solve(yi);
	}


	void CovarainceNoEigvAdaptation::adapt_ps(const Weights& w)
	{
		ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * dz);
	}

	bool CovarainceNoEigvAdaptation::perform_eigendecomposition(const parameters::Settings& settings)
	{
		const Eigen::LLT<Matrix> chol(C);
		if (chol.info() != Eigen::Success)
		{
			if (settings.verbose)
			{
				std::cout << "Cholesky solver failed, we need to restart reason:"
					<< chol.info() << '\n';
			}
			return false;
		}

		A = chol.matrixL();
		return true;
	}

	Vector CovarainceNoEigvAdaptation::invert_y(const Vector& yi)
	{
		return A.triangularView<Eigen::Lower>().solve(yi);
	}

	void NaturalGradientAdaptation::adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w, const parameters::Stats& stats, size_t mu, size_t lambda)
	{
		ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * dz);
	}

	bool NaturalGradientAdaptation::adapt_matrix(
		const parameters::Weights& w, const parameters::Modules& m,
		const Population& pop, size_t mu, const parameters::Settings& settings, parameters::Stats& stats)
	{

		stats.last_update = stats.t;
		stats.n_updates++;
		static Float eta = 0.6 * (3 + std::log(settings.dim)) / std::pow(settings.dim, 1.5);

		G.setZero();
		for (int i = 0; i < w.positive.rows(); ++i)
		{
			const auto& z = pop.Z.col(i);
			G.noalias() += w.positive(i) * (z * z.transpose() - I);
		}
		
		// Remove isotropic (sigma-related) component: make G trace-free
		G -= (G.trace() / dd) * I;

		// Ensure symmetry for numerical stability
		G = 0.5 * (G + G.transpose().eval()); 

		// Apply the exponential update to A
		A *= ((0.5 * eta) * G).exp();

		return true;
	}

	void NaturalGradientAdaptation::restart(const parameters::Settings& settings)
	{
		Adaptation::restart(settings);
		A = Matrix::Identity(settings.dim, settings.dim);
		G = Matrix::Zero(settings.dim, settings.dim);
	}

	Vector NaturalGradientAdaptation::compute_y(const Vector& zi)
	{
		return A * zi;
	}

	Vector NaturalGradientAdaptation::invert_y(const Vector& yi)
	{
		//return A.triangularView<Eigen::Lower>().solve(yi);
		return A.fullPivLu().solve(yi);
	}

}
