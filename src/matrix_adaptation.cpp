#include "matrix_adaptation.hpp"

namespace matrix_adaptation
{
	using namespace parameters;


	Vector Adaptation::invert_x(const Vector& xi, const Float sigma)
	{
		return (xi - m) / sigma;
	}

	static void one_plus_one_path_update(
		Vector& path, 
		const Population& pop, 
		const parameters::Stats& stats, 
		const Float c,
		const Float sqrt_c_mueff,
		const Vector& v
	)
	{
		constexpr static Float max_success_ratio = 0.44;
		if (!stats.has_improved)
			return;

		path = (1.0 - c) * path;
		if (stats.success_ratio < max_success_ratio)
			path += sqrt_c_mueff * v;
	}

	void Adaptation::adapt_evolution_paths(const Population& pop, const Weights& w,
		const Stats& stats, const parameters::Settings& settings, const size_t lambda, const size_t mu)
	{
		const auto sigma = pop.s.mean();
		dm = (m - m_old) / sigma; 
		dz = pop.Z.leftCols(mu) * w.positive.head(mu);
		adapt_evolution_paths_inner(pop, w, stats, settings, mu, lambda);
	}

	void Adaptation::adapt_ps(const parameters::Weights& w)
	{
		ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * dz);
	}

	void None::adapt_evolution_paths_inner(const Population& pop, const Weights& w,
		const Stats& stats, const parameters::Settings& settings, const size_t mu, const size_t lambda)
	{
		if (!settings.one_plus_one)
			adapt_ps(w);
	}

	Vector None::compute_y(const Vector& zi)
	{
		return zi;
	}


	Vector None::invert_y(const Vector& yi)
	{
		return yi;
	}

	void CovarianceAdaptation::adapt_ps(const Weights& w)
	{
		ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * inv_root_C * dm);
	}

	void CovarianceAdaptation::adapt_evolution_paths_inner(const Population& pop, const Weights& w,
		const Stats& stats, const parameters::Settings& settings, const size_t mu, const size_t lambda)
	{
		if (settings.one_plus_one)
		{
			one_plus_one_path_update(pc, pop, stats, w.cc, w.sqrt_cc_mueff, pop.Y.col(0));
			return;
		}
		
		adapt_ps(w);
		const Float actual_ps_length = ps.norm() / sqrt(
			1.0 - pow(1.0 - w.cs, 2.0 * (stats.evaluations / lambda)));

		hs = actual_ps_length < w.expected_length_ps;
		pc = (1.0 - w.cc) * pc + (hs * w.sqrt_cc_mueff) * dm;
	}

	void CovarianceAdaptation::adapt_covariance_matrix(
		const Weights& w, 
		const Modules& m, 
		const Population& pop,
		const size_t mu
	)
	{
		const auto dhs = (1.0 - hs) * w.cc * (2.0 - w.cc);
		const auto& rank_one = w.c1 * pc * pc.transpose();

		const auto& weights = m.active ? w.weights.topRows(pop.Y.cols()) : w.positive;
		const auto& popY = m.active ? pop.Y : pop.Y.leftCols(mu);

		const Float decay = (1 - (w.c1 * dhs) - w.c1 - (w.cmu * weights.sum()));
		const auto& old_c = decay * C;

		Vector rank_mu_w = weights.eval();
		for (size_t i = mu; i < weights.size() - mu; i++)
			rank_mu_w(i) *= dd / (inv_root_C * popY.col(i)).squaredNorm();

		const auto& rank_mu = w.cmu * (popY * rank_mu_w.asDiagonal() * popY.transpose());
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

		d.noalias() = d.cwiseSqrt().eval();
		inv_root_C.noalias() = B * d.cwiseInverse().asDiagonal() * B.transpose();
		A.noalias() = B * d.asDiagonal();
		return true;
	}

	bool CovarianceAdaptation::adapt_matrix_inner(const Weights& w, const Modules& m, const Population& pop, const size_t mu,
		const Settings& settings, parameters::Stats& stats)
	{

		if (static_cast<Float>(stats.t) >= static_cast<Float>(stats.last_update) + w.lazy_update_interval)
		{
			stats.last_update = stats.t;
			stats.n_updates++;
			adapt_covariance_matrix(w, m, pop, mu);
			auto succ = perform_eigendecomposition(settings);
			if (!succ and settings.verbose)
			{
				std::cout << "t: " << stats.t << ". C:\n";
				std::cout << C << std::endl << std::endl;
			}
			return succ;
		}
		return true;

	}

	void CovarianceAdaptation::restart(const Settings& settings, const Float sigma)
	{
		Adaptation::restart(settings, sigma);
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


	void SeparableAdaptation::adapt_evolution_paths_inner(const Population& pop,
		const parameters::Weights& w,
		const parameters::Stats& stats, const parameters::Settings& settings, size_t mu, size_t lambda)
	{
		if (settings.one_plus_one)
		{
			one_plus_one_path_update(pc, pop, stats, w.cc, w.sqrt_cc_mueff, pop.Y.col(0));
			return;
		}
		
		adapt_ps(w);

		const Float actual_ps_length = ps.norm() / sqrt(
			1.0 - pow(1.0 - w.cs, 2.0 * (stats.evaluations / lambda)));

		hs = actual_ps_length < w.expected_length_ps;
		pc = (1.0 - w.cc) * pc + (hs * w.sqrt_cc_mueff) * dm;
	}

	bool SeparableAdaptation::adapt_matrix_inner(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
		const parameters::Settings& settings, parameters::Stats& stats)
	{

		stats.last_update = stats.t;
		stats.n_updates++;

		const auto dhs = (1 - hs) * w.cc * (2.0 - w.cc);

		const auto& weights = m.active ? w.weights.topRows(pop.Y.cols()) : w.positive;
		const auto& popY = m.active ? pop.Y : pop.Y.leftCols(mu);
		const auto decay_c = (1 - (w.c1 * dhs) - w.c1 - (w.cmu * weights.sum()));

		for (size_t j = 0; j < settings.dim; j++)
		{
			const auto rank_mu = (popY.row(j).array().pow(2) * weights.transpose().array()).sum();
			c(j) = (decay_c * c(j)) + (w.c1 * pow(pc(j), 2)) + (w.cmu * rank_mu);
			c(j) = std::max(c(j), Float{ 1e-12 });
			d(j) = std::sqrt(c(j));
		}

		return true;
	}

	void SeparableAdaptation::restart(const parameters::Settings& settings, const Float sigma)
	{
		Adaptation::restart(settings, sigma);
		c.setOnes();
		d.setOnes();
		pc.setZero();
	}

	Vector SeparableAdaptation::compute_y(const Vector& zi)
	{
		return d.array() * zi.array();
	}

	Vector SeparableAdaptation::invert_y(const Vector& yi)
	{
		return yi.array() / d.array();
	}


	void MatrixAdaptation::adapt_evolution_paths_inner(const Population& pop, const Weights& w,
		const Stats& stats, const parameters::Settings& settings, const size_t mu, const size_t lambda)
	{
		if (settings.one_plus_one && !stats.has_improved)
			return;
		adapt_ps(w);
	}

	bool MatrixAdaptation::adapt_matrix_inner(const Weights& w, const Modules& m, const Population& pop, const size_t mu,
		const Settings& settings, parameters::Stats& stats)
	{
		
		stats.last_update = stats.t;
		stats.n_updates++;
		
		const auto& weights = m.active ? w.weights.topRows(pop.Z.cols()) : w.positive;
		const auto& popZ = m.active ? pop.Z : pop.Z.leftCols(mu);
		const auto& popY = m.active ? pop.Y : pop.Y.leftCols(mu);
		
		// Normal MA-ES -> O(n^3)
		// 
		// constexpr Float epsilon = 1e-10;
		// const auto& Z = popZ * weights.asDiagonal() * popZ.transpose();
		// ZwI.noalias() = (w.cmu / 2.0) * (Z - I);
		// ssI.noalias() = (w.c1 / 2.0) * (ps * ps.transpose() - I);
		// M = M * (I + ssI + ZwI);
		// M_inv = (I - ssI - ZwI + epsilon * I) * M_inv;

		// Fast MA-ES -> O(n^2)
		const Float tau_1 = w.c1 / 2.0;
		const Float tau_m = w.cmu / 2.0;
		const Float decay_m = (1.0 - tau_1 - tau_m);

		M = (decay_m * M) 
			+ (tau_1 * (M * ps) * ps.transpose()) 
			+ (popY * (tau_m * weights).asDiagonal() * popZ.transpose());


		if (settings.modules.elitist && !settings.one_plus_one)
			M_inv = (decay_m * M_inv)
				+ (tau_1 * ps * (ps.transpose() * M_inv))
				+ ((popY * (tau_m * weights).asDiagonal()) * (popZ.transpose() * M_inv));
		else
			outdated_M_inv = true; // Rely on moore penrose pseudo-inv (only when needed)
		return true;
	}

	void MatrixAdaptation::restart(const Settings& settings, const Float sigma)
	{
		Adaptation::restart(settings, sigma);
		M = Matrix::Identity(settings.dim, settings.dim);
		M_inv = Matrix::Identity(settings.dim, settings.dim);
		outdated_M_inv = false;
	}

	Vector MatrixAdaptation::compute_y(const Vector& zi)
	{
		return M * zi;
	}

	Vector MatrixAdaptation::invert_y(const Vector& yi)
	{
		if (outdated_M_inv) {
			M_inv = M.completeOrthogonalDecomposition().pseudoInverse();
			outdated_M_inv = false;
		}
		return M_inv * yi;
	}


	void CholeskyAdaptation::adapt_evolution_paths_inner(const Population& pop,
		const parameters::Weights& w,
		const parameters::Stats& stats, const parameters::Settings& settings, size_t mu, size_t lambda)
	{
		if (settings.one_plus_one)
		{
			one_plus_one_path_update(pc, pop, stats, w.cc, w.sqrt_cc_mueff, pop.Y.col(0));
			return;
		}

		adapt_ps(w);
		pc = (1.0 - w.cc) * pc + (w.sqrt_cc_mueff * dm);
	}

	bool CholeskyAdaptation::adapt_matrix_inner(const parameters::Weights & w, const parameters::Modules & m, const Population & pop, size_t mu,
		const parameters::Settings& settings, parameters::Stats& stats)
	{

		stats.last_update = stats.t;
		stats.n_updates++;

		A *= std::sqrt(1 - w.c1 - w.cmu);

		Eigen::internal::llt_rank_update_lower(A, pc, w.c1);
		for (size_t i = 0; i < mu; i++)
			Eigen::internal::llt_rank_update_lower(A, pop.Y.col(i), w.cmu * w.positive(i));

		if (m.active)
			for (size_t i = 0; i < pop.Y.cols() - mu; i++)
				Eigen::internal::llt_rank_update_lower(A, pop.Y.col(mu + i), w.cmu * w.negative(i));


		return true;
	}

	void CholeskyAdaptation::restart(const parameters::Settings& settings, const Float sigma)
	{
		Adaptation::restart(settings, sigma);
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

	void SelfAdaptation::adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w, const parameters::Stats& stats, const parameters::Settings& settings, size_t mu, size_t lambda)
	{
		
		if (!settings.one_plus_one)
			adapt_ps(w);
	}

	bool SelfAdaptation::adapt_matrix_inner(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu, const parameters::Settings& settings, parameters::Stats& stats)
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

	void SelfAdaptation::restart(const parameters::Settings& settings, const Float sigma)
	{
		Adaptation::restart(settings, sigma);
		A = Matrix::Identity(settings.dim, settings.dim);
		C = Matrix::Identity(settings.dim, settings.dim);
	}

	Vector SelfAdaptation::compute_y(const Vector& zi)
	{
		return A * zi;
	}

	Vector SelfAdaptation::invert_y(const Vector& yi)
	{
		return A.triangularView<Eigen::Lower>().solve(yi);
	}

	void CovarianceNoEigvAdaptation::adapt_ps(const Weights& w)
	{
		Adaptation::adapt_ps(w);
	}

	bool CovarianceNoEigvAdaptation::perform_eigendecomposition(const parameters::Settings& settings)
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

	Vector CovarianceNoEigvAdaptation::invert_y(const Vector& yi)
	{
		return A.triangularView<Eigen::Lower>().solve(yi);
	}

	void NaturalGradientAdaptation::adapt_evolution_paths_inner(
		const Population& pop, 
		const parameters::Weights& w, 
		const parameters::Stats& stats, 
		const parameters::Settings& settings, 
		size_t mu, 
		size_t lambda 
	)
	{
		if (!settings.one_plus_one)
		{
			adapt_ps(w);
			compute_gradients(pop, w, stats, settings, mu, lambda);
			return;
		}

		if (stats.has_improved)
		{
			const auto& z = pop.Z.col(0);
			G.noalias() = (z * z.transpose() - Matrix::Identity(settings.dim, settings.dim));
		}
	}

	void NaturalGradientAdaptation::compute_gradients(
		const Population& pop, 
		const parameters::Weights& w, 
		const parameters::Stats& stats, 
		const parameters::Settings& settings,
		size_t mu, 
		size_t lambda
	)
	{
		const size_t dim = pop.Z.rows();

		const auto& weights = settings.modules.active ? w.weights.topRows(pop.Z.cols()) : w.positive;
		
		G.setZero();
		for (int i = 0; i < weights.rows(); ++i)
		{
			const auto& z = pop.Z.col(i);
			G.noalias() += weights(i) * (z * z.transpose() - Matrix::Identity(dim, dim));
		}

		// Remove isotropic (sigma-related) component: make G trace-free
		sigma_g = (G.trace() / dd);
		
		if (!settings.one_plus_one)
			G.diagonal().array() -= sigma_g;

		// Ensure symmetry for numerical stability
		G = 0.5 * (G + G.transpose().eval());
	}

	bool NaturalGradientAdaptation::adapt_matrix_inner(
		const parameters::Weights& w, const parameters::Modules& m,
		const Population& pop, size_t mu, const parameters::Settings& settings, parameters::Stats& stats)
	{

		stats.last_update = stats.t;
		stats.n_updates++;

		A *= (w.cc * G).exp();
		outdated_A_inv = true;

		return true;
	}

	void NaturalGradientAdaptation::restart(const parameters::Settings& settings, const Float sigma)
	{
		Adaptation::restart(settings, sigma);
		A = Matrix::Identity(settings.dim, settings.dim) / sigma;
		A_inv = Matrix::Identity(settings.dim, settings.dim);
		G = Matrix::Zero(settings.dim, settings.dim);
		outdated_A_inv = false;
		sigma_g = 0.;
	}

	Vector NaturalGradientAdaptation::compute_y(const Vector& zi)
	{
		return A * zi;
	}

	Vector NaturalGradientAdaptation::invert_y(const Vector& yi)
	{
		if (outdated_A_inv)
			A_inv = A.completeOrthogonalDecomposition().pseudoInverse();
		return A_inv * yi;
	}
}
