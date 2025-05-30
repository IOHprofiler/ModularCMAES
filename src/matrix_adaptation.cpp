#include "matrix_adaptation.hpp"

namespace matrix_adaptation
{
	using namespace parameters;

	Vector Adaptation::invert_x(const Vector& xi, const Float sigma)
	{
		return (xi - m) / sigma;
	}

	void Adaptation::adapt_evolution_paths(const Population& pop, const Weights& w,
		const std::shared_ptr<mutation::Strategy>& mutation,
		const Stats& stats, const size_t mu, const size_t lambda)
	{
		dm = (m - m_old) / mutation->sigma;
		dz = pop.Z.leftCols(mu) * w.positive.head(mu);
		adapt_evolution_paths_inner(pop, w, mutation, stats, mu, lambda);
	}


	void CovarianceAdaptation::adapt_evolution_paths_inner(const Population& pop, const Weights& w,
		const std::shared_ptr<mutation::Strategy>& mutation,
		const Stats& stats, const size_t mu, const size_t lambda)
	{
		const auto& expr = constants::calc_eigv ? inv_root_C * dm : dz;
		
		ps = (1.0 - mutation->cs) * ps + (sqrt(mutation->cs * (2.0 - mutation->cs) * w.mueff) * expr);

		const Float actual_ps_length = ps.norm() / sqrt(
			1.0 - pow(1.0 - mutation->cs, 2.0 * (stats.evaluations / lambda)));

		const Float expected_ps_length = (1.4 + (2.0 / (dd + 1.0))) * expected_length_z;

		hs = actual_ps_length < expected_ps_length;
		pc = (1.0 - w.cc) * pc + (hs * sqrt(w.cc * (2.0 - w.cc) * w.mueff)) * dm;
	}

	void CovarianceAdaptation::adapt_covariance_matrix(const Weights& w, const Modules& m, const Population& pop,
		const size_t mu)
	{
		const auto rank_one = w.c1 * pc * pc.transpose();
		const auto dhs = (1 - hs) * w.cc * (2.0 - w.cc);
		const auto old_c = (1 - (w.c1 * dhs) - w.c1 - (w.cmu * w.positive.sum())) * C;

		if (m.active)
		{
			auto weights = w.weights.topRows(pop.Y.cols());
			C = old_c + rank_one + w.cmu * ((pop.Y.array().rowwise() * weights.array().transpose()).matrix() * pop.Y.transpose());
		}
		else
		{
			C = old_c + rank_one + (w.cmu * ((pop.Y.leftCols(mu).array().rowwise() * w.positive.array().transpose()).matrix() * pop.Y.
				leftCols(mu).transpose()));

		}
		C = 0.5 * (C + C.transpose().eval());
	}

	bool CovarianceAdaptation::perform_eigendecomposition(const Settings& settings)
	{
		if (!constants::calc_eigv) 
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
		B = Matrix::Identity(settings.dim, settings.dim);
		C = Matrix::Identity(settings.dim, settings.dim);
		A = Matrix::Identity(settings.dim, settings.dim);
		inv_root_C = Matrix::Identity(settings.dim, settings.dim);
		d.setOnes();
		m = settings.x0.value_or(Vector::Zero(settings.dim));
		m_old.setZero();
		dm.setZero();
		pc.setZero();
		ps.setZero(); 
	}

	Vector CovarianceAdaptation::compute_y(const Vector& zi)
	{
		return A * zi;
	}

	Vector CovarianceAdaptation::invert_y(const Vector& yi)
	{
		if (!constants::calc_eigv)
			return A.triangularView<Eigen::Lower>().solve(yi);
		
		return (B.transpose() * yi).cwiseQuotient(d);
	}

	bool SeperableAdaptation::perform_eigendecomposition(const Settings& settings)
	{
		d = C.diagonal().cwiseSqrt();
		return d.minCoeff() > 0.0;
	}


	void OnePlusOneAdaptation::adapt_evolution_paths_inner(const Population& pop, const parameters::Weights& w,
		const std::shared_ptr<mutation::Strategy>& mutation, const parameters::Stats& stats,
		size_t mu, size_t lambda)
	{
		if (!stats.has_improved)
			return;

		if (stats.success_ratio < max_success_ratio)
			pc = ((1.0 - w.cc) * pc) + (std::sqrt(w.cc * (2.0 - w.cc)) * pop.Y.col(0));
		else
			pc = (1.0 - w.cc) * pc;
	}

	bool OnePlusOneAdaptation::adapt_matrix(const parameters::Weights& w, const parameters::Modules& m, const Population& pop, size_t mu,
		const parameters::Settings& settings, parameters::Stats& stats)
	{
		if (!stats.has_improved)
		{
			return true;
		}
		return CovarianceAdaptation::adapt_matrix(w, m, pop, mu, settings, stats);
	}



	void MatrixAdaptation::adapt_evolution_paths_inner(const Population& pop, const Weights& w,
	                                             const std::shared_ptr<mutation::Strategy>& mutation,
	                                             const Stats& stats, const size_t mu, const size_t lambda)
	{
		ps = (1.0 - mutation->cs) * ps + (sqrt(mutation->cs * (2.0 - mutation->cs) * w.mueff) * dz);
	}

	bool MatrixAdaptation::adapt_matrix(const Weights& w, const Modules& m, const Population& pop, const size_t mu,
	                                    const Settings& settings, parameters::Stats& stats)
	{
		const auto old_m = (1. - (0.5 * w.c1) - (0.5 * w.cmu)) * M;
		const auto scaled_ps = (0.5 * w.c1) * (M * ps) * ps.transpose();

		const auto old_m_inv = (1. + (0.5 * w.c1) + (0.5 * w.cmu)) * M_inv;
		const auto scaled_inv_ps = (0.5 * w.c1) * ps * (ps.transpose() * M);

		if (m.active)
		{
			const auto scaled_weights = ((0.5 * w.cmu) * w.weights.topRows(pop.Y.cols())).array().transpose();
			const auto scaled_y = (pop.Y.array().rowwise() * scaled_weights).matrix();

			M = old_m + scaled_ps + scaled_y * pop.Z.transpose();
			M_inv = old_m_inv - scaled_inv_ps - scaled_y * (pop.Z.transpose() * M_inv);
		}
		else
		{
			const auto scaled_weights = ((0.5 * w.cmu) * w.positive).array().transpose();
			const auto scaled_y = (pop.Y.leftCols(mu).array().rowwise() * scaled_weights).matrix();
			M = old_m + scaled_ps + scaled_y * pop.Z.leftCols(mu).transpose();
			M_inv = old_m_inv - scaled_inv_ps - scaled_y * (pop.Z.leftCols(mu).transpose() * M_inv);
		}
		return true;
	}

	void MatrixAdaptation::restart(const Settings& settings)
	{
		ps.setOnes();
		m = settings.x0.value_or(Vector::Zero(settings.dim));
		m_old.setZero();
		dm.setZero();
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
	                                 const std::shared_ptr<mutation::Strategy>& mutation, const
	                                 Stats& stats, const size_t mu, const size_t lambda)
	{
		ps = (1.0 - mutation->cs) * ps + (sqrt(mutation->cs * (2.0 - mutation->cs) * w.mueff) * dz);
	}

	void None::restart(const Settings& settings)
	{
		ps.setZero();
		m = settings.x0.value_or(Vector::Zero(settings.dim));
		m_old.setZero();
		dm.setZero();
	}

	Vector None::compute_y(const Vector& zi)
	{
		return zi;
	}


	Vector None::invert_y(const Vector& yi)
	{
		return yi;
	}
}
