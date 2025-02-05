#include "matrix_adaptation.hpp"

namespace matrix_adaptation
{
	using namespace parameters;

	Vector Adaptation::invert_x(const Vector& xi, const double sigma)
	{
		return (xi - m) / sigma;
	}

	void CovarianceAdaptation::adapt_evolution_paths(const Population& pop, const Weights& w,
	                                                 const std::shared_ptr<mutation::Strategy>& mutation,
	                                                 const Stats& stats, const size_t mu, const size_t lambda)
	{
		dm = (m - m_old) / mutation->sigma;
		ps = (1.0 - mutation->cs) * ps + (sqrt(mutation->cs * (2.0 - mutation->cs) * w.mueff) * inv_root_C * dm);

		const double actual_ps_length = ps.norm() / sqrt(
			1.0 - pow(1.0 - mutation->cs, 2.0 * (stats.evaluations / lambda)));
		const double expected_ps_length = (1.4 + (2.0 / (dd + 1.0))) * expected_length_z;

		hs = actual_ps_length < expected_ps_length;
		pc = (1.0 - w.cc) * pc + (hs * sqrt(w.cc * (2.0 - w.cc) * w.mueff)) * dm;
	}

	void CovarianceAdaptation::adapt_covariance_matrix(const Weights& w, const Modules& m, const Population& pop,
	                                                   const size_t mu)
	{
		const auto rank_one = w.c1 * pc * pc.transpose();
		const auto dhs = (1 - hs) * w.cc * (2.0 - w.cc);
		const auto old_c = (1 - (w.c1 * dhs) - w.c1 - (w.cmu * w.positive.sum())) * C;

		Matrix rank_mu;
		if (m.active)
		{
			auto weights = w.weights.topRows(pop.Y.cols());
			rank_mu = w.cmu * ((pop.Y.array().rowwise() * weights.array().transpose()).matrix() * pop.Y.transpose());
		}
		else
		{
			rank_mu = w.cmu * ((pop.Y.leftCols(mu).array().rowwise() * w.positive.array().transpose()).matrix() * pop.Y.
				leftCols(mu).transpose());
		}

		C = old_c + rank_one + rank_mu;

		C = C.triangularView<Eigen::Upper>().toDenseMatrix() +
			C.triangularView<Eigen::StrictlyUpper>().toDenseMatrix().transpose();
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
		inv_C = ((B * d.cwiseInverse().asDiagonal()) * B.transpose());
		d = d.cwiseSqrt();
		inv_root_C = (B * d.cwiseInverse().asDiagonal()) * B.transpose();
		return true;
	}

	bool CovarianceAdaptation::adapt_matrix(const Weights& w, const Modules& m, const Population& pop, const size_t mu,
	                                        const Settings& settings)
	{
		adapt_covariance_matrix(w, m, pop, mu);
		return perform_eigendecomposition(settings);
	}

	void CovarianceAdaptation::restart(const Settings& settings)
	{
		B = Matrix::Identity(settings.dim, settings.dim);
		C = Matrix::Identity(settings.dim, settings.dim);
		inv_root_C = Matrix::Identity(settings.dim, settings.dim);
		inv_C = Matrix::Identity(settings.dim, settings.dim);
		d.setOnes();
		m = settings.x0.value_or(Vector::Zero(settings.dim));
		m_old.setZero();
		dm.setZero();
		pc.setZero();
		ps.setZero();
	}

	Vector CovarianceAdaptation::compute_y(const Vector& zi)
	{
		return B * (d.asDiagonal() * zi);
	}

	Vector CovarianceAdaptation::invert_y(const Vector& yi) 
	{
		return d.cwiseInverse().asDiagonal() * (B.transpose() * yi);
	}

	bool SeperableAdaptation::perform_eigendecomposition(const Settings& settings)
	{
		d = C.diagonal().cwiseSqrt();
		return d.minCoeff() > 0.0;
	}

	void MatrixAdaptation::adapt_evolution_paths(const Population& pop, const Weights& w,
	                                             const std::shared_ptr<mutation::Strategy>& mutation,
	                                             const Stats& stats, const size_t mu, const size_t lambda)
	{
		dm = (m - m_old) / mutation->sigma;

		const auto dz = (pop.Z.leftCols(mu).array().rowwise() * w.positive.array().transpose()).rowwise().sum().
			matrix();
		ps = (1.0 - mutation->cs) * ps + (sqrt(mutation->cs * (2.0 - mutation->cs) * w.mueff) * dz);
	}

	bool MatrixAdaptation::adapt_matrix(const Weights& w, const Modules& m, const Population& pop, const size_t mu,
	                                    const Settings& settings)
	{
		const auto old_m = (1. - (0.5 * w.c1) - (0.5 * w.cmu)) * M;
		const auto scaled_ps = (0.5 * w.c1) * (M * ps) * ps.transpose();

		const auto old_m_inv = (1. + (0.5 * w.c1) + (0.5 * w.cmu)) * M_inv;
		const auto scaled_inv_ps = (0.5 * w.c1) * ps * (ps.transpose() * M);

		Matrix new_m, new_m_inv;
		if (m.active)
		{
			// TODO: Check if we can do this like this
			const auto scaled_weights = ((0.5 * w.cmu) * w.weights.topRows(pop.Y.cols())).array().transpose();
			const auto scaled_y = (pop.Y.array().rowwise() * scaled_weights).matrix();
			new_m = scaled_y * pop.Z.transpose();
			new_m_inv = scaled_y * (pop.Z.transpose() * M_inv);
		}
		else
		{
			const auto scaled_weights = ((0.5 * w.cmu) * w.positive).array().transpose();
			const auto scaled_y = (pop.Y.leftCols(mu).array().rowwise() * scaled_weights).matrix();
			new_m = scaled_y * pop.Z.leftCols(mu).transpose();
			new_m_inv = scaled_y * (pop.Z.leftCols(mu).transpose() * M_inv);
		}

		M = old_m + scaled_ps + new_m;
		M_inv = old_m_inv - scaled_inv_ps - new_m_inv;
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


	void None::adapt_evolution_paths(const Population& pop, const Weights& w,
	                                 const std::shared_ptr<mutation::Strategy>& mutation, const
	                                 Stats& stats, const size_t mu, const size_t lambda)
	{
		dm = (m - m_old) / mutation->sigma;

		const auto dz = (pop.Z.leftCols(mu).array().rowwise() * w.positive.array().transpose()).rowwise().sum().
			matrix();
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
