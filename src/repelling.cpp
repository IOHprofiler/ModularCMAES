#include "repelling.hpp"
#include "parameters.hpp"

namespace repelling
{
	namespace distance
	{
		double manhattan(const Vector &u, const Vector &v)
		{
			return (u - v).cwiseAbs().sum();
		}

		double euclidian(const Vector &u, const Vector &v)
		{
			return (u - v).norm();
		}

		double mahanolobis(const Vector &u, const Vector &v, const Matrix &C_inv)
		{
			const auto delta = u - v;
			return std::sqrt(delta.transpose() * C_inv * delta);
		}

		//! Returns true when u belongs to the same basin as v
		bool hill_valley_test_p(
			const Solution &u,
			const Solution &v,
			FunctionType &f,
			const size_t n_evals,
			parameters::Parameters &p)
		{
			const double max_f = std::max(u.y, v.y);
			for (size_t k = 1; k < n_evals + 1; k++)
			{
				const double a = static_cast<double>(k) / (static_cast<double>(n_evals) + 1.0);
				const Vector x = v.x + a * (u.x - v.x);
				const double y = f(x);
				p.stats.evaluations++;
				if (max_f < y)
					return false;
				p.stats.update_best(x, y);
			}
			return true;
		}

		bool hill_valley_test(
			const Solution &u,
			const Solution &v,
			FunctionType &f,
			const size_t n_evals
			)
		{
			const double max_f = std::max(u.y, v.y);
			for (size_t k = 1; k < n_evals + 1; k++)
			{
				const double a = static_cast<double>(k) / (static_cast<double>(n_evals) + 1.0);
				const Vector x = v.x + a * (u.x - v.x);
				const double y = f(x);
				if (max_f < y)
					return false;
			}
			return true;
		}
	}

	bool TabooPoint::rejects(const Vector &xi, const parameters::Parameters &p, const int attempts) const
	{
		const double rejection_radius = std::pow(shrinkage, attempts) * radius;
		const double delta_xi = distance::mahanolobis(xi, solution.x, p.adaptation->inv_C) / p.mutation->sigma;

		if (delta_xi < rejection_radius)
			return true;

		return false;
	}

	bool TabooPoint::shares_basin(FunctionType &objective, const Solution &sol, parameters::Parameters &p) const
	{
		return distance::hill_valley_test_p(sol, solution, objective, 10, p);
	}

	void TabooPoint::calculate_criticality(const parameters::Parameters &p)
	{
		const double delta_m = distance::mahanolobis(p.adaptation->m, solution.x, p.adaptation->inv_C) / p.mutation->sigma;
		const auto u = delta_m + radius;
		const auto l = delta_m - radius;
		criticality = cdf(u) - cdf(l);
	}

	void Repelling::prepare_sampling(const parameters::Parameters &p)
	{
		attempts = 0;
		for (auto &point : archive)
			point.calculate_criticality(p);

		std::sort(archive.begin(), archive.end(), [](const TabooPoint &a, const TabooPoint &b)
				  { return a.criticality > b.criticality; });

		//! If it is not intialized
		/*
		if (C.cols() != p.settings.dim)
		{
			C = Matrix::Identity(p.settings.dim, p.settings.dim);
			C_inv = Matrix::Identity(p.settings.dim, p.settings.dim);
		}

		if (!(p.settings.modules.matrix_adaptation == parameters::MatrixAdaptationType::NONE ||
			  p.settings.modules.matrix_adaptation == parameters::MatrixAdaptationType::MATRIX))
		{
			using namespace matrix_adaptation;
			const auto dynamic = std::dynamic_pointer_cast<CovarianceAdaptation>(p.adaptation);

			const double d_sigma = p.mutation->sigma / p.settings.sigma0;
			if (d_sigma > constants::sigma_threshold)
			{
				C = dynamic->C / dynamic->C.maxCoeff();
				C_inv = dynamic->inv_C / dynamic->inv_C.maxCoeff();
			}
		}
		*/
	}

	void Repelling::update_archive(FunctionType &objective, parameters::Parameters &p)
	{
		const auto candidate_point = p.stats.centers.back();

		bool accept_candidate = true;

		for (auto &point : archive)
		{
			if (point.shares_basin(objective, candidate_point, p))
			{
				point.n_rep++;
				accept_candidate = false;

				if (point.solution > candidate_point)
				{
					point.solution = candidate_point;
				}
			}
		}

		if (accept_candidate)
			archive.emplace_back(candidate_point, 1.0);// , C, C_inv);

		const double volume_per_n = p.settings.volume / (p.settings.sigma0 * coverage * p.stats.solutions.size());
		const double n = p.adaptation->dd;
		const double gamma_f = std::pow(std::tgamma(n / 2.0 + 1.0), 1.0 / n) / std::sqrt(M_PI);
		for (auto &point : archive)
			point.radius = std::pow(volume_per_n * point.n_rep, 1.0 / n) * gamma_f;
	}

	bool Repelling::is_rejected(const Vector &xi, parameters::Parameters &p)
	{
		static constexpr double criticality_threshold = 0.01;
		if (!archive.empty())
		{
			for (const auto &point : archive)
			{
				// This assumes archive is sorted by criticality
				if (point.criticality < criticality_threshold)
					break;

				if (point.rejects(xi, p, attempts))
				{
					attempts++;
					return true;
				}
			}
		}

		return false;
	}
}
