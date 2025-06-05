#include "repelling.hpp"
#include "parameters.hpp"

namespace repelling
{
	namespace distance
	{
		Float manhattan(const Vector &u, const Vector &v)
		{
			return (u - v).cwiseAbs().sum();
		}

		Float euclidian(const Vector &u, const Vector &v)
		{
			return (u - v).norm();
		}

		Float mahanolobis(const Vector &u, const Vector &v, const Matrix &C_inv)
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
			const Float max_f = std::max(u.y, v.y);
			for (size_t k = 1; k < n_evals + 1; k++)
			{
				const Float a = static_cast<Float>(k) / (static_cast<Float>(n_evals) + 1.0);
				const Vector x = v.x + a * (u.x - v.x);
				const Float y = f(x);
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
			const Float max_f = std::max(u.y, v.y);
			for (size_t k = 1; k < n_evals + 1; k++)
			{
				const Float a = static_cast<Float>(k) / (static_cast<Float>(n_evals) + 1.0);
				const Vector x = v.x + a * (u.x - v.x);
				const Float y = f(x);
				if (max_f < y)
					return false;
			}
			return true;
		}
	}

	bool TabooPoint::rejects(const Vector &xi, const parameters::Parameters &p, const int attempts) const
	{
		const Float rejection_radius = std::pow(shrinkage, attempts) * radius;
		const Float delta_xi = p.adaptation->distance(xi, solution.x) / p.mutation->sigma;
		
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
		const Float delta_m = p.adaptation->distance_from_center(solution.x) / p.mutation->sigma;
			
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
			archive.emplace_back(candidate_point, 1.0);
		
		const Float volume_per_n = p.settings.volume / (p.settings.sigma0 * coverage * p.stats.solutions.size());
		const Float n = p.adaptation->dd;
		const Float gamma_f = std::pow(std::tgamma(n / 2.0 + 1.0), 1.0 / n) / std::sqrt(M_PI);
		for (auto &point : archive)
			point.radius = (std::pow(volume_per_n * point.n_rep, 1.0 / n) * gamma_f) / std::sqrt(n);
	}

	bool Repelling::is_rejected(const Vector &xi, parameters::Parameters &p)
	{
		static constexpr Float criticality_threshold = 0.01;
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
