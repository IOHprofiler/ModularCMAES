#include "restart.hpp"
#include "parameters.hpp"
#include "matrix_adaptation.hpp"

#include <algorithm>

namespace restart
{
	//! max - min, for the last n elements of a vector
	double ptp_tail(const std::vector<double> &v, const size_t n)
	{
		const auto na = std::min(v.size(), n);
		if (na == 1)
		{
			return v[0];
		}

		const double min = *std::min_element(v.end() - na, v.end());
		const double max = *std::max_element(v.end() - na, v.end());
		return max - min;
	}

	// TODO: this is duplicate code
	double median(const Vector &x)
	{
		if (x.size() % 2 == 0)
			return (x(x.size() / 2) + x(x.size() / 2 - 1)) / 2.0;
		return x(x.size() / 2);
	}

	double median(const std::vector<double> &v, const size_t from, const size_t to)
	{
		const size_t n = to - from;
		if (n % 2 == 0)
			return (v[from + (n / 2)] + v[from + (n / 2) - 1]) / 2.0;
		return v[from + (n / 2)];
	}

	void RestartCriteria::update(const parameters::Parameters &p)
	{
		flat_fitnesses(p.stats.t % p.settings.dim) = p.pop.f(0) == p.pop.f(flat_fitness_index);
		median_fitnesses.push_back(median(p.pop.f));
		best_fitnesses.push_back(p.pop.f(0));

		time_since_restart = p.stats.t - last_restart;
		recent_improvement = ptp_tail(best_fitnesses, n_bin);
		n_flat_fitness = static_cast<size_t>(flat_fitnesses.sum());

		d_sigma = p.mutation->sigma / p.settings.sigma0;
		tolx_condition = 10e-12 * p.settings.sigma0;

		if (p.settings.modules.matrix_adaptation == parameters::MatrixAdaptationType::COVARIANCE ||
			p.settings.modules.matrix_adaptation == parameters::MatrixAdaptationType::SEPERABLE)
		{
			using namespace matrix_adaptation;
			const std::shared_ptr<CovarianceAdaptation> dynamic = std::dynamic_pointer_cast<CovarianceAdaptation>(
				p.adaptation);

			tolx_vector.head(p.settings.dim) = dynamic->C.diagonal() * d_sigma;
			tolx_vector.tail(p.settings.dim) = dynamic->pc * d_sigma;

			root_max_d = std::sqrt(dynamic->d.maxCoeff());
			condition_c = pow(dynamic->d.maxCoeff(), 2.0) / pow(dynamic->d.minCoeff(), 2);

			effect_coord = 0.2 * p.mutation->sigma * dynamic->C.diagonal().cwiseSqrt();

			const Eigen::Index t = p.stats.t % p.settings.dim;
			effect_axis = 0.1 * p.mutation->sigma * std::sqrt(dynamic->d(t)) * dynamic->B.col(t);
		}
	}

	bool RestartCriteria::exceeded_max_iter() const
	{
		return max_iter < time_since_restart;
	}

	bool RestartCriteria::no_improvement() const
	{
		return time_since_restart > n_bin and recent_improvement == 0;
	}

	bool RestartCriteria::flat_fitness() const
	{
		return time_since_restart > static_cast<size_t>(flat_fitnesses.size()) and n_flat_fitness > max_flat_fitness;
	}

	bool RestartCriteria::tolx() const
	{
		return (tolx_vector.array() < tolx_condition).all();
	}

	bool RestartCriteria::tolupsigma() const
	{
		return d_sigma > constants::tolup_sigma * root_max_d;
	}

	bool RestartCriteria::conditioncov() const
	{
		return condition_c > constants::tol_condition_cov;
	}

	bool RestartCriteria::noeffectaxis() const
	{
		return (effect_axis.array() == 0).all();
	}

	bool RestartCriteria::noeffectcoor() const
	{
		return (effect_coord.array() == 0).all();
	}

	bool RestartCriteria::min_sigma() const
	{
		return d_sigma < constants::tol_min_sigma;
	}

	bool RestartCriteria::stagnation() const
	{
		const size_t pt = static_cast<size_t>(constants::stagnation_quantile * time_since_restart);
		return time_since_restart > n_stagnation and ((median(best_fitnesses, pt, time_since_restart) >= median(
																											 best_fitnesses, 0, pt)) and
													  (median(median_fitnesses, pt, time_since_restart) >= median(median_fitnesses, 0, pt)));
	}

	bool RestartCriteria::operator()(const parameters::Parameters &p)
	{
		update(p);
		any = exceeded_max_iter() or no_improvement() or flat_fitness() or stagnation() or min_sigma();
		any = any or (p.settings.modules.matrix_adaptation == parameters::MatrixAdaptationType::COVARIANCE and (tolx() or tolupsigma() or conditioncov() or noeffectaxis() or noeffectcoor()));
		if (any)
		{
			if (p.settings.verbose)
			{
				std::cout << "restart criteria: " << p.stats.t << " (";
				std::cout << time_since_restart << std::boolalpha;
				std::cout << ") flat_fitness: " << flat_fitness();
				std::cout << " exeeded_max_iter: " << exceeded_max_iter();
				std::cout << " no_improvement: " << no_improvement();
				std::cout << " tolx: " << tolx();
				std::cout << " tolupsigma: " << tolupsigma();
				std::cout << " conditioncov: " << conditioncov();
				std::cout << " noeffectaxis: " << noeffectaxis();
				std::cout << " noeffectcoor: " << noeffectcoor();
				std::cout << " stagnation: " << stagnation() << '\n';
			}
			return true;
		}
		return false;
	}

	void Strategy::evaluate(parameters::Parameters &p)
	{
		if (criteria(p))
		{
			restart(p);
		}
	}

	void Restart::restart(parameters::Parameters &p)
	{
		p.perform_restart();
	}

	void IPOP::restart(parameters::Parameters &p)
	{
		const size_t max_lambda = static_cast<size_t>(std::pow(p.settings.dim * p.lambda, 2));
		if (p.mu < max_lambda)
		{
			p.mu *= static_cast<size_t>(ipop_factor);
			p.lambda *= static_cast<size_t>(ipop_factor);
		}
		p.perform_restart();
	}

	void BIPOP::restart(parameters::Parameters &p)
	{
		static std::uniform_real_distribution<> dist;

		const auto last_used_budget = p.stats.evaluations - used_budget;
		used_budget += last_used_budget;
		const auto remaining_budget = budget - used_budget;

		if (!lambda_large)
		{
			lambda_large = lambda_init * 2;
			budget_small = remaining_budget / 2;
			budget_large = remaining_budget - budget_small;
		}
		else if (large())
		{
			budget_large -= last_used_budget;
			lambda_large *= 2;
		}
		else
		{
			budget_small -= last_used_budget;
		}

		lambda_small = static_cast<size_t>(std::floor(
			static_cast<double>(lambda_init) * std::pow(.5 / static_cast<double>(lambda_large) / lambda_init,
														std::pow(dist(rng::GENERATOR), 2))));

		if (lambda_small % 2 != 0)
			lambda_small++;

		p.lambda = std::max(size_t{2}, large() ? lambda_large : lambda_small);
		p.mu = std::max(1.0, p.lambda * mu_factor);
		p.perform_restart(large() ? 2. : 2e-2 * dist(rng::GENERATOR));
	}
}
