#include "restart_strategy.hpp"
#include "parameters.hpp"
#include "matrix_adaptation.hpp"



namespace restart
{

	Float Strategy::update(parameters::Parameters &p)
	{ 
		return p.settings.sigma0;
	}
	
	Float IPOP::update(parameters::Parameters &p)
	{
		const size_t max_lambda = static_cast<size_t>(std::pow(p.settings.dim * p.lambda, 2));
		if (p.mu < max_lambda)
		{
			p.mu *= static_cast<size_t>(ipop_factor);
			p.lambda *= static_cast<size_t>(ipop_factor);
		}
		return p.settings.sigma0;
	}

	Float BIPOP::update(parameters::Parameters &p)
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
			static_cast<Float>(lambda_init) * std::pow(.5 / static_cast<Float>(lambda_large) / lambda_init,
														std::pow(dist(rng::GENERATOR), 2))));

		if (lambda_small % 2 != 0)
			lambda_small++;

		p.lambda = std::max(size_t{2}, large() ? lambda_large : lambda_small);
		p.mu = std::max(Float{1.0}, p.lambda * mu_factor);
		return large() ? p.settings.sigma0 : p.settings.sigma0 * std::pow(10, -2 * dist(rng::GENERATOR));
	}
}
