#include "parameters.hpp"

namespace parameters
{
	Parameters::Parameters(const Settings& settings) :
		lambda(settings.lambda0),
		mu(settings.mu0),
		settings(settings),
		weights(settings.dim, settings.mu0, settings.lambda0, settings),
		pop(settings.dim, settings.lambda0),
		old_pop(settings.dim, settings.lambda0),
		adaptation(matrix_adaptation::get(settings.modules, settings.dim,
		                                  settings.x0.value_or(Vector::Zero(settings.dim)))),
		sampler(sampling::get(settings.dim, settings.modules, settings.lambda0)),
		mutation(mutation::get(settings.modules,
		                       settings.mu0, weights.mueff,
		                       static_cast<double>(settings.dim),
		                       settings.sigma0,
		                       settings.cs
		)),
		selection(std::make_shared<selection::Strategy>(settings.modules)),
		restart(restart::get(settings.modules.restart_strategy,
		                     static_cast<double>(settings.dim),
		                     static_cast<double>(settings.lambda0),
		                     static_cast<double>(settings.mu0),
		                     settings.budget)
		),
		bounds(bounds::get(settings.modules.bound_correction, settings.lb, settings.ub)),
		repelling(repelling::get(settings.modules))
	{
	}

	Parameters::Parameters(const size_t dim) : Parameters(Settings(dim, {}))
	{
	}

	void Parameters::perform_restart(const std::optional<double>& sigma)
	{
		weights = Weights(settings.dim, mu, lambda, settings);
		sampler = sampling::get(settings.dim, settings.modules, lambda);

		pop = Population(settings.dim, lambda);
		old_pop = Population(settings.dim, lambda);

		mutation = mutation::get(settings.modules, mu, weights.mueff,
		                         static_cast<double>(settings.dim),
		                         sigma.value_or(settings.sigma0),
		                         settings.cs
		);
		adaptation->restart(settings);

		restart->criteria = restart::RestartCriteria(settings.dim, lambda, stats.t);

		stats.solutions.push_back(stats.current_best);
		stats.current_best = {};
	}

	bool Parameters::invalid_state() const
	{
		const bool sigma_out_of_bounds = 1e-16 > mutation->sigma or mutation->sigma > 1e4;

		if (sigma_out_of_bounds && settings.verbose)
		{
			std::cout << "sigma out of bounds: " << mutation->sigma << " restarting\n";
		}
		return sigma_out_of_bounds;
	}

	void Parameters::adapt()
	{
		adaptation->adapt_evolution_paths(pop, weights, mutation, stats, mu, lambda);
		mutation->adapt(weights, adaptation, pop, old_pop, stats, lambda);

		auto successfull_adaptation = adaptation->adapt_matrix(weights, settings.modules, pop, mu, settings);
		if (!successfull_adaptation or invalid_state())
			perform_restart();

		old_pop = pop;
		restart->evaluate(*this);

		stats.t++;
	}
}

std::ostream& operator<<(std::ostream& os, const parameters::Stats& s)
{
	return os
		<< "Stats"
		<< " g=" << s.t
		<< " evals=" << s.evaluations
		<< " best=" << s.global_best;
}
