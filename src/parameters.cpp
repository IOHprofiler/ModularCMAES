#include "parameters.hpp"

namespace parameters
{
	Parameters::Parameters(const Settings& settings) : lambda(settings.lambda0),
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
			settings.cs)),
		selection(std::make_shared<selection::Strategy>(settings.modules)),
		restart(restart::get(
			settings.modules.restart_strategy,
			settings.sigma0,
			static_cast<double>(settings.dim),
			static_cast<double>(settings.lambda0),
			static_cast<double>(settings.mu0),
			settings.budget)),
		bounds(bounds::get(settings.modules.bound_correction, settings.lb, settings.ub)),
		repelling(repelling::get(settings.modules)),
		center_placement(center::get(settings.modules.center_placement))
	{
	}

	Parameters::Parameters(const size_t dim) : Parameters(Settings(dim, {}))
	{
	}

	void Parameters::finalize_restart(FunctionType& objective)
	{
		stats.evaluations++;
		stats.centers.emplace_back(adaptation->m, objective(adaptation->m), stats.t, stats.evaluations);
		stats.update_best(stats.centers.back().x, stats.centers.back().y);
		repelling->update_archive(objective, *this);
	}

	void Parameters::perform_restart()
	{
		const auto sigma = restart->get_sigma0(*this);
		restart->update_parameters(*this);
		stats.solutions.push_back(stats.current_best);

		weights = Weights(settings.dim, mu, lambda, settings);
		sampler->reset(settings.modules, lambda);

		pop = Population(settings.dim, lambda);
		old_pop = Population(settings.dim, lambda);

		mutation = mutation::get(settings.modules, mu, weights.mueff,
			static_cast<double>(settings.dim), sigma, settings.cs);

		adaptation->restart(settings);
		(*center_placement)(*this);
		restart->criteria = restart::RestartCriteria(sigma, settings.dim, lambda, stats.t);
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

	bool Parameters::adapt()
	{
		adaptation->adapt_evolution_paths(pop, weights, mutation, stats, mu, lambda);
		mutation->adapt(weights, adaptation, pop, old_pop, stats, lambda);
		old_pop = pop;
		stats.t++;

		const auto successful_adaptation = adaptation->adapt_matrix(weights, settings.modules, pop, mu, settings);
		const auto restart_criteria_met = restart->criteria(*this);
		const auto should_restart = !successful_adaptation or invalid_state() or restart_criteria_met;

		return should_restart;

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
