#include "parameters.hpp"

namespace parameters
{
	Parameters::Parameters(const Settings &settings) : lambda(settings.lambda0),
													   mu(settings.mu0),
													   settings(settings),
													   weights(settings.dim, settings.mu0, settings.lambda0, settings),
													   pop(settings.dim, settings.lambda0),
													   old_pop(settings.dim, settings.lambda0),
													   sampler(sampling::get(settings.dim, settings.modules, settings.lambda0)),
													   adaptation(matrix_adaptation::get(settings.modules, settings.dim,
																						 settings.x0.value_or(Vector::Zero(settings.dim)),
																						 sampler->expected_length())),
													   mutation(mutation::get(settings.modules,
																			  settings.mu0, weights.mueff,
																			  static_cast<double>(settings.dim),
																			  settings.sigma0,
																			  settings.cs,
																			  sampler->expected_length())),
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

	void Parameters::perform_restart(FunctionType &objective, const std::optional<double> &sigma)
	{
		stats.solutions.push_back(stats.current_best);

		stats.evaluations++;
		stats.centers.emplace_back(adaptation->m, objective(adaptation->m), stats.t, stats.evaluations);
		stats.update_best(stats.centers.back().x, stats.centers.back().y);
		stats.has_improved = false;
		repelling->update_archive(objective, *this);

		weights = Weights(settings.dim, mu, lambda, settings);
		sampler->reset(settings.modules, lambda);

		pop = Population(settings.dim, lambda);
		old_pop = Population(settings.dim, lambda);

		mutation = mutation::get(settings.modules, mu, weights.mueff,
								 static_cast<double>(settings.dim),
								 sigma.value_or(settings.sigma0),
								 settings.cs, sampler->expected_length());
		adaptation->restart(settings);
		(*center_placement)(*this);
		restart->criteria = restart::RestartCriteria(sigma.value_or(settings.sigma0), settings.dim, lambda, stats.t);
		stats.current_best = {};
	}

	bool Parameters::invalid_state() const
	{
		if (constants::clip_sigma)
			mutation->sigma = std::min(std::max(mutation->sigma, constants::lb_sigma), constants::ub_sigma);
		
		const bool sigma_out_of_bounds = constants::lb_sigma > mutation->sigma or mutation->sigma > constants::ub_sigma;

		if (sigma_out_of_bounds && settings.verbose)
		{
			std::cout << "sigma out of bounds: " << mutation->sigma << " restarting\n";
		}
		return sigma_out_of_bounds;
	}

	void Parameters::adapt(FunctionType &objective)
	{
		adaptation->adapt_evolution_paths(pop, weights, mutation, stats, mu, lambda);
		mutation->adapt(weights, adaptation, pop, old_pop, stats, lambda);

		auto successfull_adaptation = adaptation->adapt_matrix(weights, settings.modules, pop, mu, settings, stats);

		if (!successfull_adaptation or invalid_state())
			perform_restart(objective);

		old_pop = pop;
		restart->evaluate(objective, *this);
		stats.t++;
	}
}

std::ostream &operator<<(std::ostream &os, const parameters::Stats &s)
{
	return os
		   << "Stats"
		   << " t=" << s.t
		   << " e=" << s.evaluations
		   << " best=" << s.global_best
	       << " improved=" << std::boolalpha << s.has_improved
	;
}
