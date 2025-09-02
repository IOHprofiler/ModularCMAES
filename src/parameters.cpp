#include "parameters.hpp"

namespace parameters
{
	Parameters::Parameters(const Settings& settings) : successfull_adaptation(true),
		lambda(settings.lambda0),
		mu(settings.mu0),
		settings(settings),
		stats{},
		sampler(sampling::get(settings.dim, settings.modules, settings.lambda0)),
		weights(settings.dim, settings.mu0, settings.lambda0, settings, sampler->expected_length()),
		pop(settings.dim, settings.lambda0),
		old_pop(settings.dim, settings.lambda0),
		criteria(restart::Criteria::get(settings.modules)),
		adaptation(matrix_adaptation::get(settings.modules, settings.dim,
			settings.x0.value_or(Vector::Zero(settings.dim)),
			sampler->expected_length(),
			settings.sigma0
		)),
		mutation(mutation::get(settings.modules,
			settings.mu0,
			static_cast<Float>(settings.dim),
			settings.sigma0
		)),
		selection(std::make_shared<selection::Strategy>(settings.modules)),
		restart_strategy(restart::strategy::get(
			settings.modules,
			static_cast<Float>(settings.lambda0),
			static_cast<Float>(settings.mu0),
			settings.budget)),
		bounds(bounds::get(settings.modules.bound_correction, settings.lb, settings.ub)),
		repelling(repelling::get(settings.modules)),
		center_placement(center::get(settings.modules.center_placement))
	{
		criteria.reset(*this);
	}

	Parameters::Parameters(const size_t dim) : Parameters(Settings(dim, {}))
	{
	}

	void Parameters::perform_restart(FunctionType& objective, const std::optional<Float>& sigma)
	{
		if (settings.verbose)
		{
			std::cout << "performing restart. t(" << stats.t <<", " << stats.evaluations
					  << ") reason: " << criteria.reason() << std::endl;
			std::cout << "setting parameters (mu, lambda, sigma): (" <<
				sigma.value_or(settings.sigma0) << ", "
				<< mu << ", " << lambda << ")\n";
		}
		stats.solutions.push_back(stats.current_best);
		stats.evaluations++;
		stats.centers.emplace_back(adaptation->m, objective(adaptation->m), stats.t - 1, stats.evaluations);
		stats.update_best(stats.centers.back().x, stats.centers.back().y);
		stats.has_improved = false;
		repelling->update_archive(objective, *this);

		sampler->reset(settings.modules, lambda);
		weights = Weights(settings.dim, mu, lambda, settings, sampler->expected_length());

		pop = Population(settings.dim, lambda);
		old_pop = Population(settings.dim, lambda);

		mutation = mutation::get(settings.modules, mu,
			static_cast<Float>(settings.dim),
			sigma.value_or(settings.sigma0));
		adaptation->restart(settings, sigma.value_or(settings.sigma0));
		(*center_placement)(*this);
		criteria.reset(*this);
		stats.current_best = {};
	}

	void Parameters::adapt()
	{
		adaptation->adapt_evolution_paths(pop, weights, stats, settings, lambda, mu);
		mutation->adapt(weights, adaptation, pop, old_pop, stats, lambda);

		if (constants::clip_sigma)
			mutation->sigma = std::min(std::max(mutation->sigma, restart::MinSigma::tolerance), restart::MaxSigma::tolerance);

		successfull_adaptation = adaptation->adapt_matrix(weights, settings.modules, pop, mu, settings, stats);

		criteria.update(*this);
		stats.t++;
	}

	void Parameters::start(FunctionType& objective)
	{
		old_pop = pop;
		if (criteria.any())
		{
			const auto sig = restart_strategy->update(*this);
			perform_restart(objective, sig);
		}
	}
}

std::ostream& operator<<(std::ostream& os, const parameters::Stats& s)
{
	return os
		<< "Stats"
		<< " t=" << s.t
		<< " e=" << s.evaluations
		<< " best=" << s.global_best
		<< " improved=" << std::boolalpha << s.has_improved;
}
