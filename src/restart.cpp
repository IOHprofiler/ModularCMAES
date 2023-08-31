#include "restart.hpp"
#include "parameters.hpp"

#include <algorithm>



namespace restart {
	//! max - min, for the last n elements of a vector
	double ptp_tail(const std::vector<double>& v, const size_t n) {
		double min = *std::min_element(v.end() - n, v.end());
		double max = *std::max_element(v.end() - n, v.end());
		return max - min;
	}

	// TODO: this is duplicate code
	double median(const Vector& x)
	{
		if (x.size() % 2 == 0)
			return (x(x.size() / 2) + x(x.size() / 2 - 1)) / 2.0;
		return x(x.size() / 2);
	}

	double median(const std::vector<double>& v, const size_t from, const size_t to) {
		const size_t n = to - from;
		if (n % 2 == 0)
			return (v[from + (n / 2)] + v[from + (n / 2) - 1]) / 2.0;
		return v[from + (n / 2)];
	}
		
	void Restart::evaluate(parameters::Parameters& p)  {
		
		flat_fitnesses(p.stats.t % p.dim) = p.pop.f(0) == p.pop.f(flat_fitness_index);
		median_fitnesses.push_back(median(p.pop.f));
		best_fitnesses.push_back(p.pop.f(0));

		if (termination_criteria(p)) {
			restart(p);
			setup(p.dim, p.lambda, p.stats.t);
		}
	}

	bool Restart::termination_criteria(const parameters::Parameters& p) const {
		const size_t time_since_restart = p.stats.t - last_restart;
		
		const bool exeeded_max_iter = max_iter < time_since_restart;
		const bool no_improvement = time_since_restart > n_bin and ptp_tail(best_fitnesses, n_bin) == 0;
		const bool flat_fitness = time_since_restart > p.dim and flat_fitnesses.sum() > p.dim / 3;

		const size_t pt = static_cast<size_t>(0.3 * time_since_restart);
		const bool stagnation = time_since_restart > n_stagnation and (
			(median(best_fitnesses, pt, time_since_restart) >= median(best_fitnesses, 0, pt))
			and (median(median_fitnesses, pt, time_since_restart) >= median(median_fitnesses, 0, pt))
		);

		// TODO: the more compilicated criteria
		if (exeeded_max_iter or no_improvement or flat_fitness or stagnation) {
			if (p.verbose) {
				std::cout << "restarting: " << p.stats.t << " (";
				std::cout << time_since_restart;
				std::cout << ") flat_fitness: " << flat_fitness;
				std::cout << " exeeded_max_iter: " << exeeded_max_iter;
				std::cout << " no_improvement: " << no_improvement;
				std::cout << " stagnation: " << stagnation << std::endl;
			}
			return true;
		}
			
		return false;
	}
	
	void Restart::restart(parameters::Parameters& p) {
		p.perform_restart(); 
	}

	void IPOP::restart(parameters::Parameters& p) {
		//max_lambda_ = (self.d * self.lambda_) * *2
		if (p.mu < 512) {
			p.mu *= ipop_factor;
			p.lambda *= ipop_factor;
		}		
		p.perform_restart();
	}
	void BIPOP::restart(parameters::Parameters& p) {
		static std::uniform_real_distribution<> dist;

		const auto last_used_budget = p.stats.evaluations - used_budget;
		used_budget += last_used_budget;
		const auto remaining_budget = budget - used_budget;
		
		if (!lambda_large) {
			lambda_large = lambda_init * 2;
			budget_small = remaining_budget / 2;
			budget_large = remaining_budget - budget_small;
		}
		else if (large()){
			budget_large -= last_used_budget;
			lambda_large *= 2;
		}
		else {
			budget_small -= last_used_budget;
		}

		lambda_small = static_cast<size_t>(std::floor(
			static_cast<double>(lambda_init) * std::pow(.5 / static_cast<double>(lambda_large) / lambda_init, 
				std::pow(dist(rng::GENERATOR), 2))));

		if (lambda_small % 2 != 0)
			lambda_small++;

		p.lambda = std::max(size_t{ 2 }, large() ? lambda_large : lambda_small);
		p.mu = std::max(1.0, p.lambda * mu_factor);
		p.perform_restart(large() ? 2. : 2e-2 * dist(rng::GENERATOR));
	}
}