#pragma once

#include "common.hpp"

namespace parameters {
	struct Parameters;
}

namespace restart {
	enum class StrategyType {
		NONE, RESTART, IPOP, BIPOP
	};
	 
	struct Strategy {
		virtual void evaluate(parameters::Parameters&) = 0;
	};

	struct None : Strategy {
		void evaluate(parameters::Parameters&) override {}
	};

	struct Restart : Strategy {
		size_t last_restart;

		size_t max_iter;
		size_t n_bin;
		size_t n_stagnation;
		size_t flat_fitness_index;

		Eigen::Array<int, Eigen::Dynamic, 1> flat_fitnesses;
		std::vector<double> median_fitnesses;
		std::vector<double> best_fitnesses;

		Restart(const double d, const double lambda) {
			setup(d, lambda, 0);
		}

		void setup(const double d, const double lambda, const size_t t) {
			last_restart = t;
			max_iter = static_cast<size_t>(100 + 50 * std::pow((d + 3), 2.0) / std::sqrt(lambda));
			n_bin = 10 + static_cast<size_t>(std::ceil(30 * d / lambda));
			n_stagnation = static_cast<size_t>(std::min(static_cast<int>(120 + (30 * d / lambda)), 20000));
			flat_fitness_index = static_cast<size_t>(std::round(.1 + lambda / 4));
			
			// Stats
			flat_fitnesses = Eigen::Array<int, Eigen::Dynamic, 1>::Constant(5, 0);
			
			median_fitnesses = std::vector<double>();
			median_fitnesses.reserve(max_iter);

			best_fitnesses = std::vector<double>();
			best_fitnesses.reserve(max_iter);
		}

		void evaluate(parameters::Parameters& p) override;

		bool termination_criteria(const parameters::Parameters&) const;
		virtual void restart(parameters::Parameters&);
		
	};

	struct IPOP : Restart {
		using Restart::Restart;
		double ipop_factor = 2.0;
		void restart(parameters::Parameters&) override;
	};

	struct BIPOP : Restart {
		
		size_t lambda_init;
		double mu_factor;
		size_t budget;

		size_t lambda_large = 0;
		size_t lambda_small = 0;
		size_t budget_small = 0;
		size_t budget_large = 0;
		size_t used_budget = 0;
		

		BIPOP(const double d, const double lambda, const double mu, const size_t budget): 
			Restart(d, lambda), lambda_init(static_cast<size_t>(lambda)), mu_factor(mu / lambda), budget(budget)
		{
		}

		void restart(parameters::Parameters&) override;

		bool large() const {
			return budget_large >= budget_small and budget_large > 0;
		}
	};

	inline std::shared_ptr<Strategy> get(const StrategyType s, const double d, const double lambda, const double mu, const size_t budget) {
		switch (s)
		{
		case StrategyType::RESTART:
			return std::make_shared<Restart>(d, lambda);
		case StrategyType::IPOP:
			return std::make_shared<IPOP>(d, lambda);
		case StrategyType::BIPOP:
			return std::make_shared<BIPOP>(d, lambda, mu, budget);
		default:
		case StrategyType::NONE:
			return std::make_shared<None>();
		}
	}
}