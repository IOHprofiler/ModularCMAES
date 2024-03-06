#pragma once

#include "common.hpp"
#include "modules.hpp"

namespace parameters
{
	struct Parameters;
}

namespace restart
{
	class RestartCriteria
	{
		void update(const parameters::Parameters &p);

	public:
		size_t last_restart;
		size_t max_iter;
		size_t max_flat_fitness;
		size_t n_bin;
		size_t n_stagnation;
		size_t flat_fitness_index;

		Eigen::Array<int, Eigen::Dynamic, 1> flat_fitnesses;
		std::vector<double> median_fitnesses;
		std::vector<double> best_fitnesses;

		size_t time_since_restart;
		double recent_improvement;
		size_t n_flat_fitness;
		double d_sigma;

		double tolx_condition;
		Vector tolx_vector;

		double root_max_d;
		double condition_c;
		Vector effect_coord;
		Vector effect_axis;

		bool any = false;

		RestartCriteria(const double d, const double lambda, const size_t t)
			: last_restart(t),
			  max_iter(static_cast<size_t>(100 + 50 * std::pow((d + 3), 2.0) / std::sqrt(lambda))),
			  max_flat_fitness(static_cast<size_t>(std::ceil(d / 3))),
			  n_bin(10 + static_cast<size_t>(std::ceil(30 * d / lambda))),
			  n_stagnation(static_cast<size_t>(std::min(static_cast<int>(120 + (30 * d / lambda)), 20000))),
			  flat_fitness_index(static_cast<size_t>(std::round(.1 + lambda / 4))),
			  flat_fitnesses(Eigen::Array<int, Eigen::Dynamic, 1>::Constant(static_cast<size_t>(d), 0)),
			  median_fitnesses{},
			  best_fitnesses{},
			  time_since_restart(0),
			  recent_improvement(0.),
			  n_flat_fitness(0),
			  d_sigma(0.),
			  tolx_condition(0.),
			  tolx_vector{static_cast<size_t>(d * 2)},
			  root_max_d(0.),
			  condition_c(0.),
			  effect_coord(static_cast<size_t>(d)),
			  effect_axis(static_cast<size_t>(d))
		{ 
			median_fitnesses.reserve(max_iter);
			best_fitnesses.reserve(max_iter);
		}

		bool exceeded_max_iter() const;

		bool no_improvement() const;

		bool flat_fitness() const;

		bool tolx() const;

		bool tolupsigma() const;

		bool conditioncov() const;

		bool noeffectaxis() const;

		bool noeffectcoor() const;

		bool stagnation() const;

		bool operator()(const parameters::Parameters &p);
	};


	struct Strategy
	{
		RestartCriteria criteria;

		Strategy(const double d, const double lambda) : criteria{ d, lambda, 0 } {}
		
		void evaluate(parameters::Parameters &p);
			
		virtual void restart(parameters::Parameters &) = 0;
	};

	struct None : Strategy
	{
		using Strategy::Strategy;
		void restart(parameters::Parameters &p) override {}
	};

	struct Stop : Strategy
	{
		using Strategy::Strategy;
		void restart(parameters::Parameters &p) override {}
	};

	struct Restart : Strategy
	{
		using Strategy::Strategy;
		void restart(parameters::Parameters &) override;
	};

	struct IPOP : Strategy
	{
		double ipop_factor = 2.0;
		using Strategy::Strategy;
		void restart(parameters::Parameters &) override;
	};

	struct BIPOP : Strategy
	{

		size_t lambda_init;
		double mu_factor;
		size_t budget;

		size_t lambda_large = 0;
		size_t lambda_small = 0;
		size_t budget_small = 0;
		size_t budget_large = 0;
		size_t used_budget = 0;

		BIPOP(const double d, const double lambda, const double mu, const size_t budget) : Strategy(d, lambda), lambda_init(static_cast<size_t>(lambda)), mu_factor(mu / lambda), budget(budget)
		{
		}

		void restart(parameters::Parameters &) override;

		bool large() const
		{
			return budget_large >= budget_small and budget_large > 0;
		}
	};

	inline std::shared_ptr<Strategy> get(const parameters::RestartStrategyType s, const double d, const double lambda, const double mu, const size_t budget)
	{
		using namespace parameters;
	switch (s) 
		{
		case RestartStrategyType::RESTART:
			return std::make_shared<Restart>(d, lambda);
		case RestartStrategyType::IPOP:
			return std::make_shared<IPOP>(d, lambda);
		case RestartStrategyType::BIPOP:
			return std::make_shared<BIPOP>(d, lambda, mu, budget);
		case RestartStrategyType::STOP:
			return std::make_shared<Stop>(d, lambda);
		default:
		case RestartStrategyType::NONE:
			return std::make_shared<None>(d, lambda);
		}
	}
}