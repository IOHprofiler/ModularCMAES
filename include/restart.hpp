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
		Float sigma0;
		size_t last_restart;
		size_t max_iter;
		size_t max_flat_fitness;
		size_t n_bin;
		size_t n_stagnation;
		size_t flat_fitness_index;

		Eigen::Array<int, Eigen::Dynamic, 1> flat_fitnesses;
		std::vector<Float> median_fitnesses;
		std::vector<Float> best_fitnesses;

		size_t time_since_restart;
		Float recent_improvement;
		size_t n_flat_fitness;
		Float d_sigma;

		Float tolx_condition;
		Vector tolx_vector;

		Float root_max_d;
		Float condition_c;
		Vector effect_coord;
		Vector effect_axis;

		bool any = false;

		RestartCriteria(const Float sigma0, const Float d, const Float lambda, const size_t t)
			: sigma0(sigma0),
			  last_restart(t),
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

		bool min_sigma() const;

		bool operator()(const parameters::Parameters &p);
	};

	struct Strategy
	{
		RestartCriteria criteria;

		Strategy(const Float sigma0, const Float d, const Float lambda) : criteria{sigma0, d, lambda, 0} {}

		void evaluate(FunctionType& objective, parameters::Parameters &p);

		virtual void restart(FunctionType& objective, parameters::Parameters &) = 0;
	};

	struct None : Strategy
	{
		using Strategy::Strategy;
		void restart(FunctionType& objective, parameters::Parameters &p) override {}
	};

	struct Stop : Strategy
	{
		using Strategy::Strategy;
		void restart(FunctionType& objective, parameters::Parameters &p) override {}
	};

	struct Restart : Strategy
	{
		using Strategy::Strategy;
		void restart(FunctionType& objective, parameters::Parameters &) override;
	};

	struct IPOP : Strategy
	{
		Float ipop_factor = 2.0;
		using Strategy::Strategy;
		void restart(FunctionType& objective, parameters::Parameters &) override;
	};

	struct BIPOP : Strategy
	{

		size_t lambda_init;
		Float mu_factor;
		size_t budget;

		size_t lambda_large = 0;
		size_t lambda_small = 0;
		size_t budget_small = 0;
		size_t budget_large = 0;
		size_t used_budget = 0;

		BIPOP(
			const Float sigma0, const Float d, const Float lambda, const Float mu, const size_t budget) : Strategy(sigma0, d, lambda), lambda_init(static_cast<size_t>(lambda)), mu_factor(mu / lambda), budget(budget)
		{
		}

		void restart(FunctionType& objective, parameters::Parameters &) override;

		bool large() const
		{
			return budget_large >= budget_small and budget_large > 0;
		}
	};

	inline std::shared_ptr<Strategy> get(const parameters::RestartStrategyType s, const Float sigma0, const Float d, const Float lambda, const Float mu, const size_t budget)
	{
		using namespace parameters;
		switch (s)
		{
		case RestartStrategyType::RESTART:
			return std::make_shared<Restart>(sigma0, d, lambda);
		case RestartStrategyType::IPOP:
			return std::make_shared<IPOP>(sigma0, d, lambda);
		case RestartStrategyType::BIPOP:
			return std::make_shared<BIPOP>(sigma0, d, lambda, mu, budget);
		case RestartStrategyType::STOP:
			return std::make_shared<Stop>(sigma0, d, lambda);
		default:
		case RestartStrategyType::NONE:
			return std::make_shared<None>(sigma0, d, lambda);
		}
	}
}