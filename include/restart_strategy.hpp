#pragma once

#include "common.hpp"
#include "modules.hpp"
#include "restart_criteria.hpp"

namespace parameters
{
	struct Parameters;
}

namespace restart
{
	struct Strategy
	{
		virtual Float update(parameters::Parameters &p);
	};

	struct IPOP : Strategy
	{
		Float ipop_factor = 2.0;
		Float update(parameters::Parameters &) override;
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
			const Float lambda, const Float mu, const size_t budget) : 
				Strategy(), 
				lambda_init(static_cast<size_t>(lambda)), 
				mu_factor(mu / lambda), 
				budget(budget)
		{
		}

		Float update(parameters::Parameters &) override;

		bool large() const
		{
			return budget_large >= budget_small and budget_large > 0;
		}
	};

	namespace strategy {
		inline std::shared_ptr<Strategy> get(
			const parameters::Modules modules,
			const Float lambda, 
			const Float mu, 
			const size_t budget
		)
		{
			using namespace parameters;
			switch (modules.restart_strategy)
			{
			case RestartStrategyType::IPOP:
				return std::make_shared<IPOP>();
			case RestartStrategyType::BIPOP:
				return std::make_shared<BIPOP>(lambda, mu, budget);
			default:
			case RestartStrategyType::STOP:
			case RestartStrategyType::NONE:
			case RestartStrategyType::RESTART:
				return std::make_shared<Strategy>();
			}
		}
	}
}