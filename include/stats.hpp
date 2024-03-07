#pragma once

#include "common.hpp"

namespace parameters
{
	struct Stats
	{
		size_t t = 0;
		size_t evaluations = 0;

		double current_avg = std::numeric_limits<double>::infinity();
		std::vector<Solution> solutions = {};
		Solution current_best = {};
		Solution global_best = {};
	};
}
