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
		std::vector<Solution> centers = {};
		Solution current_best = {};
		Solution global_best = {};
		bool has_improved = false;

		void update_best(const Vector &x, const double y)
		{
			has_improved = false;
			if (y < current_best.y)
			{
				current_best = Solution(x, y, t, evaluations);

				if (current_best < global_best)
					global_best = current_best;

				has_improved = true;
			}

		}
	};
}
