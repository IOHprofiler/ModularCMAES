#pragma once

#include "common.hpp"

namespace parameters
{
	struct Stats
	{
		size_t t = 0;
		size_t evaluations = 0;

		Float current_avg = std::numeric_limits<Float>::infinity();
		std::vector<Solution> solutions = {};
		std::vector<Solution> centers = {};
		Solution current_best = {};
		Solution global_best = {};
		bool has_improved = false;
		Float success_ratio = 2.0 / 11.0;
		Float cp = 1.0 / 12.0;
		size_t last_update = 0;
		size_t n_updates = 0;

		void update_best(const Vector &x, const Float y)
		{
			has_improved = false;
			if (y < current_best.y)
			{
				current_best = Solution(x, y, t, evaluations);

				if (current_best < global_best)
					global_best = current_best;

				has_improved = true;
			}
			success_ratio = (1 - cp) * success_ratio + (cp * has_improved);
		}
	};
}
