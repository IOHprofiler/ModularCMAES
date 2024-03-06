#pragma once

#include "common.hpp"
#include <modules.hpp>

namespace parameters
{
	struct Parameters;
}

namespace repelling {

	struct Repelling {
		virtual ~Repelling() = default;
		virtual void operator()(parameters::Parameters& p);
	};

	struct NoRepelling final : Repelling {
		void operator()(parameters::Parameters& p) override {}
	};


	inline std::shared_ptr<Repelling> get(const parameters::Modules& m)
	{
		if (m.repelling_restart)
			return std::make_shared<Repelling>();
		return std::make_shared<NoRepelling>();
	}
}