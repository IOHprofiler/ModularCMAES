#pragma once

#include "common.hpp"

namespace parameters {
	struct Parameters;
	struct Modules;
}

namespace selection {
	struct Elitsm {
		size_t k = -1;
		virtual void operator()(parameters::Parameters& p) const;
	};

	struct NoElitsm: Elitsm {
		void operator()(parameters::Parameters& p) const override {}
	};

	struct Pairwise {
		virtual void operator()(parameters::Parameters& p) const;
	};

	struct NoPairwise: Pairwise { 
		void operator()(parameters::Parameters& p) const override {}
	};

	struct Strategy {
		std::shared_ptr<Pairwise> pairwise;
		std::shared_ptr<Elitsm> elitsm; 

		Strategy(const parameters::Modules&);

		void select(parameters::Parameters& p) const;
	};
	
}
