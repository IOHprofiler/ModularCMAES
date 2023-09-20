#include "selection.hpp"
#include "parameters.hpp"


namespace selection {
	Strategy::Strategy(const parameters::Modules& modules) : 
		pairwise(modules.mirrored == sampling::Mirror::PAIRWISE ? 
			std::make_shared<Pairwise>() : 
			std::make_shared<NoPairwise>()),
		elitsm(modules.elitist ? 
			std::make_shared<Elitsm>(): 
			std::make_shared<NoElitsm>())
	{}

	void Strategy::select(parameters::Parameters& p) const {
		(*pairwise)(p);
		
		(*elitsm)(p);		
		
		p.pop.sort();
		p.pop.resize_cols(p.lambda);

		if (p.pop.f(0) < p.stats.fopt)
		{
			p.stats.fopt = p.pop.f(0);
			p.stats.xopt = p.pop.X(Eigen::all, 0);
		}
	}

	void Pairwise::operator()(parameters::Parameters& p) const {
		assert(p.pop.f.size() % 2 == 0);
		for (size_t i = 0, j = 0; i < static_cast<size_t>(p.pop.f.size()); i += 2){
			size_t idx = i + (1 * (p.pop.f(i) < p.pop.f(i + 1)));
			p.pop.f(idx) = std::numeric_limits<double>::infinity();
		}
	}

	void Elitsm::operator()(parameters::Parameters& p) const {
		if (p.stats.t != 0) {
			p.old_pop.resize_cols(k);
			p.pop += p.old_pop;
		}
	}		
}