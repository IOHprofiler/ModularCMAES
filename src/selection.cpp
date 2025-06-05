#include "selection.hpp"
#include "parameters.hpp"


namespace selection
{
	Strategy::Strategy(const parameters::Modules& modules) :
		pairwise(modules.mirrored == parameters::Mirror::PAIRWISE
			         ? std::make_shared<Pairwise>()
			         : std::make_shared<NoPairwise>()),
		elitsm(modules.elitist ? std::make_shared<Elitsm>() : std::make_shared<NoElitsm>())
	{
	}

	void Strategy::select(parameters::Parameters& p) const
	{
		(*pairwise)(p);

		(*elitsm)(p);

		p.pop.sort();
		p.pop.resize_cols(p.lambda);

		p.stats.current_avg = p.pop.f.array().mean();
		p.stats.update_best(p.pop.X(Eigen::all, 0), p.pop.f(0));
	}

	void Pairwise::operator()(parameters::Parameters& p) const
	{
		assert(p.pop.f.size() % 2 == 0);
		for (size_t i = 0; i < static_cast<size_t>(p.pop.f.size()); i += 2)
		{
			Eigen::Index idx = i + (1 * (p.pop.f(i) < p.pop.f(i + 1)));
			p.pop.f(idx) = std::numeric_limits<Float>::infinity();
		}
	}

	void Elitsm::operator()(parameters::Parameters& p) const
	{
		if (p.stats.t != 0)
		{
			p.old_pop.resize_cols(k);

			if (!p.settings.one_plus_one)
			{
				for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(p.old_pop.n); i++)
				{
					p.old_pop.Y.col(i).noalias() = p.adaptation->invert_x(p.old_pop.X.col(i), p.old_pop.s(i));
					p.old_pop.Z.col(i).noalias() = p.adaptation->invert_y(p.old_pop.Y.col(i));
				}
			}
			p.pop += p.old_pop;
		}
	}
}
