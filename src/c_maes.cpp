#include "c_maes.hpp"


void ModularCMAES::recombine()
{
	p->dynamic.m_old = p->dynamic.m;
	p->dynamic.m = p->dynamic.m_old + ((p->pop.X.leftCols(p->mu).colwise() - p->dynamic.m_old) * p->weights.positive);
}

bool ModularCMAES::step(std::function<double(Vector)> objective)
{
	p->mutation->mutate(objective, p->pop.Z.cols(), *p);
	p->selection->select(*p);
	recombine();
	p->adapt();
	return !break_conditions();
}

void ModularCMAES::operator()(std::function<double(Vector)> objective)
{
	while (step(objective))
	{
		if (p->stats.t % (p->dim * 2) == 0 and p->verbose)
			std::cout << p->stats << " (mu, lambda, sigma): " << p->mu
					  << ", " << p->lambda << ", " << p->mutation->sigma << std::endl;
	}
	if (p->verbose)
		std::cout << p->stats << std::endl;
}

bool ModularCMAES::break_conditions() const
{
	const auto target_reached = p->stats.target >= p->stats.fopt;
	const auto budget_used_up = p->stats.evaluations >= p->stats.budget;
	const auto exceed_gens = p->stats.t >= p->stats.max_generations;

	return exceed_gens or target_reached or budget_used_up;
}

