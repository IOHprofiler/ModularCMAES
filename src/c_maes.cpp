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
	if (p->stats.t % (p->settings.dim * 2) == 0 and p->settings.verbose)
		std::cout << p->stats << " (mu, lambda, sigma): " << p->mu
					<< ", " << p->lambda << ", " << p->mutation->sigma << std::endl;
	return !break_conditions();
}

void ModularCMAES::operator()(std::function<double(Vector)> objective)
{
	while (step(objective));

	if (p->settings.verbose)
		std::cout << p->stats << std::endl;
}

bool ModularCMAES::break_conditions() const
{
	const auto target_reached = p->settings.target and p->settings.target.value() >= p->stats.fopt;
	const auto budget_used_up = p->stats.evaluations >= p->settings.budget;
	const auto exceed_gens = p->settings.max_generations and p->stats.t >= p->settings.max_generations;
	return exceed_gens or target_reached or budget_used_up;
}
