#include "c_maes.hpp"

void ModularCMAES::recombine() const
{
	p->adaptation->m_old = p->adaptation->m;
	p->adaptation->m = p->adaptation->m_old + (
		(p->pop.X.leftCols(p->mu).colwise() - p->adaptation->m_old) 
		* p->weights.positive
	);
}

void ModularCMAES::mutate(FunctionType &objective) const
{
	p->start(objective);
	p->mutation->mutate(objective, p->lambda, *p);
}

void ModularCMAES::select() const 
{
	p->selection->select(*p);
}

void ModularCMAES::adapt() const 
{
	p->adapt();
}

bool ModularCMAES::step(FunctionType& objective) const
{
	mutate(objective);
	select();
	recombine();
	adapt();
	// if (p->stats.t % (p->settings.dim * 2) == 0 and p->settings.verbose)
	// 	std::cout << p->stats << " (mu, lambda, sigma): " << p->mu
	// 		<< ", " << p->lambda << ", " << p->mutation->sigma << '\n';
	return !break_conditions();
}

void ModularCMAES::operator()(FunctionType& objective) const
{
	while (step(objective));

	// if (p->settings.verbose)
	// 	std::cout << p->stats << '\n';
}

bool ModularCMAES::break_conditions() const
{
	const auto target_reached = p->settings.target and p->settings.target.value() >= p->stats.global_best.y;
	const auto budget_used_up = p->stats.evaluations >= p->settings.budget;
	const auto exceed_gens = p->settings.max_generations and p->stats.t >= p->settings.max_generations;
	const auto restart_strategy_criteria = p->settings.modules.restart_strategy == parameters::RestartStrategyType::STOP
		and p->criteria.any();
	return exceed_gens or target_reached or budget_used_up or restart_strategy_criteria;
}
