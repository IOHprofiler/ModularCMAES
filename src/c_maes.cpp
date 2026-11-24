#include "c_maes.hpp"

void ModularCMAES::recombine() const
{
    auto &a = *p->adaptation;
    a.m_old = a.m;
    // m = m_old + (X_mu - m_old) w; the difference is evaluated into a buffer (Eigen evaluated it
    // into a temporary for the product), and m_old + p == p + m_old
    // m = m_old + (X_mu - m_old) w, evaluated as Eigen does (m = m_old; m += product, the
    // difference in a buffer as Eigen's temporary), without allocating
    // m = m_old + (X_mu - m_old) w. Eigen evaluated this product (its left-hand side is an
    // expression, not a matrix) as dst = m_old; dst += w_k (x_k - m_old) for k = 0..mu-1; the same
    // loop, without the temporary, gives the same bits.
    a.m = a.m_old;
    for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(p->mu); ++k)
        a.m.noalias() += (1.0 * p->weights.positive(k)) * (p->pop.X.col(k) - a.m_old);
}

void ModularCMAES::mutate(FunctionType &objective) const
{
    p->start(objective);
    p->mutation->mutate(objective, p->lambda, *p);
}

void ModularCMAES::select() const { p->selection->select(*p); }

void ModularCMAES::adapt() const { p->adapt(); }

bool ModularCMAES::step(FunctionType &objective) const
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

void ModularCMAES::operator()(FunctionType &objective) const
{
    while (step(objective))
        ;

    // if (p->settings.verbose)
    // 	std::cout << p->stats << '\n';
}

bool ModularCMAES::break_conditions() const
{
    const auto target_reached =
        p->settings.target and p->settings.target.value() >= p->stats.global_best.y;
    const auto budget_used_up = (p->stats.evaluations + p->lambda) >= p->settings.budget;
    const auto exceed_gens =
        p->settings.max_generations and p->stats.t >= p->settings.max_generations;
    const auto restart_strategy_criteria =
        p->settings.modules.restart_strategy == parameters::RestartStrategyType::STOP and
        p->criteria.any();
    return exceed_gens or target_reached or budget_used_up or restart_strategy_criteria;
}
