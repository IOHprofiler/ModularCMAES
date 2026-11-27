#include "es.hpp"
#include "bounds.hpp"

namespace es
{

Vector OnePlusOneES::sample()
{
    size_t n_rej = 0;
    Vector x1;
    do
    {
        const Vector z = (*sampler)();
        x1 = x + sigma * z;

        const auto mask = corrector->is_out_of_bounds(x1, settings);
        if (mask.any())
            x1 = corrector->correct_x(x1, mask, sigma, settings);

    } while (rejection_sampling && n_rej++ < 5 * d &&
             bounds::any_out_of_bounds(x1, settings.lb, settings.ub));
    return x1;
}

void OnePlusOneES::step(FunctionType &objective)
{
    const auto x1 = sample();
    const auto f1 = objective(x1);
    // x is the parent and the best point: only a point in the domain (#168; with RESAMPLE a point
    // can stay outside the box after 5 d tries) replaces it
    const bool has_improved = f1 < f && settings.in_domain(x1);
    sigma *= pow(std::exp(static_cast<Float>(has_improved) - 0.2), decay);
    if (has_improved)
    {
        x = x1;
        f = f1;
    }
    t++;
}
void OnePlusOneES::operator()(FunctionType &objective)
{
    while (t < budget && f > target)
        step(objective);
}

Vector MuCommaLambdaES::sample(const Vector si)
{
    size_t n_rej = 0;
    Vector x;
    do
    {
        const Vector z = (*sampler)();
        x = m.array() + (si.array() * z.array());

        const auto mask = corrector->is_out_of_bounds(x, settings);
        if (mask.any())
            x = corrector->correct_x(x, mask, si.mean(), settings);

    } while (rejection_sampling && n_rej++ < 5 * d &&
             bounds::any_out_of_bounds(x, settings.lb, settings.ub));
    return x;
}

void MuCommaLambdaES::step(FunctionType &objective)
{
    for (size_t i = 0; i < lambda; i++)
    {
        const Float psi_k = std::exp(tau * (*global_sigma_sampler)()[0]);
        const Vector psi_kv = (tau_i * (*sigma_sampler)()).array().exp().matrix();
        S.col(i) = sigma.array() * psi_kv.array() * psi_k;
        X.col(i) = sample(S.col(i));
        f(i) = objective(X.col(i));
        e++;
    }
    // NaN last, ties in index order (#117: a comparison with < is undefined for NaN)
    std::vector<size_t> idx;
    utils::sort_order(f, idx);
    X = X(Eigen::all, idx).eval();
    S = S(Eigen::all, idx).eval();
    f = f(idx).eval();

    // The best point so far: only a point in the domain (#168), the best of the generation that
    // lies in it
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(lambda); ++i)
        if (settings.in_domain(X.col(i)))
        {
            if (f[i] < f_min)
            {
                f_min = f[i];
                x_min = X.col(i);
            }
            break;
        }

    sigma.setZero();
    m.setZero();

    for (size_t i = 0; i < mu; i++)
    {
        m += mu_inv * X.col(i);
        sigma += mu_inv * S.col(i);
    }
    t++;
}

void MuCommaLambdaES::operator()(FunctionType &objective)
{
    // A generation (lambda evaluations) runs only when it fits the budget (#117)
    while (e + lambda <= budget && f_min > target)
        step(objective);
}
} // namespace es
