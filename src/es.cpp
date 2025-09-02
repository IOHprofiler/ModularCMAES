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

            const auto mask = corrector->is_out_of_bounds(x1);
            if (mask.any())
                x1 = corrector->correct_x(x1, mask, sigma);

        } while (rejection_sampling && n_rej++ < 5*d && bounds::any_out_of_bounds(x1, corrector->lb, corrector->ub) );
        return x1;
    }

    void OnePlusOneES::step(FunctionType &objective)
    {
        const auto x1 = sample();
        const auto f1 = objective(x1);
        const bool has_improved = f1 < f;
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

            const auto mask = corrector->is_out_of_bounds(x);
            if (mask.any())
                x = corrector->correct_x(x, mask, si.mean());

        } while (rejection_sampling && n_rej++ < 5*d && bounds::any_out_of_bounds(x, corrector->lb, corrector->ub));
        return x;
    }

    void MuCommaLambdaES::step(FunctionType &objective)
    {
        static sampling::Gaussian g_sigma_sampler(1);

        for (size_t i = 0; i < lambda; i++)
        {
            const Float psi_k = std::exp(tau * g_sigma_sampler()[0]);
            const Vector psi_kv = (tau_i * (*sigma_sampler)()).array().exp().matrix();
            S.col(i) = sigma.array() * psi_kv.array() * psi_k;
            X.col(i) = sample(S.col(i));
            f(i) = objective(X.col(i));
            e++;
        }
        const auto idx = utils::sort_indexes(f);
        X = X(Eigen::all, idx).eval();
        S = S(Eigen::all, idx).eval();
        f = f(idx).eval();

        if (f[0] < f_min)
        {
            f_min = f[0];
            x_min = X.col(0);
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
        while (e < budget && f_min > target)
            step(objective);
    }
}
