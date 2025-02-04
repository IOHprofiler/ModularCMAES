#pragma once

#include "sampling.hpp"
#include "stats.hpp"
#include "bounds.hpp"

namespace es
{
    struct OnePlusOneES
    {
        OnePlusOneES(
            const size_t d,
            const Vector &x0,
            const double f0,
            const double sigma0,
            const size_t budget,
            const double target,
            const parameters::Modules &modules)
            : d(d), sigma(sigma0), decay(1.0 / std::sqrt(static_cast<double>(d) + 1)),
              x(x0), f(f0), t(1), budget(budget), target(target),
              rejection_sampling(modules.bound_correction == parameters::CorrectionMethod::RESAMPLE),
              sampler(sampling::get(d, modules, 1)),
              corrector(bounds::get(modules.bound_correction, Vector::Ones(d) * -5.0, Vector::Ones(d) * 5.0))
        {
        }

        Vector sample();
        void step(FunctionType &objective);
        void operator()(FunctionType &objective);

        size_t d;
        double sigma;
        double decay;
        Vector x;
        double f;
        size_t t;
        size_t budget;
        double target;
        bool rejection_sampling;

        std::shared_ptr<sampling::Sampler> sampler;
        std::shared_ptr<bounds::BoundCorrection> corrector;
    };

    struct MuCommaLambdaES
    {

        MuCommaLambdaES(
            const size_t d,
            const Vector &x0,
            const double sigma0, 
            const size_t budget,
            const double target,
            const parameters::Modules &modules)
            : d(d), lambda(d * 5), mu(std::floor(lambda / 4)),
              tau(1.0 / std::sqrt(static_cast<double>(d))), // v1 ->
              tau_i(1.0 / pow(static_cast<double>(d), .25)), 
              mu_inv(1.0 / mu),
              m(x0), sigma(sigma0 * Vector::Ones(d)),
              f(Vector::Constant(lambda, std::numeric_limits<double>::infinity())),
              X(d, lambda), S(d, lambda),
              f_min(std::numeric_limits<double>::infinity()),
              x_min(Vector::Constant(d, std::numeric_limits<double>::signaling_NaN())),
              t(0), e(0), budget(budget), target(target),
              sampler(sampling::get(d, modules, lambda)),
              sigma_sampler(std::make_shared<sampling::Gaussian>(d)),
              rejection_sampling(modules.bound_correction == parameters::CorrectionMethod::RESAMPLE),
              corrector(bounds::get(modules.bound_correction, Vector::Ones(d) * -5.0, Vector::Ones(d) * 5.0))
        {
            // tau = 1.0 / sampler->expected_length();
            // tau_i = 1.0 / std::sqrt(sampler->expected_length());
        }

        Vector sample(const Vector si);
        void step(FunctionType &objective);
        void operator()(FunctionType &objective);

        size_t d;
        size_t lambda;
        size_t mu;
        double tau;
        double tau_i;
        double mu_inv;

        Vector m;
        Vector sigma;
        Vector f;
        Matrix X;
        Matrix S;

        double f_min;
        Vector x_min;
        size_t t;
        size_t e;
        size_t budget;
        double target;

        std::shared_ptr<sampling::Sampler> sampler;
        std::shared_ptr<sampling::Sampler> sigma_sampler;

        bool rejection_sampling;
        std::shared_ptr<bounds::BoundCorrection> corrector;
    };
}