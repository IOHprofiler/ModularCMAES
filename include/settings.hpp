#pragma once

#include "common.hpp"
#include "modules.hpp"

namespace parameters
{
    struct Settings
    {
        size_t dim;
        Modules modules;

        std::optional<double> target;
        std::optional<size_t> max_generations;
        size_t budget;

        double sigma0;
        size_t lambda0;
        size_t mu0;

        std::optional<Vector> x0;
        Vector lb;
        Vector ub;
        std::optional<double> cs;
        std::optional<double> cc;
        std::optional<double> cmu;
        std::optional<double> c1;
        bool verbose;

        Settings(size_t dim,
                 std::optional<Modules> mod = std::nullopt,
                 std::optional<double> target = std::nullopt,
                 std::optional<size_t> max_generations = std::nullopt,
                 std::optional<size_t> budget = std::nullopt,
                 std::optional<double> sigma = std::nullopt,
                 std::optional<size_t> lambda = std::nullopt,
                 std::optional<size_t> mu = std::nullopt,
                 std::optional<Vector> x0 = std::nullopt,
                 std::optional<Vector> lb = std::nullopt,
                 std::optional<Vector> ub = std::nullopt,
                 std::optional<double> cs = std::nullopt,
                 std::optional<double> cc = std::nullopt,
                 std::optional<double> cmu = std::nullopt,
                 std::optional<double> c1 = std::nullopt,
                 bool verbose = false) : dim(dim),
                                         modules(mod.value_or(Modules())),
                                         target(target),
                                         max_generations(max_generations),
                                         budget(budget.value_or(dim * 1e4)),
                                         sigma0(sigma.value_or(2.0)),
                                         lambda0(lambda.value_or(4 + std::floor(3 * std::log(dim)))),
                                         mu0(mu.value_or(lambda0 / 2)),
                                         x0(x0),
                                         lb(lb.value_or(Vector::Ones(dim) * -5)),
                                         ub(ub.value_or(Vector::Ones(dim) * 5)),
                                         cs(cs),
                                         cc(cc),
                                         cmu(cmu),
                                         c1(c1),
                                         verbose(verbose)
        {
            if (modules.mirrored == Mirror::PAIRWISE and lambda0 % 2 != 0)
                lambda0++;

            if (mu0 > lambda0)
            {
                mu0 = lambda0 / 2;
            }
        }
    };

}