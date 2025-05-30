#pragma once

#include "common.hpp"
#include "modules.hpp"

namespace parameters
{
    struct Settings
    {
        size_t dim;
        Modules modules;

        std::optional<Float> target;
        std::optional<size_t> max_generations;
        size_t budget;

        Float sigma0;
        size_t lambda0;
        size_t mu0;

        std::optional<Vector> x0;
        Vector lb;
        Vector ub;
        std::optional<Float> cs;
        std::optional<Float> cc;
        std::optional<Float> cmu;
        std::optional<Float> c1;
        bool verbose;
        Float volume;

        Settings(size_t dim,
                 std::optional<Modules> mod = std::nullopt,
                 std::optional<Float> target = std::nullopt,
                 std::optional<size_t> max_generations = std::nullopt,
                 std::optional<size_t> budget = std::nullopt,
                 std::optional<Float> sigma = std::nullopt,
                 std::optional<size_t> lambda = std::nullopt,
                 std::optional<size_t> mu = std::nullopt,
                 std::optional<Vector> x0 = std::nullopt,
                 std::optional<Vector> lb = std::nullopt,
                 std::optional<Vector> ub = std::nullopt,
                 std::optional<Float> cs = std::nullopt,
                 std::optional<Float> cc = std::nullopt,
                 std::optional<Float> cmu = std::nullopt,
                 std::optional<Float> c1 = std::nullopt,
                 bool verbose = false) : dim(dim),
                                         modules(mod.value_or(Modules())),
                                         target(target),
                                         max_generations(max_generations),
                                         budget(budget.value_or(dim * 1e4)),
                                         sigma0(sigma.value_or(2.0)),
                                         lambda0(lambda.value_or(4 + std::floor(3 * std::log(dim)))),
                                         mu0(mu.value_or(lambda0 / 2)),
                                         x0(x0),
                                         lb(lb.value_or(Vector::Ones(dim) * -std::numeric_limits<double>::infinity())),
                                         ub(ub.value_or(Vector::Ones(dim) * std::numeric_limits<double>::infinity())),
                                         cs(cs),
                                         cc(cc),
                                         cmu(cmu),
                                         c1(c1),
                                         verbose(verbose),
                                         volume(0.0)
        {
            if (modules.mirrored == Mirror::PAIRWISE and lambda0 % 2 != 0)
                lambda0++;

            if (mu0 > lambda0)
            {
                mu0 = lambda0 / 2;
            }

            if (lambda0 == 1)
            {
                mu0 = 1;
                modules.elitist = true;
                modules.active = false;
                modules.weights = RecombinationWeights::EQUAL;
                modules.ssa = StepSizeAdaptation::SR;
                modules.matrix_adaptation = MatrixAdaptationType::ONEPLUSONE;
                cc = 2.0 / (static_cast<Float>(dim) + 2.0);
                c1 = 2.0 / (pow(static_cast<Float>(dim),2) + 6.0);

                if (modules.restart_strategy == RestartStrategyType::BIPOP || modules.restart_strategy == RestartStrategyType::IPOP)
                    modules.restart_strategy = RestartStrategyType::RESTART;
            }
            volume = (this->ub.cwiseMin(10 * sigma0) - this->lb.cwiseMax(-10 * sigma0)).prod();
        }
    };

}