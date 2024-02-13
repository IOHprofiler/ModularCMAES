#pragma once

#include "bounds.hpp"
#include "mutation.hpp"
#include "population.hpp"
#include "matrix_adaptation.hpp"
#include "restart.hpp"
#include "sampling.hpp"
#include "stats.hpp"
#include "selection.hpp"
#include "weights.hpp"

namespace parameters
{
    struct Parameters
    {
        size_t lambda;
        size_t mu;

        Settings settings;
        Stats stats;
        Weights weights;

        Population pop;
        Population old_pop;

        std::shared_ptr<matrix_adaptation::Adaptation> adaptation;
        std::shared_ptr<sampling::Sampler> sampler;
        std::shared_ptr<mutation::Strategy> mutation;
        std::shared_ptr<selection::Strategy> selection;
        std::shared_ptr<restart::Strategy> restart;
        std::shared_ptr<bounds::BoundCorrection> bounds;

        Parameters(const size_t dim);
        Parameters(const Settings &settings);

        void adapt();

        void perform_restart(const std::optional<double> &sigma = std::nullopt);

        bool invalid_state() const;
    };

}

std::ostream &operator<<(std::ostream &os, const parameters::Stats &dt);