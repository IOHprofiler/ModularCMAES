#pragma once

#include "common.hpp"
#include "modules.hpp"

namespace parameters
{
    struct Parameters;
}

namespace center
{
    struct Placement
    {
        virtual void operator()(parameters::Parameters &p) = 0;
    };

    struct X0 : Placement
    {
        void operator()(parameters::Parameters &p) override;
    };

    struct Uniform : Placement
    {
        void operator()(parameters::Parameters &p) override;
    };

    struct Zero : Placement
    {
        void operator()(parameters::Parameters &p) override;
    };


    inline std::shared_ptr<Placement> get(const parameters::CenterPlacement &p)
    {

        using namespace parameters;
        switch (p)
        {
        case CenterPlacement::UNIFORM:
            return std::make_shared<Uniform>();
        case CenterPlacement::ZERO:
            return std::make_shared<Zero>();
        default:
        case CenterPlacement::X0:
            return std::make_shared<X0>();
        }
    }
}