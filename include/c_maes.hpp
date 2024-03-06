#pragma once

#include "parameters.hpp"

struct ModularCMAES
{
    std::shared_ptr<parameters::Parameters> p;

    ModularCMAES(const std::shared_ptr<parameters::Parameters> p) : p(p) {}

    void recombine() const;

    bool step(FunctionType& objective) const;

    void operator()(FunctionType& objective) const;
     
    bool break_conditions() const;
};

