#pragma once

#include "parameters.hpp"

struct ModularCMAES
{
    std::shared_ptr<parameters::Parameters> p;

    ModularCMAES(const std::shared_ptr<parameters::Parameters> p) : p(p) {}

    void recombine();

    bool step(FunctionType& objective);

    void operator()(FunctionType& objective);
     
    bool break_conditions() const;
};

