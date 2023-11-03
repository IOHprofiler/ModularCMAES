#pragma once

#include "parameters.hpp"

struct ModularCMAES
{
    std::shared_ptr<parameters::Parameters> p;

    ModularCMAES(const std::shared_ptr<parameters::Parameters> p) : p(p) {}

    void recombine();

    bool step(std::function<double(Vector)> objective);

    void operator()(std::function<double(Vector)> objective);

    bool break_conditions() const;
};

