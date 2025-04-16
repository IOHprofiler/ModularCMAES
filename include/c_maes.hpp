#pragma once

#include "parameters.hpp"

struct ModularCMAES
{
    std::shared_ptr<parameters::Parameters> p;

    ModularCMAES(const std::shared_ptr<parameters::Parameters> p) : p(p) {}
    ModularCMAES(const size_t dim) : ModularCMAES(std::make_shared<parameters::Parameters>(dim)) {}
    ModularCMAES(const parameters::Settings &settings) : ModularCMAES(std::make_shared<parameters::Parameters>(settings)) {}
    
    void recombine() const;

    void select() const;

    void adapt() const;

    void mutate(FunctionType &objective) const;

    bool step(FunctionType& objective) const;

    void operator()(FunctionType& objective) const;
     
    bool break_conditions() const;
};

