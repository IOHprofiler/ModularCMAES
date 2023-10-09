#pragma once

#include "common.hpp"

namespace parameters
{
    struct Stats
    {
        size_t t = 0;
        size_t evaluations = 0;
        Vector xopt = Vector(0);
        double fopt = std::numeric_limits<double>::infinity();
    };
}