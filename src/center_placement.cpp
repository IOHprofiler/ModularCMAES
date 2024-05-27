#include "center_placement.hpp"
#include "parameters.hpp"

namespace center
{
    void X0::operator()(parameters::Parameters &p)
    {
        p.adaptation->m = p.settings.x0.value_or(Vector::Zero(p.settings.dim));
    }

    void Uniform::operator()(parameters::Parameters &p)
    {
        // Only works for square spaces
        p.adaptation->m = Vector::Random(p.settings.dim) * p.settings.ub;
    }

    void Zero::operator()(parameters::Parameters &p)
    {
        p.adaptation->m.setZero();
    }
}
