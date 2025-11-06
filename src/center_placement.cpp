#include "center_placement.hpp"
#include "parameters.hpp"

namespace center
{
    void X0::operator()(parameters::Parameters &p)
    {
        p.adaptation->m = p.settings.x0.value_or(p.settings.center);
    }

    void Uniform::operator()(parameters::Parameters &p)
    {
        p.adaptation->m = p.settings.lb + (Vector::Random(p.settings.dim) * p.settings.db);
    }

    void Zero::operator()(parameters::Parameters &p)
    {
        p.adaptation->m.setZero();
    }

    void Center::operator()(parameters::Parameters& p)
    {
        p.adaptation->m = p.settings.center;   
    }
}
