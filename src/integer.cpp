#include "integer.hpp"
#include "parameters.hpp"

namespace integer
{
    void IntegerHandling::update_diagonal(const parameters::Parameters& p) 
    {
        if (p.settings.integer_variables.size() == 0)
            return;
        
        effective_y = p.adaptation->compute_y(ones);
    }

    Array IntegerHandling::get_effective_sigma(const parameters::Parameters& p, const size_t idx) 
    {
        Array effective_sigma = Array::Constant(p.settings.dim, p.pop.s(idx));
        for (const auto& iidx: p.settings.integer_variables)
        {
            
        }

        return effective_sigma;
    }
}