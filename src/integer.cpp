#include "integer.hpp"
#include "parameters.hpp"

namespace integer
{

    Array IntegerHandling::get_effective_sigma(const parameters::Parameters& p, const size_t idx) 
    {
        Array effective_sigma = Array::Constant(p.settings.dim, p.pop.s(idx));

        const Array& Cdiag = p.adaptation->coordinate_wise_variances;
        for (const auto& iidx: p.settings.integer_variables)
        {
            const Float Cii = std::max(Cdiag[iidx], Float(1e-16));
            const Float sigma_from_lb = lb_sigma / std::sqrt(Cii);
            effective_sigma[iidx] = std::max(sigma_from_lb, effective_sigma[iidx]);
        }

        return effective_sigma;
    }




}