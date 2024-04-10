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

    void Repelling::operator()(parameters::Parameters &p)
    {
        const size_t n_points = p.repelling->archive.size();
        p.adaptation->m.setZero();

        if (n_points > 1)
        {
            if (n_points % 2 == 0)
            {
                for (Eigen::Index i = 0; i < p.adaptation->m.size(); i++)
                {
                    for (const auto &point : p.repelling->archive)
                        p.adaptation->m(i) += point.solution.x(i);
                    p.adaptation->m(i) /= n_points;
                }
            }
            else
            {
                const size_t n_samples = 100;
                double max_distance = 0;
                for (size_t i = 0; i < n_samples; i++)
                {
                    using namespace repelling::distance;
                    const auto sample = Vector::Random(p.settings.dim) * p.settings.ub;
                    double dist = std::min(manhattan(sample, p.settings.lb), manhattan(sample, p.settings.ub));
                    for (const auto &point : p.repelling->archive)
                        dist += euclidian(sample, point.solution.x);

                    if (dist > max_distance)
                    {
                        max_distance = dist;
                        p.adaptation->m = sample;
                    }
                }
            }
        }
        else 
        {
            p.adaptation->m = Vector::Random(p.settings.dim) * p.settings.ub;
        }
    }

}
