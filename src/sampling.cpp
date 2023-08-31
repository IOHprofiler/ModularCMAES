#include "sampling.hpp"
#include "parameters.hpp"

namespace sampling
{
    [[nodiscard]] Vector Tester::operator()() 
    {
        Vector x(d);
        ++i;
        x.array() = static_cast<double>(i);
        return x;
    };


    [[nodiscard]] Vector Mirrored::operator()()
    {
        if (!mirror)
        {
            previous = (*sampler)();
            mirror = true;
            return previous;
        }
        mirror = false;
        return -previous;
    }

    [[nodiscard]] Vector Orthogonal::operator()()
    {
        if (current >= n)
            current = 0;

        if (!current)
        {
            for (size_t i = 0; i < n; ++i)
                samples.col(i) = (*sampler)();

            auto norm = samples.colwise().norm().asDiagonal();

            qr.compute(samples.transpose());
            samples = ((qr.householderQ() * I).transpose() * norm);
        }
        return samples.col(current++);
    }

    Halton::Halton(const size_t d, const size_t i) : Sampler(d), i(i)
    {
        primes = sieve(std::max(6, static_cast<int>(d)));
        while (primes.size() < d)
            primes = sieve(static_cast<int>(primes.size() * primes.size()));
        primes.resize(d);
    }

    [[nodiscard]] Vector Halton::operator()()
    {
        Vector res(d);
        for (size_t j = 0; j < d; ++j)
            res(j) = ppf(next(static_cast<int>(i), primes[j]));
        i++;
        return res;
    }

    double Halton::next(int index, int base)
    {
        double y = 1., x = 0.;
        while (index > 0)
        {
            auto dm = divmod(index, base);
            index = dm.first;
            y *= static_cast<double>(base);
            x += static_cast<double>(dm.second) / y;
        }
        return x;
    }

    std::pair<int, int> Halton::divmod(const double top, const double bottom)
    {
        const auto div = static_cast<int>(top / bottom);
        return {div, static_cast<int>(top - div * bottom)};
    }

    std::vector<int> Halton::sieve(const int n)
    {
       std::vector<int> mask(n + 1, 1);
                
        for (int p = 2; p * p <= n; p++)
        {
            if (mask[p])
                for (int i = p * p; i <= n; i += p)
                    mask[i] = 0;
        }
        
        std::vector<int> primes;
        for (int p = 2; p <= n; p++)
            if (mask[p])
                primes.push_back(p);
                
        return primes;
    }

    [[nodiscard]] Vector Sobol::operator()()
    {
        Vector res(d);
        i8_sobol(static_cast<int>(d), &seed, res.data());
        for (size_t j = 0; j < d; ++j)
            res(j) = ppf(res(j));

        seed += d;
        return res;
    }


    std::shared_ptr<Sampler> get(const size_t dim, const parameters::Modules& modules, const size_t lambda)
    {
        std::shared_ptr<Sampler> sampler;
        switch (modules.sampler)
        {
        case BaseSampler::GAUSSIAN:
            sampler = std::make_shared<Gaussian>(dim);
            break;
        case BaseSampler::SOBOL:
            sampler = std::make_shared<Sobol>(dim);
            break;
        case BaseSampler::HALTON:
            sampler = std::make_shared<Halton>(dim);
            break;
        case BaseSampler::TESTER:
            sampler = std::make_shared<Tester>(dim);
            break;
        };

        auto not_mirrored = modules.mirrored == Mirror::NONE;
        if (modules.orthogonal)
        {
            auto has_tpa = modules.ssa == mutation::StepSizeAdaptation::TPA;
            auto n_samples = std::max(1, (static_cast<int>(lambda) / (2 - not_mirrored)) - (2 * has_tpa));
            sampler = std::make_shared<Orthogonal>(sampler, n_samples);
        }
        if (not not_mirrored)
            sampler = std::make_shared<Mirrored>(sampler);
        return sampler;
    }

}