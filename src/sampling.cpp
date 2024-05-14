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

			const auto norm = samples.colwise().norm().asDiagonal();

			qr.compute(samples.transpose());
			samples = ((qr.householderQ() * I).transpose() * norm);
		}
		return samples.col(current++);
	}

	size_t Orthogonal::get_n_samples(const parameters::Modules &modules, const size_t lambda)
	{
		using namespace parameters;
		const auto not_mirrored = modules.mirrored == Mirror::NONE;
		const auto has_tpa = modules.ssa == StepSizeAdaptation::TPA;
		return std::max(1, (static_cast<int>(lambda) / (2 - not_mirrored)) - (2 * has_tpa));
	}

	void Orthogonal::reset(const parameters::Modules &mod, const size_t lambda)
	{
		sampler->reset(mod, lambda);
		n = std::max(d, Orthogonal::get_n_samples(mod, lambda));
		qr = Eigen::HouseholderQR<Matrix>(n, d);
		samples = Matrix(d, n);
		I = Matrix::Identity(n, d);
		current = 0;
	}

	Halton::Halton(const size_t d, const size_t budget) : Sampler(d), shuffler(utils::nearest_power_of_2(budget))
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
			res(j) = ppf(next(static_cast<int>(shuffler.next()), primes[j]));
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

	Sobol::Sobol(const size_t dim) : Sampler(dim), cache(dim)
	{
		long long seed = 2;
		for (size_t i = 0; i < cache.n_samples; i++)
			i8_sobol(static_cast<int>(d), &seed, cache.cache.data() + i * d);
	}

	[[nodiscard]] Vector Sobol::operator()()
	{
		Vector res = cache.next();
		for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(d); ++j)
			res(j) = ppf(res(j));
		return res;
	}

	std::shared_ptr<Sampler> get(const size_t dim, const size_t budget, const parameters::Modules &modules, const size_t lambda)
	{
		using namespace parameters;
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
			sampler = std::make_shared<Halton>(dim, budget);
			break;
		case BaseSampler::TESTER:
			sampler = std::make_shared<Tester>(dim);
			break;
		}

		if (modules.orthogonal)
		{
			sampler = std::make_shared<Orthogonal>(sampler, Orthogonal::get_n_samples(modules, lambda));
		}
		if (not(modules.mirrored == Mirror::NONE))
			sampler = std::make_shared<Mirrored>(sampler);
		return sampler;
	}
}
