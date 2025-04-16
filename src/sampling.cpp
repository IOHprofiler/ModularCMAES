#include "sampling.hpp"
#include "parameters.hpp"

namespace sampling
{
	[[nodiscard]] Vector Tester::operator()()
	{
		Vector x(d);
		++i;
		x.array() = static_cast<Float>(i);
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

	Halton::Halton(const size_t d, const bool scramble) : Sampler(d),
														  index_(6),
														  scramble_(scramble),
														  primes_(n_primes(d)),
														  permutations_(d)
	{
		if (scramble_)
			permutations_ = get_permutations(primes_);
	}

	[[nodiscard]] Vector Halton::operator()()
	{
		Vector res(d);

		if (scramble_)
			for (size_t j = 0; j < d; ++j)
				res(j) = next(index_, primes_[j], permutations_[j]);
		else
			for (size_t j = 0; j < d; ++j)
				res(j) = next(index_, primes_[j]);

		index_++;
		return res;
	}

	Float Halton::next(int index, const int base)
	{
		Float result = 0.0, f = 1.0 / base;
		while (index > 0)
		{
			result += static_cast<Float>(index % base) * f;
			index = index / base;
			f = f / static_cast<Float>(base);
		}
		return result;
	}

	Float Halton::next(int index, const int base, const std::vector<std::vector<int>> &permutations)
	{
		Float result = 0.0, f = 1.0 / base;
		for (const auto &permutation : permutations)
		{
			const Float remainder = permutation[index % base];
			result += remainder * f;
			index = static_cast<int>(std::floor(index / static_cast<Float>(base)));
			f = f / static_cast<Float>(base);
		}
		return result;
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

	std::vector<std::vector<std::vector<int>>> Halton::get_permutations(const std::vector<int> &primes)
	{
		std::vector<std::vector<std::vector<int>>> permutations(primes.size());
		for (size_t i = 0; i < primes.size(); i++)
		{
			const auto count = static_cast<size_t>(std::ceil(54 / std::log2(primes[i])) - 1);
			permutations[i].resize(count);

			std::vector<int> order(primes[i]);
			std::iota(order.begin(), order.end(), 0);

			for (size_t j = 0; j < count; j++)
			{
				std::shuffle(order.begin(), order.end(), rng::GENERATOR);
				permutations[i][j] = order;
			}
		}
		return permutations;
	}

	std::vector<int> Halton::n_primes(const size_t d)
	{
		std::vector<int> primes = sieve(std::max(6, static_cast<int>(d)));
		while (primes.size() < d)
			primes = sieve(static_cast<int>(primes.size() * primes.size()));
		primes.resize(d);
		return primes;
	}

	Sobol::Sobol(const size_t dim) : Sampler(dim), cache(dim)
	{
		long long seed = 2;
		for (size_t i = 0; i < cache.n_samples; i++)
			i8_sobol(static_cast<int>(d), &seed, cache.cache.data() + i * d);
		// cache.transform(ppf);
	}

	std::shared_ptr<Sampler> get(const size_t dim, const parameters::Modules &modules, const size_t lambda)
	{
		using namespace parameters;
		std::shared_ptr<Sampler> sampler;
		switch (modules.sampler)
		{
		case BaseSampler::UNIFORM:
			sampler = std::make_shared<Uniform>(dim);
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
		}

		switch (modules.sample_transformation)
		{
		case SampleTranformerType::GAUSSIAN:
			sampler = std::make_shared<GaussianTransformer>(sampler);
			break;
		case SampleTranformerType::SCALED_UNIFORM:
			sampler = std::make_shared<UniformScaler>(sampler);
			break;
		case SampleTranformerType::LAPLACE:
			sampler = std::make_shared<LaplaceTransformer>(sampler);
			break;
		case SampleTranformerType::LOGISTIC:
			sampler = std::make_shared<LogisticTransformer>(sampler);
			break;
		case SampleTranformerType::CAUCHY:
			sampler = std::make_shared<CauchyTransformer>(sampler);
			break;
		case SampleTranformerType::DOUBLE_WEIBULL:
			sampler = std::make_shared<DoubleWeibullTransformer>(sampler);
			break;
		case SampleTranformerType::NONE:
			sampler = std::make_shared<IdentityTransformer>(sampler);
			break;
		}

		if (modules.orthogonal)
			sampler = std::make_shared<Orthogonal>(sampler, Orthogonal::get_n_samples(modules, lambda));

		if (not(modules.mirrored == Mirror::NONE))
			sampler = std::make_shared<Mirrored>(sampler);

		if (constants::cache_samples && modules.sampler != BaseSampler::SOBOL)
			sampler = std::make_shared<CachedSampler>(sampler);
		return sampler;
	}
}
