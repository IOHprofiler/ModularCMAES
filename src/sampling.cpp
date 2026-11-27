#include "sampling.hpp"
#include "parameters.hpp"

namespace sampling
{
void Tester::sample(Eigen::Ref<Vector> out)
{
    ++i;
    out.array() = static_cast<Float>(i);
}

void Mirrored::sample(Eigen::Ref<Vector> out)
{
    if (!mirror)
    {
        sampler->sample(out);
        previous = out;
        mirror = true;
        return;
    }
    mirror = false;
    out = -previous;
}

void Orthogonal::sample(Eigen::Ref<Vector> out)
{
    if (current >= block_size)
        next_block();
    out = samples.col(current++);
}

void Orthogonal::next_block()
{
    // Without begin_generation, every n consecutive samples are a generation
    if (remaining == 0)
        remaining = n;
    block_size = static_cast<Eigen::Index>(std::min(d, remaining));
    remaining -= static_cast<size_t>(block_size);
    current = 0;

    auto block = samples.leftCols(block_size);
    for (Eigen::Index i = 0; i < block_size; ++i)
        sampler->sample(block.col(i));

    // The lengths of the samples, before the block is overwritten (Wang et al. 2019, Eq. 4-6)
    lengths.head(block_size) = block.colwise().norm().transpose();
    qr.compute(block);
    // Q R with diag(R) > 0 is the factorization that the Gram-Schmidt process gives (Wang et al.
    // 2019, Sec. 3.2): the first sample keeps its direction. Eigen's Householder QR leaves the
    // signs of diag(R) to the data; uncorrected, sample j of every block was biased towards -e_j
    // (#76, F30; Mezzadri 2007)
    const Matrix &R = qr.matrixQR();
    for (Eigen::Index j = 0; j < block_size; ++j)
        if (R(j, j) < 0)
            lengths(j) = -lengths(j);
    // The first block_size columns of Q: Q applied to the identity, in place (no temporaries)
    block.setIdentity();
    qr.householderQ().applyThisOnTheLeft(block, workspace);
    block *= lengths.head(block_size).asDiagonal();
}

void Orthogonal::begin_generation()
{
    sampler->begin_generation();
    remaining = n;
    block_size = 0;
    current = 0;
}

size_t Orthogonal::get_n_samples(const parameters::Modules &modules, const size_t lambda)
{
    // TPA's two points are extra points, not part of lambda: no correction for them (#76, F11)
    const bool mirrored = modules.mirrored != parameters::Mirror::NONE;
    return std::max<size_t>(1, mirrored ? (lambda + 1) / 2 : lambda);
}

void Orthogonal::reset(const parameters::Modules &mod, const size_t lambda)
{
    sampler->reset(mod, lambda);
    n = Orthogonal::get_n_samples(mod, lambda);
    begin_generation();
}

Halton::Halton(const size_t d,
               const bool scramble,
               const std::shared_ptr<rng::Generator> &generator) :
    Sampler(d),
    index_(6),
    scramble_(scramble),
    primes_(n_primes(d)),
    permutations_(d)
{
    if (scramble_)
        permutations_ = get_permutations(primes_, *generator);
}

void Halton::sample(Eigen::Ref<Vector> out)
{
    if (scramble_)
        for (size_t j = 0; j < d; ++j)
            out(j) = next(index_, primes_[j], permutations_[j]);
    else
        for (size_t j = 0; j < d; ++j)
            out(j) = next(index_, primes_[j]);

    index_++;
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

std::vector<std::vector<std::vector<int>>> Halton::get_permutations(const std::vector<int> &primes,
                                                                    rng::Generator &gen)
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
            // Fisher-Yates by hand: the algorithm of std::shuffle is implementation-defined, and
            // libc++ and libstdc++ gave different permutations (#116)
            rng::shuffle(order, gen);
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

Sobol::Sobol(const size_t dim, const std::shared_ptr<rng::Generator> &generator) :
    Sampler(dim),
    cache(dim, *generator)
{
    long long seed = 2;
    for (size_t i = 0; i < cache.n_samples; i++)
        i8_sobol(static_cast<int>(d), &seed, cache.cache.data() + i * d);
}

Float expected_norm(const size_t d,
                    const std::function<std::pair<Float, Float>(Float)> &phi_psi,
                    const Float second_moment)
{
    // The integrand (1 - phi(e^x)^d) e^(-x/2) behaves as d E[z^2] e^(x/2) for x -> -inf and as
    // e^(-x/2) for x -> inf; outside [x0, x1] these tails are integrated in closed form (each is
    // 2 e^-25 ~ 3e-11). The integrand is analytic in |Im x| < pi/2 (phi is analytic for
    // Re t > 0), so the trapezoidal rule with step h has an error of about exp(-pi^2 / h).
    const Float dd = static_cast<Float>(d);
    const Float x0 = -2.0 * std::log(dd * second_moment) - 50.0;
    const Float x1 = 50.0;
    const auto steps = static_cast<int>(std::ceil((x1 - x0) / 0.25));
    const Float h = (x1 - x0) / static_cast<Float>(steps);

    Float sum = 0.0;
    for (int k = 0; k <= steps; ++k)
    {
        const Float x = x0 + h * static_cast<Float>(k);
        const Float t = std::exp(x);
        const auto [phi, psi] = phi_psi(t);
        // 1 - phi^d, without cancellation: from psi = 1 - phi for small t, from phi for large t
        const Float one_minus = t < 1.0   ? -std::expm1(dd * std::log1p(-psi))
                                : phi > 0 ? -std::expm1(dd * std::log(phi))
                                          : 1.0;
        sum += (k == 0 || k == steps ? 0.5 : 1.0) * one_minus * std::exp(-0.5 * x);
    }
    const Float tails = 2.0 * dd * second_moment * std::exp(0.5 * x0) + 2.0 * std::exp(-0.5 * x1);
    return (h * sum + tails) / (2.0 * std::sqrt(M_PI));
}

std::function<std::pair<Float, Float>(Float)>
phi_psi_from_density(const std::function<Float(Float)> &density)
{
    // z = e^y on a fixed grid, y in [-38, 4]: wide enough for t = e^50 (the peak of the integrand
    // of phi is at y = -ln(t) / 2 = -25), and a density that decays exponentially is below 1e-20
    // at z = e^4 = 55 (for the unit-variance Laplace and logistic distributions). The integrand
    // of phi is analytic in |Im y| < pi / 4 (and of the logistic density up to pi / 2): a step
    // of 0.2 gives an error of about exp(-pi^2 / (2 * 0.2)) ~ 2e-11.
    const Float y0 = -38.0, y1 = 4.0, h = 0.2;
    const auto steps = static_cast<int>(std::ceil((y1 - y0) / h));
    std::vector<Float> z2(static_cast<size_t>(steps) + 1), weight(static_cast<size_t>(steps) + 1);
    for (int k = 0; k <= steps; ++k)
    {
        const Float z = std::exp(y0 + h * static_cast<Float>(k));
        z2[static_cast<size_t>(k)] = z * z;
        // dz = z dy
        weight[static_cast<size_t>(k)] = (k == 0 || k == steps ? 0.5 : 1.0) * h * density(z) * z;
    }
    return [z2 = std::move(z2), weight = std::move(weight)](const Float t) {
        Float phi = 0.0, psi = 0.0;
        for (size_t k = 0; k < z2.size(); ++k)
        {
            phi += weight[k] * std::exp(-t * z2[k]);
            psi += weight[k] * -std::expm1(-t * z2[k]);
        }
        return std::make_pair(phi, psi);
    };
}

std::pair<Float, Float> phi_psi_uniform(const Float a, const Float t)
{
    // phi(t) = int_0^1 exp(-t a^2 u^2) du = sqrt(pi) erf(sqrt(w)) / (2 sqrt(w)), w = a^2 t; for
    // small w the series psi = sum_k>=1 -(-w)^k / (k! (2k + 1)), without cancellation
    const Float w = a * a * t;
    if (w < 0.5)
    {
        Float psi = 0.0, term = 1.0;
        for (int k = 1; k < 30; ++k)
        {
            term *= -w / static_cast<Float>(k);
            psi -= term / static_cast<Float>(2 * k + 1);
        }
        return {1.0 - psi, psi};
    }
    const Float phi = std::sqrt(M_PI) * std::erf(std::sqrt(w)) / (2.0 * std::sqrt(w));
    return {phi, 1.0 - phi};
}

Float IdentityTransformer::expected_length()
{
    if (!length)
        length = expected_norm(d, [](const Float t) { return phi_psi_uniform(1.0, t); }, 1.0 / 3.0);
    return *length;
}

Float UniformScaler::expected_length()
{
    if (!length)
        length = expected_norm(d, [](const Float t) { return phi_psi_uniform(scale, t); }, 1.0);
    return *length;
}

Float LaplaceTransformer::expected_length()
{
    // |z| is exponential with mean b
    if (!length)
        length = expected_norm(
            d, phi_psi_from_density([](const Float z) { return std::exp(-z / b) / b; }), 1.0);
    return *length;
}

Float LogisticTransformer::expected_length()
{
    // |z| has the density 2 e^(-z/s) / (s (1 + e^(-z/s))^2)
    if (!length)
        length = expected_norm(d,
                               phi_psi_from_density([](const Float z) {
                                   const Float e = std::exp(-z / s);
                                   return 2.0 * e / (s * (1.0 + e) * (1.0 + e));
                               }),
                               1.0);
    return *length;
}

Float DoubleWeibullTransformer::expected_length()
{
    if (!length)
        length =
            std::exp(std::lgamma(static_cast<Float>(d) + 0.5) - std::lgamma(static_cast<Float>(d)));
    return *length;
}

namespace
{
//! The base sampler and its transformation
std::shared_ptr<Sampler> base_sampler(const size_t dim,
                                      const parameters::Modules &modules,
                                      const std::shared_ptr<rng::Generator> &generator)
{
    using namespace parameters;
    std::shared_ptr<Sampler> sampler;
    switch (modules.sampler)
    {
    case BaseSampler::UNIFORM:
        sampler = std::make_shared<Uniform>(dim, generator);
        break;
    case BaseSampler::SOBOL:
        sampler = std::make_shared<Sobol>(dim, generator);
        break;
    case BaseSampler::HALTON:
        sampler = std::make_shared<Halton>(dim, true, generator);
        break;
    case BaseSampler::TESTER:
        sampler = std::make_shared<Tester>(dim);
        break;
    }

    switch (modules.sample_transformation)
    {
    case SampleTranformerType::GAUSSIAN:
        return std::make_shared<GaussianTransformer>(sampler);
    case SampleTranformerType::SCALED_UNIFORM:
        return std::make_shared<UniformScaler>(sampler);
    case SampleTranformerType::LAPLACE:
        return std::make_shared<LaplaceTransformer>(sampler);
    case SampleTranformerType::LOGISTIC:
        return std::make_shared<LogisticTransformer>(sampler);
    case SampleTranformerType::CAUCHY:
        return std::make_shared<CauchyTransformer>(sampler);
    case SampleTranformerType::DOUBLE_WEIBULL:
        return std::make_shared<DoubleWeibullTransformer>(sampler);
    case SampleTranformerType::NONE:
    default:
        return std::make_shared<IdentityTransformer>(sampler);
    }
}
} // namespace

std::shared_ptr<Sampler> get(const size_t dim,
                             const parameters::Modules &modules,
                             const size_t lambda,
                             const std::shared_ptr<rng::Generator> &generator,
                             const bool fast_normal)
{
    using namespace parameters;
    std::shared_ptr<Sampler> sampler;
    if (fast_normal && modules.sampler == BaseSampler::UNIFORM &&
        modules.sample_transformation == SampleTranformerType::GAUSSIAN)
        sampler = std::make_shared<FastGaussian>(dim, *generator);
    else
        sampler = base_sampler(dim, modules, generator);

    if (modules.orthogonal)
        sampler = std::make_shared<Orthogonal>(sampler, Orthogonal::get_n_samples(modules, lambda));

    // Mirrored sampling is not a sampler here: mutation::Strategy samples the pairs of each
    // generation together (#75)

    if (constants::cache_samples && modules.sampler != BaseSampler::SOBOL)
        sampler = std::make_shared<CachedSampler>(sampler);
    return sampler;
}
} // namespace sampling
