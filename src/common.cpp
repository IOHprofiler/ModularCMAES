#include "common.hpp"
#include <stdexcept>
#include <string>

#include <mutex>

template <typename T> std::ostream &operator<<(std::ostream &os, const std::vector<T> &x)
{
    for (auto &xi : x)
        os << xi << ' ';
    return os;
}

namespace constants
{
size_t cache_max_doubles = 2'000'000;
size_t cache_min_samples = 128;
bool cache_samples = false;
bool clip_sigma = false;
bool use_box_muller = false;
int eigen_solver = -1;
size_t lapack_min_dim = 120;
int cholesky_solver = -1;
int blas = -1;
size_t blas_min_dim = 100;
} // namespace constants

namespace utils
{
void sort_index_inplace(const Vector &v, std::vector<size_t> &idx)
{
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
}

void sort_index_inplace(const std::vector<size_t> &v, std::vector<size_t> &idx)
{
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
}

void sort_order(Eigen::Ref<const Vector> v, std::vector<size_t> &idx)
{
    idx.resize(static_cast<size_t>(v.size()));
    std::iota(idx.begin(), idx.end(), size_t{0});
    std::sort(idx.begin(), idx.end(), [&v](const size_t a, const size_t b) {
        const Float fa = v(static_cast<Eigen::Index>(a));
        const Float fb = v(static_cast<Eigen::Index>(b));
        const bool nan_a = std::isnan(fa), nan_b = std::isnan(fb);
        if (nan_a != nan_b)
            return nan_b;
        if (!nan_a && fa != fb)
            return fa < fb;
        return a < b;
    });
}

std::vector<size_t> sort_indexes(const Vector &v)
{
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    sort_index_inplace(v, idx);
    return idx;
}

std::vector<size_t> sort_indexes(const std::vector<size_t> &v)
{
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    sort_index_inplace(v, idx);
    return idx;
}

void hstack(Matrix &X, const Matrix &Y)
{
    X.conservativeResize(Eigen::NoChange, X.cols() + Y.cols());
    X.rightCols(Y.cols()) = Y;
}

void vstack(Matrix &X, const Matrix &Y)
{
    X.conservativeResize(X.rows() + Y.rows(), Eigen::NoChange);
    X.bottomRows(Y.rows()) = Y;
}

void concat(Vector &x, const Vector &y)
{
    x.conservativeResize(x.rows() + y.rows(), Eigen::NoChange);
    x.bottomRows(y.rows()) = y;
}

std::pair<Float, size_t> compute_ert(const std::vector<size_t> &running_times, const size_t budget)
{
    size_t successful_runs = 0, total_rt = 0;

    for (const auto &rt : running_times)
    {
        if (rt < budget)
            successful_runs++;
        total_rt += rt;
    }
    return {static_cast<Float>(total_rt) / static_cast<Float>(successful_runs), successful_runs};
}
} // namespace utils

namespace rng
{
int SEED = std::random_device()();
Generator GENERATOR(SEED);

namespace
{
std::mutex global_mutex;
}

void set_seed(const int seed)
{
    const std::lock_guard<std::mutex> lock(global_mutex);
    SEED = seed;
    GENERATOR.seed(seed);
    srand(seed);
}

std::shared_ptr<Generator> global() { return std::shared_ptr<Generator>(std::shared_ptr<void>{}, &GENERATOR); }

std::shared_ptr<Generator> fork()
{
    const std::lock_guard<std::mutex> lock(global_mutex);
    auto generator = std::make_shared<Generator>(GENERATOR);
    GENERATOR.seed(GENERATOR());
    return generator;
}

int random_integer(const int l, const int h, Generator &gen)
{
    std::uniform_int_distribution<> distrib(l, std::max(l, h - 1));
    return distrib(gen);
}

Xoshiro256pp::Xoshiro256pp(uint64_t seed)
{
    // splitmix64 spreads one seed over the four words (never all zero)
    for (auto &word : s)
    {
        uint64_t z = (seed += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        word = z ^ (z >> 31);
    }
}

namespace
{
//! The tables of the 256-layer ziggurat for the standard normal density f(x) = exp(-x^2 / 2)
struct ZigguratTables
{
    static constexpr double r = 3.6541528853610087963519472518; // start of the tail
    static constexpr double v = 4.92867323399e-3;              // area of every layer
    static constexpr double scale = 0x1.0p52;                   // 52 random bits per number

    uint64_t k[256]; // accept x directly when the 52 bits are below k[i]
    double w[256];   // x = bits * w[i]
    double f[256];   // f at the layer boundaries

    ZigguratTables()
    {
        double dn = r, tn = r;
        const double q = v / std::exp(-0.5 * dn * dn);
        k[0] = static_cast<uint64_t>((dn / q) * scale);
        k[1] = 0;
        w[0] = q / scale;
        w[255] = dn / scale;
        f[0] = 1.0;
        f[255] = std::exp(-0.5 * dn * dn);
        for (int i = 254; i >= 1; i--)
        {
            dn = std::sqrt(-2.0 * std::log(v / dn + std::exp(-0.5 * dn * dn)));
            k[i + 1] = static_cast<uint64_t>((dn / tn) * scale);
            tn = dn;
            f[i] = std::exp(-0.5 * dn * dn);
            w[i] = dn / scale;
        }
    }
};

const ZigguratTables &ziggurat()
{
    static const ZigguratTables tables; // thread-safe initialisation
    return tables;
}
} // namespace

double ziggurat_normal(Xoshiro256pp &gen)
{
    const ZigguratTables &z = ziggurat();
    for (;;)
    {
        uint64_t bits = gen();
        const auto i = static_cast<int>(bits & 0xff);
        bits >>= 8;
        const bool negative = bits & 0x1;
        const uint64_t mantissa = (bits >> 1) & 0x000fffffffffffffULL;
        double x = static_cast<double>(mantissa) * z.w[i];
        if (negative)
            x = -x;
        if (mantissa < z.k[i])
            return x; // inside the layer's rectangle
        if (i == 0)
        {
            // the tail beyond r (Marsaglia's method)
            for (;;)
            {
                const double xx = -std::log1p(-gen.uniform()) / ZigguratTables::r;
                const double yy = -std::log1p(-gen.uniform());
                if (yy + yy > xx * xx)
                    return negative ? -(ZigguratTables::r + xx) : ZigguratTables::r + xx;
            }
        }
        // the wedge between the rectangle and the density
        if ((z.f[i - 1] - z.f[i]) * gen.uniform() + z.f[i] < std::exp(-0.5 * x * x))
            return x;
    }
}

void Shuffler::advance()
{
    do
    {
        seed = (seed * multiplier + offset) % modulus;
    } while (seed >= n);
}

size_t Shuffler::next()
{
    if (found > 0)
        advance();
    found++;
    return start + seed;
}

CachedShuffleSequence::CachedShuffleSequence(const size_t d, Generator &gen) :
    dim(d),
    n_samples(std::max(constants::cache_min_samples,
                       utils::nearest_power_of_2(constants::cache_max_doubles / d))),
    cache(n_samples * d, 0.0),
    shuffler(n_samples, gen)
{
}

void CachedShuffleSequence::fill(const std::vector<Float> &c)
{
    if (c.size() > cache.size())
        throw std::invalid_argument("CachedShuffleSequence.fill: " + std::to_string(c.size()) +
                                    " values, the cache holds " + std::to_string(cache.size()));
    std::copy(c.begin(), c.end(), cache.begin());
}

void CachedShuffleSequence::transform(const std::function<Float(Float)> &f)
{
    for (Float &i : cache)
        i = f(i);
}

Vector CachedShuffleSequence::get_index(const size_t idx)
{
    if (idx >= n_samples)
        throw std::out_of_range("CachedShuffleSequence.get_index: index " + std::to_string(idx) +
                                " of " + std::to_string(n_samples) + " points");
    return Eigen::Map<Vector>(cache.data() + (idx * dim), static_cast<Eigen::Index>(dim));
}

Vector CachedShuffleSequence::next() { return get_index(shuffler.next()); }

void CachedShuffleSequence::next(Eigen::Ref<Vector> out)
{
    const size_t idx = shuffler.next();
    out = Eigen::Map<const Vector>(cache.data() + (idx * dim), static_cast<Eigen::Index>(dim));
}
} // namespace rng

