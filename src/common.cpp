#include "common.hpp"

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
    return {static_cast<Float>(total_rt) / successful_runs, successful_runs};
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
    std::copy(c.begin(), c.end(), cache.begin());
}

void CachedShuffleSequence::transform(const std::function<Float(Float)> &f)
{
    for (Float &i : cache)
        i = f(i);
}

Vector CachedShuffleSequence::get_index(const size_t idx)
{
    return Eigen::Map<Vector>(cache.data() + (idx * dim), static_cast<Eigen::Index>(dim));
}

Vector CachedShuffleSequence::next() { return get_index(shuffler.next()); }

void CachedShuffleSequence::next(Eigen::Ref<Vector> out)
{
    const size_t idx = shuffler.next();
    out = Eigen::Map<const Vector>(cache.data() + (idx * dim), static_cast<Eigen::Index>(dim));
}
} // namespace rng

namespace functions
{
Float sphere(Eigen::Ref<const Vector> x)
{
    Float res = 0;
    for (auto &xi : x)
        res += xi * xi;
    return res;
}

Float rastrigin(Eigen::Ref<const Vector> x)
{
    constexpr Float a = 10.;
    constexpr Float pi2 = 2. * M_PI;
    Float res = 0;
    for (auto &xi : x)
        res += xi * xi - a * std::cos(pi2 * xi);
    return a * static_cast<Float>(x.size()) + res;
}

Float ellipse(Eigen::Ref<const Vector> x)
{
    Float res = 0;
    for (auto i = 0; i < x.size(); ++i)
        res += pow(1.0e6, static_cast<Float>(i) / (static_cast<Float>(x.size()) - 1)) * x(i) * x(i);
    return res;
}

Float rosenbrock(Eigen::Ref<const Vector> x)
{
    Float sum = 0.0;
    for (auto i = 0; i < x.size() - 1; ++i)
    {
        Float xi = x[i];
        Float xi1 = x[i + 1];
        Float term1 = 100.0 * std::pow(xi1 - xi * xi, 2);
        Float term2 = std::pow(1.0 - xi, 2);
        sum += term1 + term2;
    }
    return sum;
}

Matrix random_rotation_matrix(int n, int seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<> d(0, 1);

    Matrix A(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A(i, j) = d(gen);

    Eigen::HouseholderQR<Matrix> qr(A);
    Matrix Q = qr.householderQ();

    if (Q.determinant() < 0)
    {
        Q.col(0) *= -1;
    }

    return Q;
}

} // namespace functions
