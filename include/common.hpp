#pragma once

#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <algorithm>
#include <ciso646>
#include <numeric>
#include <optional>

#define _USE_MATH_DEFINES
#include <cmath>

#ifdef _MSC_VER
#include <corecrt_math_defines.h>
#endif

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

using Float = double;
using Matrix = Eigen::Matrix<Float, -1, -1>;
using Vector = Eigen::Matrix<Float, -1, 1>;
using size_to = std::optional<size_t>;
using Mask = Eigen::Array<bool, Eigen::Dynamic, 1>;
using Indices = Eigen::ArrayXi;

//! An objective function. It gets the point by reference (a column of the population): a C++
//! objective reads it without a copy; the Python bindings give Python objectives an array of
//! their own.
using FunctionType = std::function<Float(Eigen::Ref<const Vector>)>;

namespace constants
{
extern size_t cache_max_doubles;
extern size_t cache_min_samples;
extern bool cache_samples;
extern bool clip_sigma;
extern bool use_box_muller;
//! The eigendecomposition of the covariance matrix: 0 Eigen's SelfAdjointEigenSolver, 1 LAPACK
//! dsyevd, 2 LAPACK dsyevr (1 and 2 need LAPACK: linked with MODCMA_LAPACK, or scipy's in the
//! Python package; lapack::available()), -1 (default) automatic: dsyevd from lapack_min_dim on
//! when LAPACK is available, Eigen otherwise
extern int eigen_solver;
//! The dimension from which the automatic choice uses LAPACK (Eigen is faster below: Accelerate
//! dsyevd on an Apple M5 is 2x slower at d = 50, even at d ~ 115, 3.4x faster at d = 1000)
extern size_t lapack_min_dim;
//! The Cholesky factorization of CMSA and COVARIANCE_NO_EIGV: 0 Eigen's LLT, 1 LAPACK dpotrf,
//! -1 (default) automatic: dpotrf from lapack_min_dim on when LAPACK is available
extern int cholesky_solver;
//! Matrix products (dgemm) and the LU inverse of M (MA-ES) and A (xNES): -1 (default) automatic,
//! BLAS/LAPACK from blas_min_dim on when available; 0 Eigen; 1 always BLAS/LAPACK when available
extern int blas;
//! The dimension from which the automatic choice uses BLAS for products and LAPACK for the
//! inverse. A single dgemm is faster than Eigen's from d ~ 30, but in whole generations (Apple M5,
//! Accelerate) BLAS was 6-32 % slower at d = 50 and level or faster from d = 100 on
extern size_t blas_min_dim;
} // namespace constants

/**
 * @brief Cdf of a standard normal distribution.
 *
 * see: ndtr_ndtri.cpp
 * @param x a value
 * @return Float P(X <= x) for X ~ N(0, 1)
 */
Float cdf(const Float x);

/**
 * @brief Percent point function (inverse of cdf) of a standard normal distribution.
 *
 * see: ndtr_ndtri.cpp
 * @param x lower tail of the probability
 * @return Float quantile corresponding to the lower tail probability q
 */
Float ppf(const Float x);

/**
 * @brief Generate a sobol sequence using 8 byte integer numbers.
 * see: sobol.cpp
 *
 * @param dim_num  The dimension of the generated vector
 * @param seed The current seed of the sobol sequence
 * @param quasi the vector of random numbers in which to place the output
 */
void i8_sobol(int dim_num, long long int *seed, Float quasi[]);

struct Solution
{
    //! Coordinates
    Vector x;
    //! Function value
    Float y;
    //! Generation
    size_t t;
    //! Evaluations
    size_t e;

    Solution(Eigen::Ref<const Vector> x, const Float y, const size_t t = 0, const size_t e = 0) :
        x(x),
        y(y),
        t(t),
        e(e)
    {
    }

    Solution() :
        x(),
        y(std::numeric_limits<Float>::infinity()),
        t(0),
        e(0)
    {
    }

    [[nodiscard]] size_t n() const { return x.size(); }

    bool operator<(const Solution &other) const { return y < other.y; }

    bool operator>(const Solution &other) const { return y > other.y; }

    bool operator<=(const Solution &other) const { return y <= other.y; }

    bool operator>=(const Solution &other) const { return y >= other.y; }

    std::string repr() const
    {
        std::stringstream ss;
        ss << "Solution x: (" << x.transpose() << ") y: " << y;
        return ss.str();
    }
};

inline std::ostream &operator<<(std::ostream &os, const Solution &s) { return os << s.repr(); }

namespace utils
{
/**
 * @brief Sort an array of indexes idx inplace based
 * based on comparing values in v using std::stable_sort
 *
 * @param v Vector
 * @param idx std::vector<size_t>
 * @return void
 */
void sort_index_inplace(const Vector &v, std::vector<size_t> &idx);

/**
 * @brief The indices 0..n-1 of v in ascending order of v, into idx (reused, no allocation once it
 * has the capacity). Ties keep their index order (the order of a stable sort); NaN values come
 * last. Uses std::sort, which, unlike std::stable_sort, needs no temporary buffer.
 */
void sort_order(Eigen::Ref<const Vector> v, std::vector<size_t> &idx);

/**
 * @brief Return an array of indexes of the sorted array
 * sort indexes based on comparing values in v using std::stable_sort instead
 * of std::sort  to avoid unnecessary index re-orderings
 * when v contains elements of equal values.
 *
 * @param v
 * @return std::vector<size_t>
 */
std::vector<size_t> sort_indexes(const Vector &v);

/**
 * @brief Compute the expected running time (ERT) of a set of runs
 *
 * @param running_times the vector of measured running times
 * @param budget the maximum budget allocated to each run
 * @return std::pair<Float, size_t> (ERT, number of successful runs)
 */
std::pair<Float, size_t> compute_ert(const std::vector<size_t> &running_times, size_t budget);

/**
 * \brief calculate the nearest power of two
 * \tparam T numeric type
 * \param value the number to get the nearest power of two
 * \return the nearest power of two
 */
template <typename T> T nearest_power_of_2(const T value)
{
    const Float val = static_cast<Float>(value);
    return static_cast<T>(pow(2.0, std::floor(std::log2(val))));
}

} // namespace utils

namespace rng
{
//! The random generator of the library
using Generator = std::mt19937;

/**
 * The global random generator. Every optimizer draws from a generator of its own (see fork());
 * objects that are created outside an optimizer (a sampler on its own, the ES baselines) draw from
 * this one.
 */
extern Generator GENERATOR;

/**
 * @brief Set the global seed and reseed the random generator (mt19937)
 *
 * @param seed
 */
void set_seed(int seed);

//! The global generator as a shared pointer (not owning it), the default generator of objects
[[nodiscard]] std::shared_ptr<Generator> global();

/**
 * A generator for a new optimizer: a copy of the global generator, after which the global generator
 * moves on to a different state. After set_seed(s), the first optimizer draws exactly the numbers
 * the global generator would have drawn; a second one draws different numbers. Thread-safe.
 */
[[nodiscard]] std::shared_ptr<Generator> fork();

/**
 * @brief random integer in the half open interval [l, h)
 *
 * @param l lower bound
 * @param h upper bound
 * @param gen the generator to draw from
 * @return int a random integer
 */
int random_integer(const int l, const int h, Generator &gen = GENERATOR);

/**
 * xoshiro256++ (Blackman and Vigna, 2019): a small, fast 64-bit generator, for the opt-in fast
 * normal sampler (Settings.fast_sampling). Seeded with splitmix64.
 */
struct Xoshiro256pp
{
    using result_type = uint64_t;

    explicit Xoshiro256pp(uint64_t seed);

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return ~result_type{0}; }

    result_type operator()()
    {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    //! A uniform number in [0, 1) with 53 random bits
    double uniform() { return static_cast<double>((*this)() >> 11) * 0x1.0p-53; }

  private:
    uint64_t s[4];

    static uint64_t rotl(const uint64_t x, const int k) { return (x << k) | (x >> (64 - k)); }
};

/**
 * A standard normal number with the 256-layer ziggurat of Marsaglia and Tsang (2000), in the 64-bit
 * form of numpy's Generator.standard_normal: one 64-bit draw for about 99 % of the numbers.
 */
double ziggurat_normal(Xoshiro256pp &gen);

/**
 * @brief a shuffler that generates a random permutation of a
 * sequence of integers from start to stop using an lcg.
 */
struct Shuffler
{
    size_t start;
    size_t stop;
    size_t n;
    size_t seed;
    size_t offset;
    size_t multiplier;
    size_t modulus;
    size_t found;

    Shuffler(const size_t start, const size_t stop, Generator &gen = GENERATOR) :
        start(start),
        stop(stop),
        n(stop - start),
        seed(static_cast<size_t>(random_integer(0, static_cast<int>(stop - start), gen))),
        offset(
            static_cast<size_t>(random_integer(0, static_cast<int>(stop - start), gen) * 2 + 1)),
        multiplier(4 * ((stop - start) / 4) + 1),
        modulus(static_cast<size_t>(pow(2, std::ceil(std::log2(stop - start))))),
        found(0)
    {
    }

    Shuffler(const size_t stop, Generator &gen = GENERATOR) :
        Shuffler(0, stop, gen)
    {
    }

    void advance();
    size_t next();
};

struct CachedShuffleSequence
{
    size_t dim;
    size_t n_samples;

    std::vector<Float> cache;
    Shuffler shuffler;

    CachedShuffleSequence(const size_t d, Generator &gen = GENERATOR);

    void fill(const std::vector<Float> &c);

    Vector get_index(const size_t idx);

    Vector next();

    //! The next point, written into out (no allocation)
    void next(Eigen::Ref<Vector> out);
};

/**
 * @brief distribution which in combination with mt19997 produces the same
 * random numbers for gcc and msvc
 */
template <typename T = Float> struct uniform
{
    /**
     * @brief Generate a random uniform number in the open interval (0, 1)
     *
     * The two extreme outputs of the generator, which would give exactly 0 and 1, are rejected:
     * the transformations of the samplers (ppf, Box-Muller, Laplace, ...) map them to +-infinity.
     * Every other output gives the same number as before (u = (g - min) / (max - min)).
     *
     * @tparam G the type of the generator
     * @param gen the generator instance
     * @return T the random number
     */
    template <typename G> T operator()(G &gen)
    {
        auto g = gen();
        while (g == gen.min() || g == gen.max())
            g = gen();
        return static_cast<T>(g - gen.min()) / static_cast<T>(gen.max() - gen.min());
    }
};

/**
 * @brief Box-Muller random normal number generator. Ensures similar numbers generated
 * on different operating systems.
 */
template <typename T = Float> struct normal
{
    T mu;
    T sigma;

    normal(const T mu, const T sigma) :
        mu(mu),
        sigma(sigma)
    {
    }

    normal() :
        normal(0.0, 1.0)
    {
    }

    /**
     * @brief Generate a standard normal random number with mean 0 and std dev 1.
     *
     * @tparam G the type of the generator
     * @param gen the generator instance
     * @return T the random number
     */
    template <typename G> T operator()(G &gen)
    {
        if (generate)
        {
            T u1 = rng(gen);
            T u2 = rng(gen);
            const T root_log_u1 = std::sqrt(-2.0 * std::log(u1));
            const T two_pi_u2 = 2.0 * M_PI * u2;
            const T r1 = (sigma * (root_log_u1 * std::sin(two_pi_u2))) + mu;
            r2 = (sigma * (root_log_u1 * std::cos(two_pi_u2))) + mu;

            generate = false;
            return r1;
        }
        generate = true;
        return r2;
    }

  private:
    // The second number of each Box-Muller pair, kept per distribution object. It used to be a
    // function static, shared by all objects: one object returned the cached number of another
    // (with that one's mu and sigma), and optimizers in one process influenced each other.
    uniform<Float> rng;
    T r2 = 0;
    bool generate = true;
};
} // namespace rng

