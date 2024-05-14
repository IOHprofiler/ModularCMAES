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

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Array = Eigen::ArrayXd;
using size_to = std::optional<size_t>;

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &x);

using FunctionType = std::function<double(const Vector &)>;

namespace constants
{
	extern double tolup_sigma;
	extern double tol_condition_cov;
	extern double tol_min_sigma;
	extern double stagnation_quantile;
	extern double sigma_threshold;
	extern size_t shuffle_cache_max_doubles;
	extern size_t shuffle_cache_min_samples;
}

/**
 * @brief Cdf of a standard normal distribution.
 *
 * see: ndtr_ndtri.cpp
 * @param x lower tail of the probabilty
 * @return double quantile corresponding to the lower tail probability q
 */
double cdf(const double x);

/**
 * @brief Percent point function (inverse of cdf) of a standard normal distribution.
 *
 * see: ndtri.cpp
 * @param x lower tail of the probabilty
 * @return double quantile corresponding to the lower tail probability q
 */
double ppf(const double x);

/**
 * @brief Generate a sobol sequence using 8 byte integer numbers.
 * see: sobol.cpp
 *
 * @param dim_num  The dimension of the generated vector
 * @param seed The current seed of the sobol sequence
 * @param quasi the vector of random numbers in which to place the output
 */
void i8_sobol(int dim_num, long long int *seed, double quasi[]);

struct Solution
{
	//! Coordinates
	Vector x;
	//! Function value
	double y;
	//! Generation
	size_t t;
	//! Evaluations
	size_t e;

	Solution(const Vector &x, const double y, const size_t t = 0, const size_t e = 0) : x(x), y(y), t(t), e(e)
	{
	}

	Solution() : Solution({}, std::numeric_limits<double>::infinity()) {}

	[[nodiscard]] size_t n() const
	{
		return x.size();
	}

	bool operator<(const Solution &other) const
	{
		return y < other.y;
	}

	bool operator>(const Solution &other) const
	{
		return y > other.y;
	}

	bool operator<=(const Solution &other) const
	{
		return y <= other.y;
	}

	bool operator>=(const Solution &other) const
	{
		return y >= other.y;
	}

	std::string repr() const
	{
		std::stringstream ss;
		ss << "Solution x: (" << x.transpose() << ") y: " << y;
		return ss.str();
	}
};

inline std::ostream &operator<<(std::ostream &os, const Solution &s)
{
	return os << s.repr();
}

namespace utils
{
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

	std::vector<size_t> sort_indexes(const std::vector<size_t> &v);

	/**
	 * @brief Concat two matrices inplace, i.e. put Y in X (colwise)
	 *
	 * @param X target matrix
	 * @param Y source matrix
	 */
	void hstack(Matrix &X, const Matrix &Y);

	/**
	 * @brief Concat two matrices inplace, i.e. put Y in X (rowwise)
	 *
	 * @param X target matrix
	 * @param Y source matrix
	 */
	void vstack(Matrix &X, const Matrix &Y);

	/**
	 * @brief Concat two vectors
	 *
	 * @param x target vector
	 * @param y source vector
	 */
	void concat(Vector &x, const Vector &y);

	/**
	 * @brief Compute the expected running time (ERT) of a set of runs
	 *
	 * @param running_times the vector of measured running times
	 * @param budget the maximum budget allocated to each run
	 * @return std::pair<double, size_t> (ERT, number of successfull runs)
	 */
	std::pair<double, size_t> compute_ert(const std::vector<size_t> &running_times, size_t budget);

	/**
	 * \brief calculate the nearest power of two
	 * \tparam T numeric type
	 * \param value the number to get the nearest power of two
	 * \return the nearest power of two
	 */
	template<typename T>
	T nearest_power_of_2(const T value)
	{
		const double val = static_cast<double>(value);
		return static_cast<T>(pow(2.0, std::floor(std::log2(val))));
	}

}

namespace rng
{
	//! The global seed value
	extern int SEED;

	//! The random generator
	extern std::mt19937 GENERATOR;

	/**
	 * @brief Set the global seed and reseed the random generator (mt19937)
	 *
	 * @param seed
	 */
	void set_seed(int seed);
	/**
	 * @brief random integer generator using global GENERATOR in the half open interval [l, h)
	 *
	 * @param l lower bound
	 * @param h upper bound
	 * @return int a random integer
	 */
	int random_integer(const int l, const int h);

	

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

		Shuffler(const size_t start, const size_t stop) : start(start),
														  stop(stop),
														  n(stop - start),
														  seed(static_cast<size_t>(random_integer(0, static_cast<int>(stop - start)))),
														  offset(static_cast<size_t>(random_integer(0, static_cast<int>(stop - start)) * 2 + 1)),
														  multiplier(4 * ((stop - start) / 4) + 1),
														  modulus(static_cast<size_t>(pow(2, std::ceil(std::log2(stop - start))))),
														  found(0)
		{
		}

		Shuffler(const size_t stop) : Shuffler(0, stop) {}

		void advance();
		size_t next();
	};


	struct CachedShuffleSequence
	{
		size_t dim;
		size_t n_samples;

		std::vector<double> cache;
		Shuffler shuffler;

		CachedShuffleSequence(const size_t d);

		void fill(const std::vector<double>& c);

		Vector get_index(const size_t idx);

		Vector next();
	};

	/**
	 * @brief distribution which in combination with mt19997 produces the same
	 * random numbers for gcc and msvc
	 */
	template <typename T = double>
	struct uniform
	{
		/**
		 * @brief Generate a random uniform number in the closed interval [-1, 1]
		 *
		 * @tparam G the type of the generator
		 * @param gen the generator instance
		 * @return T the random number
		 */
		template <typename G>
		T operator()(G &gen)
		{
			return static_cast<T>(2.0 * gen() - gen.min()) / gen.max() - gen.min() - 1;
		}
	};

	/**
	 * @brief Box-Muller random normal number generator. Ensures similar numbers generated
	 * on different operating systems.
	 */
	template <typename T = double>
	struct normal
	{
		T mu;
		T sigma;

		normal(const T mu, const T sigma) : mu(mu), sigma(sigma)
		{
		}

		normal() : normal(0.0, 1.0)
		{
		}

		/**
		 * @brief Generate a standard normal random number with mean 0 and std dev 1.
		 *
		 * @tparam G the type of the generator
		 * @param gen the generator instance
		 * @return T the random number
		 */
		template <typename G>
		T operator()(G &gen)
		{
			static uniform<double> rng;
			static T r1, r2;
			static bool generate = true;

			if (generate)
			{
				T u1 = std::abs(rng(gen));
				T u2 = std::abs(rng(gen));
				const T root_log_u1 = std::sqrt(-2.0 * std::log(u1));
				const T two_pi_u2 = 2.0 * M_PI * u2;
				r1 = (sigma * (root_log_u1 * std::sin(two_pi_u2))) + mu;
				r2 = (sigma * (root_log_u1 * std::cos(two_pi_u2))) + mu;

				generate = false;
				return r1;
			}
			generate = true;
			return r2;
		}
	};
}

namespace functions
{
	double sphere(const Vector &x);
	double rastrigin(const Vector &x);
}
