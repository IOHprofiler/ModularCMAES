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
#include <corecrt_math_defines.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Array = Eigen::ArrayXd;
using size_to = std::optional<size_t>;

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &x);


using FunctionType = std::function<double(const Vector&)>; 

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

    std::vector<size_t> sort_indexes(const std::vector<size_t>& v);

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
    std::pair<double, size_t> compute_ert(const std::vector<size_t> &running_times, const size_t budget);
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
    void set_seed(const int seed);
    /**
     * @brief random integer generator using global GENERATOR
     *
     * @param l lower bound
     * @param h upper bound
     * @return int a random integer
     */
    int random_integer(int l, int h);

    /**
     * @brief distribution which in compbination with mt19997 produces the same
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
        
        normal(const T mu, const T sigma): mu(mu), sigma(sigma) {}
        normal(): normal(0.0, 1.0) {}

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
            else
            {
                generate = true;
                return r2;
            }
            
        }
    };
}

namespace functions
{
    double sphere(const Vector &x);
}