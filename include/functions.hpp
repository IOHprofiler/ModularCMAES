#pragma once

#include <string>

#include "common.hpp"

//! Test functions (minimum 0) and a random rotation, for examples and quick experiments in C++.
//! examples/cpp/functions.cpp runs them, shifted and rotated, with several matrix adaptations.
namespace functions
{
//! sum x_i^2; minimum at 0
Float sphere(Eigen::Ref<const Vector> x);
//! sum 10^(6 (i - 1)/(d - 1)) x_i^2 (condition number 10^6); minimum at 0
Float ellipse(Eigen::Ref<const Vector> x);
//! 10 d + sum (x_i^2 - 10 cos(2 pi x_i)); minimum at 0, about 10^d local minima
Float rastrigin(Eigen::Ref<const Vector> x);
//! sum 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2; minimum at (1, ..., 1)
Float rosenbrock(Eigen::Ref<const Vector> x);

//! A random rotation (orthogonal, determinant 1) of R^n: the Q of a QR decomposition of a Gaussian
//! matrix, from a generator of its own (so it does not change the optimizer's random numbers)
Matrix random_rotation_matrix(int n, int seed);

enum ObjectiveFunction
{
    ELLIPSE,
    ROSENBROCK,
    SPHERE,
    RASTRIGIN
};

inline FunctionType get(const ObjectiveFunction f)
{
    switch (f)
    {
    case ELLIPSE:
        return ellipse;
    case RASTRIGIN:
        return rastrigin;
    case ROSENBROCK:
        return rosenbrock;
    case SPHERE:
    default:
        return sphere;
    }
}

inline std::string to_string(const ObjectiveFunction f)
{
    switch (f)
    {
    case ELLIPSE:
        return "ellipse";
    case RASTRIGIN:
        return "rastrigin";
    case ROSENBROCK:
        return "rosenbrock";
    case SPHERE:
        return "sphere";
    default:
        return "unknown";
    }
}
} // namespace functions
