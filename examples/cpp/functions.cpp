// The test functions of include/functions.hpp, shifted and rotated, with several matrix adaptations.
#include <iomanip>
#include <iostream>

#include "c_maes.hpp"
#include "functions.hpp"
#include "to_string.hpp"

//! f(R (x - x_opt) + o): the minimum moves to x_opt (here 1, ..., 1) and the axes are rotated;
//! o is the minimum of the unshifted function (1, ..., 1 for Rosenbrock, 0 for the others)
struct Problem
{
    FunctionType function;
    Matrix R;
    Vector x_opt;
    Vector o;

    Problem(const functions::ObjectiveFunction f, const int d, const int seed) :
        function(functions::get(f)),
        R(functions::random_rotation_matrix(d, seed)),
        x_opt(Vector::Ones(d)),
        o(f == functions::ROSENBROCK ? Vector::Ones(d) : Vector::Zero(d))
    {
    }

    Float operator()(Eigen::Ref<const Vector> x) const
    {
        return function(R * (x - x_opt) + o);
    }
};

int main()
{
    const int d = 10;
    const auto all_functions = {functions::SPHERE, functions::ELLIPSE, functions::ROSENBROCK,
                                functions::RASTRIGIN};
    const auto matrix_adaptations = {
        parameters::MatrixAdaptationType::COVARIANCE, parameters::MatrixAdaptationType::SEPARABLE,
        parameters::MatrixAdaptationType::MATRIX, parameters::MatrixAdaptationType::CHOLESKY};

    std::cout << std::left << std::setw(12) << "function" << std::setw(12) << "matrix"
              << std::setw(14) << "evaluations" << "best f\n";
    for (const auto f : all_functions)
    {
        for (const auto ma : matrix_adaptations)
        {
            parameters::Modules modules;
            modules.matrix_adaptation = ma;
            modules.active = true;
            // Rastrigin needs restarts with larger populations
            modules.restart_strategy = parameters::RestartStrategyType::IPOP;

            parameters::Settings settings(d, modules, /* target */ 1e-8, std::nullopt,
                                          /* budget */ 200000);
            settings.seed = 1;

            const Problem problem(f, d, /* seed of the rotation */ 42);
            FunctionType objective = [&problem](Eigen::Ref<const Vector> x) { return problem(x); };
            ModularCMAES es(settings);
            es(objective);

            std::cout << std::setw(12) << functions::to_string(f) << std::setw(12)
                      << parameters::to_string(ma) << std::setw(14) << es.p->stats.evaluations
                      << std::scientific << std::setprecision(2) << es.p->stats.global_best.y
                      << std::defaultfloat << "\n";
        }
    }
    return 0;
}
