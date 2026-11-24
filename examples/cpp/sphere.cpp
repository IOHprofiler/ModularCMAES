// Minimize a function with the C++ library: run(), and the same with ask-tell.
#include <iostream>

#include "c_maes.hpp"

int main()
{
    const size_t d = 10;

    // The objective: f(x) = sum(x^2), for a point x (Eigen vector of length d)
    FunctionType sphere = [](Eigen::Ref<const Vector> x) { return x.squaredNorm(); };

    // The modules (here: MA-ES with IPOP restarts) and the settings
    parameters::Modules modules;
    modules.matrix_adaptation = parameters::MatrixAdaptationType::MATRIX;
    modules.restart_strategy = parameters::RestartStrategyType::IPOP;

    parameters::Settings settings(d, modules, /* target */ 1e-10, std::nullopt,
                                  /* budget */ 20000);
    settings.seed = 1;

    ModularCMAES es(settings);
    es(sphere); // run until a break condition holds
    std::cout << "run:      f = " << es.p->stats.global_best.y << " after "
              << es.p->stats.evaluations << " evaluations\n";

    // Ask-tell: evaluate the points (the columns of a d x k matrix) yourself
    ModularCMAES es2(settings);
    while (true)
    {
        const Matrix X = es2.ask();
        Vector f(X.cols());
        for (Eigen::Index i = 0; i < X.cols(); ++i)
            f(i) = sphere(X.col(i));
        if (es2.tell(f) && es2.break_conditions())
            break;
    }
    std::cout << "ask-tell: f = " << es2.p->stats.global_best.y << " after "
              << es2.p->stats.evaluations << " evaluations\n";
    return 0;
}
