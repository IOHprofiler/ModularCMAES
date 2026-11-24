#include "functions.hpp"

namespace functions
{
Float sphere(Eigen::Ref<const Vector> x)
{
    return x.squaredNorm();
}

Float rastrigin(Eigen::Ref<const Vector> x)
{
    constexpr Float a = 10.;
    constexpr Float two_pi = 6.283185307179586476925286766559;
    Float res = 0;
    for (const auto xi : x)
        res += xi * xi - a * std::cos(two_pi * xi);
    return a * static_cast<Float>(x.size()) + res;
}

Float ellipse(Eigen::Ref<const Vector> x)
{
    // d = 1: the sphere (the exponent would be 0 / 0)
    const Float denominator = std::max<Float>(1.0, static_cast<Float>(x.size()) - 1.0);
    Float res = 0;
    for (Eigen::Index i = 0; i < x.size(); ++i)
        res += std::pow(1.0e6, static_cast<Float>(i) / denominator) * x(i) * x(i);
    return res;
}

Float rosenbrock(Eigen::Ref<const Vector> x)
{
    Float sum = 0.0;
    for (Eigen::Index i = 0; i + 1 < x.size(); ++i)
    {
        const Float a = x(i + 1) - x(i) * x(i);
        const Float b = 1.0 - x(i);
        sum += 100.0 * a * a + b * b;
    }
    return sum;
}

Matrix random_rotation_matrix(const int n, const int seed)
{
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    std::normal_distribution<Float> d(0, 1);

    Matrix A(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A(i, j) = d(gen);

    const Eigen::HouseholderQR<Matrix> qr(A);
    Matrix Q = qr.householderQ();
    if (Q.determinant() < 0)
        Q.col(0) *= -1;
    return Q;
}
} // namespace functions
