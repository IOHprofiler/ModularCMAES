#include "matrix_adaptation.hpp"

namespace matrix_adaptation
{
using namespace parameters;



static void one_plus_one_path_update(Vector &path,
                                     const Population &pop,
                                     const parameters::Stats &stats,
                                     const Float c,
                                     const Float sqrt_c_mueff,
                                     const Vector &v)
{
    constexpr static Float max_success_ratio = 0.44;
    if (!stats.has_improved)
        return;

    path = (1.0 - c) * path;
    if (stats.success_ratio < max_success_ratio)
        path += sqrt_c_mueff * v;
}

void Adaptation::adapt_evolution_paths(const Population &pop,
                                       const Weights &w,
                                       const Stats &stats,
                                       const parameters::Settings &settings,
                                       const size_t mu,
                                       const size_t lambda)
{
    const auto sigma = pop.S.leftCols(pop.n).mean();
    dm = (m - m_old) / sigma;
    dz.noalias() = pop.Z.leftCols(mu) * w.positive.head(mu);
    adapt_evolution_paths_inner(pop, w, stats, settings, mu, lambda);
}

void Adaptation::adapt_ps(const parameters::Weights &w)
{
    ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * dz);
}

void None::adapt_evolution_paths_inner(const Population &pop,
                                       const Weights &w,
                                       const Stats &stats,
                                       const parameters::Settings &settings,
                                       const size_t mu,
                                       const size_t lambda)
{
    if (!settings.one_plus_one)
        adapt_ps(w);
}

void None::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) { y = z; }

void None::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) { z = y; }

void CovarianceAdaptation::adapt_ps(const Weights &w)
{
    // Eigen evaluates dst = a + A x as dst = a; dst += A x (the product accumulates into a);
    // done here without its temporary, with the same rounding
    ps *= (1.0 - w.cs);
    ps.noalias() += w.sqrt_cs_mueff * inv_root_C * dm;
}

void CovarianceAdaptation::adapt_evolution_paths_inner(const Population &pop,
                                                       const Weights &w,
                                                       const Stats &stats,
                                                       const parameters::Settings &settings,
                                                       const size_t mu,
                                                       const size_t lambda)
{
    if (settings.one_plus_one)
    {
        one_plus_one_path_update(pc, pop, stats, w.cc, w.sqrt_cc_mueff, pop.Y.col(0));
        return;
    }

    adapt_ps(w);
    const Float actual_ps_length =
        ps.norm() / sqrt(1.0 - pow(1.0 - w.cs, 2.0 * (stats.evaluations / lambda)));

    hs = actual_ps_length < w.expected_length_ps;
    pc = (1.0 - w.cc) * pc + (hs * w.sqrt_cc_mueff) * dm;
}

void CovarianceAdaptation::adapt_covariance_matrix(const Weights &w,
                                                   const Modules &m,
                                                   const Population &pop,
                                                   const size_t mu)
{
    const auto dhs = (1.0 - hs) * w.cc * (2.0 - w.cc);
    const auto &rank_one = w.c1 * pc * pc.transpose();

    const auto &weights = m.active ? w.for_population(pop.n) : w.positive;
    const auto &popY = pop.Y.leftCols(m.active ? pop.n : mu);

    const Float decay = (1 - (w.c1 * dhs) - w.c1 - (w.cmu * weights.sum()));
    const auto &old_c = decay * C;

    Vector rank_mu_w = weights.eval();
    // Negative weights are rescaled by d / ||C^-1/2 y_i||^2 (Hansen, The CMA Evolution Strategy: A
    // Tutorial, 2016, eq. 46); the positive ones, i < mu, are used as they are.
    // ||C^-1/2 y||^2 = y^T C^-1 y = ||A^-1 y||^2 for any factor C = A A^T, so invert_y serves both
    // the eigendecomposition (A = B D) and the Cholesky variant (COVARIANCE_NO_EIGV, which never
    // computes inv_root_C).
    for (Eigen::Index i = static_cast<Eigen::Index>(mu); i < weights.size(); i++)
        rank_mu_w(i) *= dd / invert_y(popY.col(i)).squaredNorm();

    const auto &rank_mu = w.cmu * (popY * rank_mu_w.asDiagonal() * popY.transpose());
    C = old_c + rank_one + rank_mu;

    C = 0.5 * (C + C.transpose().eval());
    coordinate_wise_variances.noalias() = C.diagonal();
}

bool CovarianceAdaptation::perform_eigendecomposition(const Settings &settings)
{
    const Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver(C);

    if (eigen_solver.info() != Eigen::Success)
    {
        if (settings.verbose)
        {
            std::cout << "Eigenvalue solver failed, we need to restart reason:"
                      << eigen_solver.info() << '\n';
        }
        return false;
    }
    d = eigen_solver.eigenvalues();
    B = eigen_solver.eigenvectors();

    if (!d.allFinite() || !B.allFinite() || d.minCoeff() < 0.0)
    {
        if (settings.verbose)
        {
            std::cout << "Negative or non-finite eigenvalues after decomposition, we need to restart.\n";
        }
        return false;
    }

    d.noalias() = d.cwiseSqrt().eval();
    inv_root_C.noalias() = B * d.cwiseInverse().asDiagonal() * B.transpose();
    A.noalias() = B * d.asDiagonal();

    return true;
}

bool CovarianceAdaptation::adapt_matrix_inner(const Weights &w,
                                              const Modules &m,
                                              const Population &pop,
                                              const size_t mu,
                                              const Settings &settings,
                                              parameters::Stats &stats)
{

    if (static_cast<Float>(stats.t) >=
        static_cast<Float>(stats.last_update) + w.lazy_update_interval)
    {
        stats.last_update = stats.t;
        stats.n_updates++;
        adapt_covariance_matrix(w, m, pop, mu);
        auto succ = perform_eigendecomposition(settings);
        if (!succ and settings.verbose)
        {
            std::cout << "t: " << stats.t << ". C:\n";
            std::cout << C << std::endl << std::endl;
        }
        return succ;
    }
    return true;
}

void CovarianceAdaptation::restart(const Settings &settings, const Float sigma)
{
    Adaptation::restart(settings, sigma);
    B = Matrix::Identity(settings.dim, settings.dim);
    C = Matrix::Identity(settings.dim, settings.dim);
    A = Matrix::Identity(settings.dim, settings.dim);
    inv_root_C = Matrix::Identity(settings.dim, settings.dim);
    d.setOnes();
    pc.setZero();
}

void CovarianceAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y.noalias() = A * z;
}

void CovarianceAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    z.noalias() = B.transpose() * y;
    z.array() /= d.array();
}

void SeparableAdaptation::adapt_evolution_paths_inner(const Population &pop,
                                                      const parameters::Weights &w,
                                                      const parameters::Stats &stats,
                                                      const parameters::Settings &settings,
                                                      size_t mu,
                                                      size_t lambda)
{
    if (settings.one_plus_one)
    {
        one_plus_one_path_update(pc, pop, stats, w.cc, w.sqrt_cc_mueff, pop.Y.col(0));
        return;
    }

    adapt_ps(w);

    const Float actual_ps_length =
        ps.norm() / sqrt(1.0 - pow(1.0 - w.cs, 2.0 * (stats.evaluations / lambda)));

    hs = actual_ps_length < w.expected_length_ps;
    pc = (1.0 - w.cc) * pc + (hs * w.sqrt_cc_mueff) * dm;
}

bool SeparableAdaptation::adapt_matrix_inner(const parameters::Weights &w,
                                             const parameters::Modules &m,
                                             const Population &pop,
                                             size_t mu,
                                             const parameters::Settings &settings,
                                             parameters::Stats &stats)
{

    stats.last_update = stats.t;
    stats.n_updates++;

    const auto dhs = (1 - hs) * w.cc * (2.0 - w.cc);

    const auto &weights = m.active ? w.for_population(pop.n) : w.positive;
    const auto &popY = pop.Y.leftCols(m.active ? pop.n : mu);
    const auto decay_c = (1 - (w.c1 * dhs) - w.c1 - (w.cmu * weights.sum()));

    for (size_t j = 0; j < settings.dim; j++)
    {
        const auto rank_mu = (popY.row(j).array().pow(2) * weights.transpose().array()).sum();
        c(j) = (decay_c * c(j)) + (w.c1 * pow(pc(j), 2)) + (w.cmu * rank_mu);
        c(j) = std::max(c(j), Float{1e-12});
        d(j) = std::sqrt(c(j));
        coordinate_wise_variances(j) = c(j);
    }

    return true;
}

void SeparableAdaptation::restart(const parameters::Settings &settings, const Float sigma)
{
    Adaptation::restart(settings, sigma);
    c.setOnes();
    d.setOnes();
    pc.setZero();
}

void SeparableAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y = d.array() * z.array();
}

void SeparableAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    z = y.array() / d.array();
}

void MatrixAdaptation::adapt_evolution_paths_inner(const Population &pop,
                                                   const Weights &w,
                                                   const Stats &stats,
                                                   const parameters::Settings &settings,
                                                   const size_t mu,
                                                   const size_t lambda)
{
    if (settings.one_plus_one && !stats.has_improved)
        return;
    adapt_ps(w);
}

bool MatrixAdaptation::adapt_matrix_inner(const Weights &w,
                                          const Modules &m,
                                          const Population &pop,
                                          const size_t mu,
                                          const Settings &settings,
                                          parameters::Stats &stats)
{

    stats.last_update = stats.t;
    stats.n_updates++;

    const auto &weights = m.active ? w.for_population(pop.n) : w.positive;
    const auto &popZ = pop.Z.leftCols(m.active ? pop.n : mu);
    const auto &popY = pop.Y.leftCols(m.active ? pop.n : mu);

    // Normal MA-ES -> O(n^3)
    //
    // constexpr Float epsilon = 1e-10;
    // const auto& Z = popZ * weights.asDiagonal() * popZ.transpose();
    // ZwI.noalias() = (w.cmu / 2.0) * (Z - I);
    // ssI.noalias() = (w.c1 / 2.0) * (ps * ps.transpose() - I);
    // M = M * (I + ssI + ZwI);
    // M_inv = (I - ssI - ZwI + epsilon * I) * M_inv;

    // Fast MA-ES -> O(n^2)
    const Float tau_1 = w.c1 / 2.0;
    const Float tau_m = w.cmu / 2.0;
    const Float decay_m = (1.0 - tau_1 - tau_m);

    M = (decay_m * M) + (tau_1 * (M * ps) * ps.transpose()) +
        (popY * (tau_m * weights).asDiagonal() * popZ.transpose());

    if (settings.modules.elitist && !settings.one_plus_one)
        M_inv = (decay_m * M_inv) + (tau_1 * ps * (ps.transpose() * M_inv)) +
                ((popY * (tau_m * weights).asDiagonal()) * (popZ.transpose() * M_inv));
    else
        outdated_M_inv = true; // Rely on moore penrose pseudo-inv (only when needed)

    coordinate_wise_variances.noalias() = (M.array().square().rowwise().sum()).matrix();
    return true;
}

void MatrixAdaptation::restart(const Settings &settings, const Float sigma)
{
    Adaptation::restart(settings, sigma);
    M = Matrix::Identity(settings.dim, settings.dim);
    M_inv = Matrix::Identity(settings.dim, settings.dim);
    outdated_M_inv = false;
}

void MatrixAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y.noalias() = M * z;
}

void MatrixAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    if (outdated_M_inv)
    {
        M_inv = M.completeOrthogonalDecomposition().pseudoInverse();
        outdated_M_inv = false;
    }
    z.noalias() = M_inv * y;
}

void CholeskyAdaptation::adapt_evolution_paths_inner(const Population &pop,
                                                     const parameters::Weights &w,
                                                     const parameters::Stats &stats,
                                                     const parameters::Settings &settings,
                                                     size_t mu,
                                                     size_t lambda)
{
    if (settings.one_plus_one)
    {
        one_plus_one_path_update(pc, pop, stats, w.cc, w.sqrt_cc_mueff, pop.Y.col(0));
        return;
    }

    adapt_ps(w);
    pc = (1.0 - w.cc) * pc + (w.sqrt_cc_mueff * dm);
}

bool CholeskyAdaptation::adapt_matrix_inner(const parameters::Weights &w,
                                            const parameters::Modules &m,
                                            const Population &pop,
                                            size_t mu,
                                            const parameters::Settings &settings,
                                            parameters::Stats &stats)
{

    stats.last_update = stats.t;
    stats.n_updates++;

    A *= std::sqrt(1 - w.c1 - w.cmu);

    Eigen::internal::llt_rank_update_lower(A, pc, w.c1);
    for (size_t i = 0; i < mu; i++)
        Eigen::internal::llt_rank_update_lower(A, pop.Y.col(i), w.cmu * w.positive(i));

    if (m.active)
    {
        const Vector weights = w.for_population(pop.n);
        for (size_t i = mu; i < pop.n; i++)
            Eigen::internal::llt_rank_update_lower(A, pop.Y.col(i), w.cmu * weights(i));
    }

    coordinate_wise_variances.noalias() = (A.array().square().rowwise().sum()).matrix();
    return true;
}

void CholeskyAdaptation::restart(const parameters::Settings &settings, const Float sigma)
{
    Adaptation::restart(settings, sigma);
    A = Matrix::Identity(settings.dim, settings.dim);
    pc.setZero();
}

void CholeskyAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y.noalias() = A * z;
}

void CholeskyAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    z = y;
    A.triangularView<Eigen::Lower>().solveInPlace(z);
}

void SelfAdaptation::adapt_evolution_paths_inner(const Population &pop,
                                                 const parameters::Weights &w,
                                                 const parameters::Stats &stats,
                                                 const parameters::Settings &settings,
                                                 size_t mu,
                                                 size_t lambda)
{

    if (!settings.one_plus_one)
        adapt_ps(w);
}

bool SelfAdaptation::adapt_matrix_inner(const parameters::Weights &w,
                                        const parameters::Modules &m,
                                        const Population &pop,
                                        size_t mu,
                                        const parameters::Settings &settings,
                                        parameters::Stats &stats)
{
    stats.last_update = stats.t;
    stats.n_updates++;

    const Float tc = 1.0 + (dd * (dd + 1)) / (2.0 * w.mueff);
    const Float tc_inv = 1.0 / tc;

    const auto &weights = m.active ? w.for_population(pop.n) : w.positive;
    const auto &popY = pop.Y.leftCols(m.active ? pop.n : mu);
    const auto &Y = popY * weights.asDiagonal() * popY.transpose();

    C = (1.0 - tc_inv) * C + (tc_inv * Y);
    C = 0.5 * (C + C.transpose().eval());
    coordinate_wise_variances.noalias() = C.diagonal();

    const Eigen::LLT<Matrix> chol(C);
    if (chol.info() != Eigen::Success)
    {
        if (settings.verbose)
            std::cout << "t: " << stats.t
                      << "Cholesky solver failed, we need to restart reason:" << chol.info()
                      << '\n';
        return false;
    }
    A = chol.matrixL();

    return true;
}

void SelfAdaptation::restart(const parameters::Settings &settings, const Float sigma)
{
    Adaptation::restart(settings, sigma);
    A = Matrix::Identity(settings.dim, settings.dim);
    C = Matrix::Identity(settings.dim, settings.dim);
}

void SelfAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y.noalias() = A * z;
}

void SelfAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    z = y;
    A.triangularView<Eigen::Lower>().solveInPlace(z);
}

void CovarianceNoEigvAdaptation::adapt_ps(const Weights &w) { Adaptation::adapt_ps(w); }

bool CovarianceNoEigvAdaptation::perform_eigendecomposition(const parameters::Settings &settings)
{
    const Eigen::LLT<Matrix> chol(C);
    if (chol.info() != Eigen::Success)
    {
        if (settings.verbose)
        {
            std::cout << "Cholesky solver failed, we need to restart reason:" << chol.info()
                      << '\n';
        }
        return false;
    }

    A = chol.matrixL();
    return true;
}

void CovarianceNoEigvAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    z = y;
    A.triangularView<Eigen::Lower>().solveInPlace(z);
}

void NaturalGradientAdaptation::adapt_evolution_paths_inner(const Population &pop,
                                                            const parameters::Weights &w,
                                                            const parameters::Stats &stats,
                                                            const parameters::Settings &settings,
                                                            size_t mu,
                                                            size_t lambda)
{
    if (!settings.one_plus_one)
    {
        adapt_ps(w);
        compute_gradients(pop, w, stats, settings, mu, lambda);
        return;
    }

    if (stats.has_improved)
    {
        const auto &z = pop.Z.col(0);
        G.noalias() = (z * z.transpose() - Matrix::Identity(settings.dim, settings.dim));
    }
}

void NaturalGradientAdaptation::compute_gradients(const Population &pop,
                                                  const parameters::Weights &w,
                                                  const parameters::Stats &stats,
                                                  const parameters::Settings &settings,
                                                  size_t mu,
                                                  size_t lambda)
{
    if (settings.modules.active)
        w.for_population_into(pop.n, weights_scratch);
    else
        weights_scratch = w.positive;
    const Vector &weights = weights_scratch;

    // G += w_i (z_i z_i^T - I), with z_i z_i^T - I evaluated into a buffer
    G.setZero();
    for (int i = 0; i < weights.rows(); ++i)
    {
        const auto &z = pop.Z.col(i);
        square_scratch.noalias() = z * z.transpose();
        square_scratch.diagonal().array() -= 1.0;
        G.noalias() += weights(i) * square_scratch;
    }

    // Remove isotropic (sigma-related) component: make G trace-free
    sigma_g = (G.trace() / dd);

    if (!settings.one_plus_one)
        G.diagonal().array() -= sigma_g;

    // Ensure symmetry for numerical stability
    square_scratch = G.transpose();
    G = 0.5 * (G + square_scratch);
}

bool NaturalGradientAdaptation::adapt_matrix_inner(const parameters::Weights &w,
                                                   const parameters::Modules &m,
                                                   const Population &pop,
                                                   size_t mu,
                                                   const parameters::Settings &settings,
                                                   parameters::Stats &stats)
{

    stats.last_update = stats.t;
    stats.n_updates++;

    A *= (w.cc * G).exp();
    outdated_A_inv = true;
    coordinate_wise_variances.noalias() = (A.array().square().rowwise().sum()).matrix();

    return true;
}

void NaturalGradientAdaptation::restart(const parameters::Settings &settings, const Float sigma)
{
    Adaptation::restart(settings, sigma);
    A = Matrix::Identity(settings.dim, settings.dim) / sigma;
    A_inv = Matrix::Identity(settings.dim, settings.dim);
    G = Matrix::Zero(settings.dim, settings.dim);
    outdated_A_inv = false;
    sigma_g = 0.;
}

void NaturalGradientAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y.noalias() = A * z;
}

void NaturalGradientAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    if (outdated_A_inv)
    {
        A_inv = A.completeOrthogonalDecomposition().pseudoInverse();
        outdated_A_inv = false;
    }
    z.noalias() = A_inv * y;
}
} // namespace matrix_adaptation
