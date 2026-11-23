#include "matrix_adaptation.hpp"

namespace matrix_adaptation
{
using namespace parameters;

/**
 * S = decay S + c1 v v^T + cmu Y diag(w) Y^T on the lower triangle of S, as rank updates (half the
 * flops of the full products), then mirrored to the upper triangle. The weights of the first
 * n_positive columns are >= 0, the others <= 0 (the order of a sorted population). v may be null.
 */
static void symmetric_rank_update(Matrix &S,
                                  const Float decay,
                                  const Vector *v,
                                  const Float c1,
                                  Eigen::Ref<const Matrix> Y,
                                  Eigen::Ref<const Vector> w,
                                  const Eigen::Index n_positive,
                                  const Float cmu,
                                  Matrix &scratch)
{
    S.triangularView<Eigen::Lower>() *= decay;
    auto lower = S.selfadjointView<Eigen::Lower>();
    if (v != nullptr)
        lower.rankUpdate(*v, c1);

    // y_i sqrt(|w_i|): the positive and the negative weights are two rank updates
    scratch.resize(Y.rows(), Y.cols());
    for (Eigen::Index i = 0; i < Y.cols(); ++i)
        scratch.col(i) = std::sqrt(std::abs(w(i))) * Y.col(i);
    const Eigen::Index n_pos = std::min(n_positive, Y.cols());
    if (n_pos > 0)
        lower.rankUpdate(scratch.leftCols(n_pos), cmu);
    if (Y.cols() > n_pos)
        lower.rankUpdate(scratch.rightCols(Y.cols() - n_pos), -cmu);

    S.triangularView<Eigen::StrictlyUpper>() = S.transpose();
}



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

void None::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) { Y = Z; }

void None::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) { z = y; }

void CovarianceAdaptation::adapt_ps(const Weights &w)
{
    // Eigen evaluates dst = a + A x as dst = a; dst += A x (the product accumulates into a);
    // done here without its temporary, with the same rounding
    // ps = (1 - cs) ps + sqrt(cs mueff) C^-1/2 dm, with C^-1/2 dm = B D^-1 B^T dm (two
    // matrix-vector products instead of forming C^-1/2)
    scratch_vector.noalias() = B.transpose() * dm;
    scratch_vector.array() /= d.array();
    ps *= (1.0 - w.cs);
    ps.noalias() += w.sqrt_cs_mueff * (B * scratch_vector);
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

    if (m.active)
        w.for_population_into(pop.n, weights_scratch);
    else
        weights_scratch = w.positive;
    Vector &rank_mu_w = weights_scratch;
    const auto popY = pop.Y.leftCols(m.active ? pop.n : mu);

    const Float decay = (1 - (w.c1 * dhs) - w.c1 - (w.cmu * rank_mu_w.sum()));

    // Negative weights are rescaled by d / ||C^-1/2 y_i||^2 (Hansen, The CMA Evolution Strategy: A
    // Tutorial, 2016, eq. 46); the positive ones, i < mu, are used as they are.
    // ||C^-1/2 y||^2 = y^T C^-1 y = ||A^-1 y||^2 for any factor C = A A^T, so invert_y serves both
    // the eigendecomposition (A = B D) and the Cholesky variant (COVARIANCE_NO_EIGV, which never
    // computes inv_root_C).
    scratch_vector.resize(popY.rows()); // invert_y_into writes through a Ref: it cannot resize
    for (Eigen::Index i = static_cast<Eigen::Index>(mu); i < rank_mu_w.size(); i++)
    {
        invert_y_into(popY.col(i), scratch_vector);
        rank_mu_w(i) *= dd / scratch_vector.squaredNorm();
    }

    // C = decay C + c1 pc pc^T + cmu Y diag(w) Y^T, as symmetric rank updates
    symmetric_rank_update(C,
                          decay,
                          &pc,
                          w.c1,
                          popY,
                          rank_mu_w,
                          static_cast<Eigen::Index>(mu),
                          w.cmu,
                          batch_scratch);
    coordinate_wise_variances.noalias() = C.diagonal();
}

bool CovarianceAdaptation::perform_eigendecomposition(const Settings &settings)
{
    eigen_solver.compute(C);

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

    d = d.cwiseSqrt();
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
    d.setOnes();
    pc.setZero();
}

void CovarianceAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y.noalias() = A * z;
}

void CovarianceAdaptation::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
{
    Y.noalias() = A * Z;
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

    if (m.active)
        w.for_population_into(pop.n, weights_scratch);
    else
        weights_scratch = w.positive;
    const Vector &weights = weights_scratch;
    const auto popY = pop.Y.leftCols(m.active ? pop.n : mu);
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

void SeparableAdaptation::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
{
    Y = d.asDiagonal() * Z;
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

    if (m.active)
        w.for_population_into(pop.n, weights_scratch);
    else
        weights_scratch = w.positive;
    const Vector &weights = weights_scratch;
    const auto popZ = pop.Z.leftCols(m.active ? pop.n : mu);
    const auto popY = pop.Y.leftCols(m.active ? pop.n : mu);

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

    // M = decay M + tau_1 (M ps) ps^T + Y diag(tau_m w) Z^T, in place
    batch_scratch.noalias() = popY * (tau_m * weights).asDiagonal();
    scratch_vector.noalias() = M * ps;
    M *= decay_m;
    M.noalias() += (tau_1 * scratch_vector) * ps.transpose();
    M.noalias() += batch_scratch * popZ.transpose();

    if (settings.modules.elitist && !settings.one_plus_one)
    {
        // M_inv = decay M_inv + tau_1 ps (ps^T M_inv) + (Y diag(tau_m w)) (Z^T M_inv)
        scratch_vector.noalias() = M_inv.transpose() * ps;
        batch_scratch2.noalias() = popZ.transpose() * M_inv;
        M_inv *= decay_m;
        M_inv.noalias() += (tau_1 * ps) * scratch_vector.transpose();
        M_inv.noalias() += batch_scratch * batch_scratch2;
    }
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

void MatrixAdaptation::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
{
    Y.noalias() = M * Z;
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
        w.for_population_into(pop.n, weights_scratch);
        for (size_t i = mu; i < pop.n; i++)
            Eigen::internal::llt_rank_update_lower(A, pop.Y.col(i), w.cmu * weights_scratch(i));
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
    y.noalias() = A.triangularView<Eigen::Lower>() * z;
}

void CholeskyAdaptation::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
{
    Y.noalias() = A.triangularView<Eigen::Lower>() * Z;
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

    if (m.active)
        w.for_population_into(pop.n, weights_scratch);
    else
        weights_scratch = w.positive;
    const auto popY = pop.Y.leftCols(m.active ? pop.n : mu);

    // C = (1 - 1/tc) C + (1/tc) Y diag(w) Y^T, as symmetric rank updates
    symmetric_rank_update(C,
                          1.0 - tc_inv,
                          nullptr,
                          0.0,
                          popY,
                          weights_scratch,
                          static_cast<Eigen::Index>(mu),
                          tc_inv,
                          batch_scratch);
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
    y.noalias() = A.triangularView<Eigen::Lower>() * z;
}

void SelfAdaptation::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
{
    Y.noalias() = A.triangularView<Eigen::Lower>() * Z;
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

void CovarianceNoEigvAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y.noalias() = A.triangularView<Eigen::Lower>() * z;
}

void CovarianceNoEigvAdaptation::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
{
    Y.noalias() = A.triangularView<Eigen::Lower>() * Z;
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

    // G = sum_i w_i (z_i z_i^T - I) = Z diag(w) Z^T - (sum_i w_i) I, as symmetric rank updates
    G.setZero();
    symmetric_rank_update(G,
                          1.0,
                          nullptr,
                          0.0,
                          pop.Z.leftCols(weights.rows()),
                          weights,
                          static_cast<Eigen::Index>(mu),
                          1.0,
                          batch_scratch);
    G.diagonal().array() -= weights.sum();

    // Remove isotropic (sigma-related) component: make G trace-free
    sigma_g = (G.trace() / dd);

    if (!settings.one_plus_one)
        G.diagonal().array() -= sigma_g;
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

void NaturalGradientAdaptation::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
{
    Y.noalias() = A * Z;
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
