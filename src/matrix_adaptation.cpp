#include "matrix_adaptation.hpp"

namespace matrix_adaptation
{
using namespace parameters;

/**
 * Y = A Z for the batch of samples: BLAS dgemm from constants::blas_min_dim on (A dense, or lower
 * triangular with a zero strictly upper part: dgemm is faster than a triangular product there,
 * with OpenBLAS by far), Eigen below.
 */
static void sample_product(const Matrix &A,
                           const bool lower_triangular,
                           Eigen::Ref<const Matrix> Z,
                           Eigen::Ref<Matrix> Y)
{
    if (lapack::use_blas(A.rows()))
        lapack::gemm(1.0, A, false, Z, false, 0.0, Y);
    else if (lower_triangular)
        Y.noalias() = A.triangularView<Eigen::Lower>() * Z;
    else
        Y.noalias() = A * Z;
}

/**
 * S = decay S + c1 v v^T + cmu Y diag(w) Y^T on the lower triangle of S, as rank updates (half the
 * flops of the full products), then mirrored to the upper triangle. The weights of the first
 * n_positive columns are >= 0, the others <= 0 (the order of a sorted population). v may be null.
 * Eigen only: with the few columns of Y (lambda) the update is bound by the passes over S, and a
 * BLAS dgemm on all of S plus the mirroring was slower at every dimension (docs/performance.md).
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



//! The number of leading non-negative weights (the weights decrease with the rank)
static Eigen::Index count_non_negative(const Vector &w)
{
    Eigen::Index n = 0;
    while (n < w.size() && w(n) >= 0.0)
        ++n;
    return n;
}

static void one_plus_one_path_update(Vector &path,
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
                                       const size_t mu)
{
    const auto sigma = pop.S.leftCols(pop.n).mean();
    dm = (m - m_old) / sigma;
    if (w.utility)
    {
        // The z of the mean shift: all evaluated points with their utilities
        w.mean_weights_into(pop.n, weights_scratch);
        dz.noalias() = pop.Z.leftCols(weights_scratch.size()) * weights_scratch;
    }
    else
        dz.noalias() = pop.Z.leftCols(mu) * w.positive.head(mu);
    adapt_evolution_paths_inner(pop, w, stats, settings);
}

void Adaptation::adapt_ps(const parameters::Weights &w)
{
    ps = (1.0 - w.cs) * ps + (w.sqrt_cs_mueff * dz);
}

void None::adapt_evolution_paths_inner(const Population &pop,
                                       const Weights &w,
                                       const Stats &stats,
                                       const parameters::Settings &settings)
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
                                                       const parameters::Settings &settings)
{
    if (settings.one_plus_one)
    {
        one_plus_one_path_update(pc, stats, w.cc, w.sqrt_cc_mueff, pop.Y.col(0));
        return;
    }

    adapt_ps(w);
    // The bias correction of h_sigma counts the generations of the current run, g + 1 in
    // generation g (Hansen 2016, pycma's countiter); it counted evaluations / lambda over all runs
    const Float actual_ps_length =
        ps.norm() / sqrt(1.0 - pow(1.0 - w.cs, 2.0 * static_cast<Float>(stats.t_run + 1)));

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
        w.for_population_into(pop.n, weights_scratch, pop.n_finite());
    else
        weights_scratch = w.positive;
    Vector &rank_mu_w = weights_scratch;
    const auto popY = pop.Y.leftCols(m.active ? pop.n : mu);

    // 1 + c1 delta(h_sigma) - c1 - cmu sum(w): when h_sigma = 0 (the update of pc was stalled), the
    // variance lost by pc is compensated (Hansen, The CMA Evolution Strategy: A Tutorial, 2016;
    // pycma: c1a = c1 (1 - delta(h_sigma)))
    const Float decay = (1 + (w.c1 * dhs) - w.c1 - (w.cmu * rank_mu_w.sum()));

    // Negative weights are rescaled by d / ||C^-1/2 y_i||^2 (Hansen, The CMA Evolution Strategy: A
    // Tutorial, 2016, eq. 46); the positive ones, i < mu, are used as they are.
    // ||C^-1/2 y||^2 = y^T C^-1 y = ||A^-1 y||^2 for any factor C = A A^T, so invert_y serves both
    // the eigendecomposition (A = B D) and the Cholesky variant (COVARIANCE_NO_EIGV, which never
    // computes inv_root_C).
    scratch_vector.resize(popY.rows()); // invert_y_into writes through a Ref: it cannot resize
    // The negative weights follow the non-negative ones (the weights decrease with the rank): mu
    // for the usual weights, fewer for UTILITY, whose last parents can have negative utilities
    const Eigen::Index n_positive = count_non_negative(rank_mu_w);
    for (Eigen::Index i = n_positive; i < rank_mu_w.size(); i++)
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
                          n_positive,
                          w.cmu,
                          batch_scratch);
    coordinate_wise_variances.noalias() = C.diagonal();
}

bool CovarianceAdaptation::perform_eigendecomposition(const Settings &settings)
{
    const int solver =
        constants::eigen_solver >= 0
            ? constants::eigen_solver
            : (lapack::available() && static_cast<size_t>(C.rows()) >= constants::lapack_min_dim ? 1 : 0);
    if (solver != 0)
    {
        // LAPACK (constants::eigen_solver: 1 dsyevd, 2 dsyevr)
        const auto method = solver == 2 ? lapack::Method::SYEVR : lapack::Method::SYEVD;
        if (!lapack::symmetric_eigen(C, d, B, method, lapack_workspace))
        {
            if (settings.verbose)
                std::cout << "LAPACK eigenvalue solver failed, we need to restart\n";
            return false;
        }
    }
    else
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
    }

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
                                              const Population &pop,
                                              const size_t mu,
                                              const Settings &settings,
                                              parameters::Stats &stats)
{
    // C is updated every generation; only its decomposition is done every lazy_update_interval
    // generations (Hansen 2016, as pycma), which keeps the O(d^3) cost off most generations
    adapt_covariance_matrix(w, settings.modules, pop, mu);
    if (static_cast<Float>(stats.t) >=
        static_cast<Float>(stats.last_update) + w.lazy_update_interval)
    {
        stats.last_update = stats.t;
        stats.n_updates++;
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

void CovarianceAdaptation::restart(const Settings &settings)
{
    Adaptation::restart(settings);
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
    sample_product(A, false, Z, Y);
}

void CovarianceAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    z.noalias() = B.transpose() * y;
    z.array() /= d.array();
}

void SeparableAdaptation::adapt_evolution_paths_inner(const Population &pop,
                                                      const parameters::Weights &w,
                                                      const parameters::Stats &stats,
                                                      const parameters::Settings &settings)
{
    if (settings.one_plus_one)
    {
        one_plus_one_path_update(pc, stats, w.cc, w.sqrt_cc_mueff, pop.Y.col(0));
        return;
    }

    adapt_ps(w);

    const Float actual_ps_length =
        ps.norm() / sqrt(1.0 - pow(1.0 - w.cs, 2.0 * static_cast<Float>(stats.t_run + 1)));

    hs = actual_ps_length < w.expected_length_ps;
    pc = (1.0 - w.cc) * pc + (hs * w.sqrt_cc_mueff) * dm;
}

bool SeparableAdaptation::adapt_matrix_inner(const parameters::Weights &w,
                                             const Population &pop,
                                             size_t mu,
                                             const parameters::Settings &settings,
                                             parameters::Stats &stats)
{

    stats.last_update = stats.t;
    stats.n_updates++;

    const auto dhs = (1 - hs) * w.cc * (2.0 - w.cc);

    if (settings.modules.active)
        w.for_population_into(pop.n, weights_scratch, pop.n_finite());
    else
        weights_scratch = w.positive;
    const Vector &weights = weights_scratch;
    const auto popY = pop.Y.leftCols(settings.modules.active ? pop.n : mu);
    const auto decay_c = (1 + (w.c1 * dhs) - w.c1 - (w.cmu * weights.sum()));

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

void SeparableAdaptation::restart(const parameters::Settings &settings)
{
    Adaptation::restart(settings);
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
                                                   const parameters::Settings &settings)
{
    if (settings.one_plus_one && !stats.has_improved)
        return;
    adapt_ps(w);
}

bool MatrixAdaptation::adapt_matrix_inner(const Weights &w,
                                          const Population &pop,
                                          const size_t mu,
                                          const Settings &settings,
                                          parameters::Stats &stats)
{

    stats.last_update = stats.t;
    stats.n_updates++;

    if (settings.modules.active)
        w.for_population_into(pop.n, weights_scratch, pop.n_finite());
    else
        weights_scratch = w.positive;
    const Vector &weights = weights_scratch;
    const auto popZ = pop.Z.leftCols(settings.modules.active ? pop.n : mu);
    const auto popY = pop.Y.leftCols(settings.modules.active ? pop.n : mu);

    // MA-ES: M <- M (I + c1/2 (ps ps^T - I) + cmu/2 (sum w z z^T - I)), computed in O(d^2 lambda):
    // Fast MA-ES -> O(n^2)
    const Float tau_1 = w.c1 / 2.0;
    const Float tau_m = w.cmu / 2.0;
    // M (I + tau_1 (ps ps^T - I) + tau_m (sum w z z^T - sum(w) I)): sum(w) = 1 for the positive
    // weights; UTILITY sums to zero
    const Float decay_m = (1.0 - tau_1 - tau_m * (w.utility ? weights.sum() : 1.0));

    // M = decay M + tau_1 (M ps) ps^T + Y diag(tau_m w) Z^T, in place
    batch_scratch.noalias() = popY * (tau_m * weights).asDiagonal();
    scratch_vector.noalias() = M * ps;
    M *= decay_m;
    M.noalias() += (tau_1 * scratch_vector) * ps.transpose();
    if (lapack::use_blas(M.rows()))
        lapack::gemm(1.0, batch_scratch, false, popZ, true, 1.0, M);
    else
        M.noalias() += batch_scratch * popZ.transpose();

    if (settings.modules.elitist && !settings.one_plus_one)
    {
        // M_inv = decay M_inv + tau_1 ps (ps^T M_inv) + (Y diag(tau_m w)) (Z^T M_inv)
        scratch_vector.noalias() = M_inv.transpose() * ps;
        const bool blas = lapack::use_blas(M.rows());
        if (blas)
        {
            batch_scratch2.resize(popZ.cols(), M_inv.cols());
            lapack::gemm(1.0, popZ, true, M_inv, false, 0.0, batch_scratch2);
        }
        else
            batch_scratch2.noalias() = popZ.transpose() * M_inv;
        M_inv *= decay_m;
        M_inv.noalias() += (tau_1 * ps) * scratch_vector.transpose();
        if (blas)
            lapack::gemm(1.0, batch_scratch, false, batch_scratch2, false, 1.0, M_inv);
        else
            M_inv.noalias() += batch_scratch * batch_scratch2;
    }
    else
        outdated_M_inv = true; // Rely on moore penrose pseudo-inv (only when needed)

    coordinate_wise_variances.noalias() = (M.array().square().rowwise().sum()).matrix();
    return true;
}

void MatrixAdaptation::restart(const Settings &settings)
{
    Adaptation::restart(settings);
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
    sample_product(M, false, Z, Y);
}

void MatrixAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    if (outdated_M_inv)
    {
        // LU (LAPACK) from blas_min_dim on, the pseudo-inverse below or when M is singular
        if (!(lapack::use_blas(M.rows()) && lapack::inverse(M, M_inv)))
            M_inv = M.completeOrthogonalDecomposition().pseudoInverse();
        outdated_M_inv = false;
    }
    z.noalias() = M_inv * y;
}

void CholeskyAdaptation::adapt_evolution_paths_inner(const Population &pop,
                                                     const parameters::Weights &w,
                                                     const parameters::Stats &stats,
                                                     const parameters::Settings &settings)
{
    if (settings.one_plus_one)
    {
        one_plus_one_path_update(pc, stats, w.cc, w.sqrt_cc_mueff, pop.Y.col(0));
        return;
    }

    adapt_ps(w);
    pc = (1.0 - w.cc) * pc + (w.sqrt_cc_mueff * dm);
}

bool CholeskyAdaptation::adapt_matrix_inner(const parameters::Weights &w,
                                            const Population &pop,
                                            size_t mu,
                                            const parameters::Settings &settings,
                                            parameters::Stats &stats)
{

    stats.last_update = stats.t;
    stats.n_updates++;

    if (w.utility)
    {
        // UTILITY: all evaluated points with their utilities (summing to zero): rank-one updates
        // for the positive ones, downdates for the negative ones
        w.for_population_into(pop.n, weights_scratch, pop.n_finite());
        A *= std::sqrt(1 - w.c1 - w.cmu * weights_scratch.sum());
        Eigen::internal::llt_rank_update_lower(A, pc, w.c1);
        for (Eigen::Index i = 0; i < weights_scratch.size(); i++)
            Eigen::internal::llt_rank_update_lower(A, pop.Y.col(i), w.cmu * weights_scratch(i));
        coordinate_wise_variances.noalias() = (A.array().square().rowwise().sum()).matrix();
        return true;
    }

    A *= std::sqrt(1 - w.c1 - w.cmu);

    Eigen::internal::llt_rank_update_lower(A, pc, w.c1);
    for (size_t i = 0; i < mu; i++)
        Eigen::internal::llt_rank_update_lower(A, pop.Y.col(i), w.cmu * w.positive(i));

    if (settings.modules.active)
    {
        w.for_population_into(pop.n, weights_scratch, pop.n_finite());
        for (size_t i = mu; i < pop.n; i++)
            Eigen::internal::llt_rank_update_lower(A, pop.Y.col(i), w.cmu * weights_scratch(i));
    }

    coordinate_wise_variances.noalias() = (A.array().square().rowwise().sum()).matrix();
    return true;
}

void CholeskyAdaptation::restart(const parameters::Settings &settings)
{
    Adaptation::restart(settings);
    A = Matrix::Identity(settings.dim, settings.dim);
    pc.setZero();
}

void CholeskyAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y.noalias() = A.triangularView<Eigen::Lower>() * z;
}

void CholeskyAdaptation::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
{
    sample_product(A, true, Z, Y);
}

void CholeskyAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    z = y;
    A.triangularView<Eigen::Lower>().solveInPlace(z);
}

void SelfAdaptation::adapt_evolution_paths_inner(const Population &pop,
                                                 const parameters::Weights &w,
                                                 const parameters::Stats &stats,
                                                 const parameters::Settings &settings)
{

    if (!settings.one_plus_one)
        adapt_ps(w);
}

bool SelfAdaptation::adapt_matrix_inner(const parameters::Weights &w,
                                        const Population &pop,
                                        size_t mu,
                                        const parameters::Settings &settings,
                                        parameters::Stats &stats)
{
    stats.last_update = stats.t;
    stats.n_updates++;

    const Float tc = 1.0 + (dd * (dd + 1)) / (2.0 * w.mueff);
    const Float tc_inv = 1.0 / tc;

    if (settings.modules.active)
        w.for_population_into(pop.n, weights_scratch, pop.n_finite());
    else
        weights_scratch = w.positive;
    const auto popY = pop.Y.leftCols(settings.modules.active ? pop.n : mu);

    // C = (1 - 1/tc) C + (1/tc) Y diag(w) Y^T, as symmetric rank updates
    symmetric_rank_update(C,
                          1.0 - tc_inv,
                          nullptr,
                          0.0,
                          popY,
                          weights_scratch,
                          count_non_negative(weights_scratch),
                          tc_inv,
                          batch_scratch);
    coordinate_wise_variances.noalias() = C.diagonal();

    // Eigen's LLT, or LAPACK dpotrf in high dimension (constants::cholesky_solver)
    if (!lapack::cholesky_factor(C, A))
    {
        if (settings.verbose)
            std::cout << "t: " << stats.t
                      << " Cholesky decomposition failed, we need to restart\n";
        return false;
    }

    return true;
}

void SelfAdaptation::restart(const parameters::Settings &settings)
{
    Adaptation::restart(settings);
    A = Matrix::Identity(settings.dim, settings.dim);
    C = Matrix::Identity(settings.dim, settings.dim);
}

void SelfAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y.noalias() = A.triangularView<Eigen::Lower>() * z;
}

void SelfAdaptation::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
{
    sample_product(A, true, Z, Y);
}

void SelfAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    z = y;
    A.triangularView<Eigen::Lower>().solveInPlace(z);
}

void CovarianceNoEigvAdaptation::adapt_ps(const Weights &w) { Adaptation::adapt_ps(w); }

bool CovarianceNoEigvAdaptation::perform_eigendecomposition(const parameters::Settings &settings)
{
    // Eigen's LLT, or LAPACK dpotrf in high dimension (constants::cholesky_solver)
    if (!lapack::cholesky_factor(C, A))
    {
        if (settings.verbose)
            std::cout << "Cholesky decomposition failed, we need to restart\n";
        return false;
    }
    return true;
}

void CovarianceNoEigvAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y.noalias() = A.triangularView<Eigen::Lower>() * z;
}

void CovarianceNoEigvAdaptation::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
{
    sample_product(A, true, Z, Y);
}

void CovarianceNoEigvAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    z = y;
    A.triangularView<Eigen::Lower>().solveInPlace(z);
}

void NaturalGradientAdaptation::adapt_evolution_paths_inner(const Population &pop,
                                                            const parameters::Weights &w,
                                                            const parameters::Stats &stats,
                                                            const parameters::Settings &settings)
{
    if (!settings.one_plus_one)
    {
        adapt_ps(w);
        compute_gradients(pop, w, settings);
        return;
    }

    if (stats.has_improved)
    {
        // G = z z^T - I
        const auto &z = pop.Z.col(0);
        gradient_samples = z;
        gradient_weights = Vector::Ones(1);
        gradient_shift = 1.0;
        if (dense_update(1))
            gradient_matrix.noalias() =
                (z * z.transpose() - Matrix::Identity(settings.dim, settings.dim));
    }
}

bool NaturalGradientAdaptation::dense_update(const Eigen::Index n) const
{
    // The low-rank update pays off when the samples do not span the space and d is not small:
    // below d ~ 16 the dense matrix exponential is cheaper than the QR and the small eigensolver
    // (measured, Apple M5: d = 10 4.3 vs 5.2 us per generation; d = 20 9.6 vs 7.2)
    constexpr Eigen::Index low_rank_min_dim = 16;
    const auto d = A.rows();
    return n >= d || d < low_rank_min_dim;
}

void NaturalGradientAdaptation::compute_gradients(const Population &pop,
                                                  const parameters::Weights &w,
                                                  const parameters::Settings &settings)
{
    if (settings.modules.active)
        w.for_population_into(pop.n, weights_scratch, pop.n_finite());
    else
        weights_scratch = w.positive;
    const Vector &weights = weights_scratch;
    const auto n = weights.rows();

    // G = sum_i w_i (z_i z_i^T - I) = Z diag(w) Z^T - (sum_i w_i) I, kept as its factor
    gradient_samples = pop.Z.leftCols(n);
    gradient_weights = weights;

    if (dense_update(n))
    {
        // G formed as a matrix, for the dense matrix exponential
        const Eigen::Index n_positive = count_non_negative(weights);
        gradient_matrix.setZero(A.rows(), A.cols());
        symmetric_rank_update(gradient_matrix,
                              1.0,
                              nullptr,
                              0.0,
                              gradient_samples,
                              weights,
                              n_positive,
                              1.0,
                              batch_scratch);
        gradient_matrix.diagonal().array() -= weights.sum();

        // Remove the isotropic (sigma-related) component: make G trace-free
        sigma_g = (gradient_matrix.trace() / dd);
        gradient_matrix.diagonal().array() -= sigma_g;
    }
    else
    {
        // trace(G) / d, with trace(Z diag(w) Z^T) = sum_i w_i |z_i|^2
        const Float weighted_square_norms = gradient_samples.colwise().squaredNorm().dot(weights);
        sigma_g = (weighted_square_norms - weights.sum() * dd) / dd;
    }
    // The trace-free G = Z diag(w) Z^T - (sum_i w_i + sigma_g) I
    // (only called for lambda > 1: the (1+1) case is adapt_evolution_paths_inner's)
    gradient_shift = weights.sum() + sigma_g;
}

bool NaturalGradientAdaptation::adapt_matrix_inner(const parameters::Weights &w,
                                                   const Population &pop,
                                                   size_t mu,
                                                   const parameters::Settings &settings,
                                                   parameters::Stats &stats)
{

    stats.last_update = stats.t;
    stats.n_updates++;

    const Float t = w.cc;
    const auto d = A.rows();
    const auto n = gradient_samples.cols();
    if (dense_update(n))
    {
        A *= (t * gradient_matrix).exp();
    }
    else
    {
        // A <- A expm(t G) with G = Q H Q^T - alpha I (thin QR of the samples, Q orthonormal
        // d x k): expm(t G) = e^(-t alpha) (I + U diag(expm1(t lambda)) U^T) with
        // H = V diag(lambda) V^T and U = Q V. Exact, O(d^2 k) instead of a dense O(d^3) matrix
        // exponential.
        if (n > 0)
        {
            const Eigen::HouseholderQR<Matrix> qr(gradient_samples);
            const Eigen::Index k = std::min(d, n);
            const Matrix Q = qr.householderQ() * Matrix::Identity(d, k);
            const Matrix R = qr.matrixQR().topRows(k).triangularView<Eigen::Upper>();
            const Matrix H = R * gradient_weights.asDiagonal() * R.transpose();
            const Eigen::SelfAdjointEigenSolver<Matrix> small(H);
            if (small.info() != Eigen::Success)
                return false;
            const Matrix U = Q * small.eigenvectors();
            const Vector factors = (t * small.eigenvalues()).array().unaryExpr([](Float x) {
                return std::expm1(x);
            });
            if (lapack::use_blas(d))
            {
                batch_scratch.resize(d, U.cols());
                lapack::gemm(1.0, A, false, U, false, 0.0, batch_scratch);
                batch_scratch *= factors.asDiagonal();
                lapack::gemm(1.0, batch_scratch, false, U, true, 1.0, A);
            }
            else
            {
                batch_scratch.noalias() = A * U;
                batch_scratch *= factors.asDiagonal();
                A.noalias() += batch_scratch * U.transpose();
            }
        }
        A *= std::exp(-t * gradient_shift);
    }
    outdated_A_inv = true;
    coordinate_wise_variances.noalias() = (A.array().square().rowwise().sum()).matrix();

    return true;
}

void NaturalGradientAdaptation::restart(const parameters::Settings &settings)
{
    Adaptation::restart(settings);
    A = Matrix::Identity(settings.dim, settings.dim);
    A_inv = Matrix::Identity(settings.dim, settings.dim);
    gradient_samples.resize(settings.dim, 0);
    gradient_weights.resize(0);
    gradient_shift = 0.0;
    gradient_matrix = Matrix::Zero(settings.dim, settings.dim);
    outdated_A_inv = false;
    sigma_g = 0.;
}

void NaturalGradientAdaptation::compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y)
{
    y.noalias() = A * z;
}

void NaturalGradientAdaptation::compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
{
    sample_product(A, false, Z, Y);
}

void NaturalGradientAdaptation::invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z)
{
    if (outdated_A_inv)
    {
        // LU (LAPACK) from blas_min_dim on, the pseudo-inverse below or when A is singular
        if (!(lapack::use_blas(A.rows()) && lapack::inverse(A, A_inv)))
            A_inv = A.completeOrthogonalDecomposition().pseudoInverse();
        outdated_A_inv = false;
    }
    z.noalias() = A_inv * y;
}
} // namespace matrix_adaptation
