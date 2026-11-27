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

/**
 * The active update's rescaling: the weights from n_positive on (the negative ones) are multiplied
 * by d / ||A^-1 y_i||^2, the squared Mahalanobis length of the step in the distribution it was
 * sampled from (Hansen, The CMA Evolution Strategy: A Tutorial, 2016, Eq. (46); pycma's sampler
 * update). squared_norm(i) gives ||A^-1 y_i||^2 = y_i^T C^-1 y_i. The positive weights are used as
 * they are. A zero step adds nothing (y y^T = 0): an elitist parent at the mean (mu = 1), or the
 * mean injected by tell(f, X); its weight stays finite: d / 0 made it inf, and inf * 0 NaN in the
 * update (pycma divides by (||.|| + 1e-9)^2 instead, #118).
 */
template <typename SquaredNorm>
static void rescale_negative_weights(Vector &w,
                                     const Eigen::Index n_positive,
                                     const Float d,
                                     SquaredNorm &&squared_norm)
{
    for (Eigen::Index i = n_positive; i < w.size(); i++)
        if (const Float norm2 = squared_norm(i); norm2 > 0.0)
            w(i) *= d / norm2;
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
    // dm: the step of the mean in units of the step size, (m - m_old) / sigma (Hansen 2016,
    // Eq. (24); the paths use it in (43) and (45)), with sigma per coordinate the step size
    // recombined from the selected points, sum_i w_i s_i (s_i: the column of S of point i; #71,
    // F14, it was the mean of all of S):
    // * every selected point has the step size sigma: (m - m_old) / sigma, the tutorial's;
    // * the lower bound of integer variables (the same in every column): the step divided by the
    //   raised step size of each coordinate, sum_i w_i y_i, as Marty et al. (2024), Eqs. (5)
    //   and (7) (pycma divides by sigma and sigma_vec);
    // * sampled step sizes: the step size recombined from the selected ones, as CMSA recombines
    //   <sigma> (Beyer & Sendhoff 2008; Spettel, Beyer & Hellwig 2018, Alg. 3, line 33).
    // (the recombined step size is computed into dm, then divided into the step: no allocation)
    dm.noalias() = pop.S.leftCols(mu) * w.positive.head(mu);
    dm = (m - m_old).cwiseQuotient(dm);
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
    // The rate cs_path = min(cs, 1): LPXNES's cs can exceed 2 (#72)
    ps = (1.0 - w.cs_path) * ps + (w.sqrt_cs_mueff * dz);
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

void None::start_from(const Matrix &, const parameters::Settings &)
{
    throw std::invalid_argument(
        "Settings: C0 cannot be used with matrix_adaptation NONE, which keeps C = I");
}

void CovarianceAdaptation::adapt_ps(const Weights &w)
{
    // Eigen evaluates dst = a + A x as dst = a; dst += A x (the product accumulates into a);
    // done here without its temporary, with the same rounding
    // ps = (1 - cs) ps + sqrt(cs mueff) C^-1/2 dm, with C^-1/2 dm = B D^-1 B^T dm (two
    // matrix-vector products instead of forming C^-1/2)
    scratch_vector.noalias() = B.transpose() * dm;
    scratch_vector.array() /= d.array();
    ps *= (1.0 - w.cs_path);
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
        ps.norm() / sqrt(1.0 - pow(1.0 - w.cs_path, 2.0 * static_cast<Float>(stats.t_run + 1)));

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
        w.for_population_into(pop.n, weights_scratch, pop.ranked());
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
    // the eigendecomposition (A = B D) and the Cholesky variant (COVARIANCE_NO_EIGV, which has no
    // C^-1/2: no eigendecomposition is kept up to date).
    scratch_vector.resize(popY.rows()); // invert_y_into writes through a Ref: it cannot resize
    // The negative weights follow the non-negative ones (the weights decrease with the rank): mu
    // for the usual weights, fewer for UTILITY, whose last parents can have negative utilities
    const Eigen::Index n_positive = count_non_negative(rank_mu_w);
    rescale_negative_weights(rank_mu_w, n_positive, dd, [&](const Eigen::Index i) {
        invert_y_into(popY.col(i), scratch_vector);
        return scratch_vector.squaredNorm();
    });

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

/**
 * The eigendecomposition of the symmetric S: the eigenvalues (ascending) into values, the
 * eigenvectors as columns into vectors. LAPACK when asked for (constants::eigen_solver 1 or 2) and
 * available, or automatically from lapack_min_dim on; Eigen otherwise (also when LAPACK is asked
 * for but missing, #117). Returns false when the solver fails.
 */
static bool symmetric_eigen(const Matrix &s,
                            Vector &values,
                            Matrix &vectors,
                            Eigen::SelfAdjointEigenSolver<Matrix> &eigen_solver,
                            lapack::Workspace &lapack_workspace,
                            const bool verbose)
{
    const int solver =
        !lapack::available()
            ? 0
            : (constants::eigen_solver >= 0
                   ? constants::eigen_solver
                   : (static_cast<size_t>(s.rows()) >= constants::lapack_min_dim ? 1 : 0));
    if (solver != 0)
    {
        // LAPACK (constants::eigen_solver: 1 dsyevd, 2 dsyevr)
        const auto method = solver == 2 ? lapack::Method::SYEVR : lapack::Method::SYEVD;
        if (!lapack::symmetric_eigen(s, values, vectors, method, lapack_workspace))
        {
            if (verbose)
                std::cout << "LAPACK eigenvalue solver failed, we need to restart\n";
            return false;
        }
        return true;
    }

    eigen_solver.compute(s);
    if (eigen_solver.info() != Eigen::Success)
    {
        if (verbose)
        {
            std::cout << "Eigenvalue solver failed, we need to restart reason:"
                      << eigen_solver.info() << '\n';
        }
        return false;
    }
    values = eigen_solver.eigenvalues();
    vectors = eigen_solver.eigenvectors();
    return true;
}

//! S = F F^T, exactly symmetric: the lower triangle by a rank update, mirrored
static void gram(const Matrix &f, Matrix &s)
{
    s.setZero(f.rows(), f.rows());
    s.selfadjointView<Eigen::Lower>().rankUpdate(f);
    s.triangularView<Eigen::StrictlyUpper>() = s.transpose();
}

Matrix Metric::inv_C() const
{
    const Vector inverse = d.array().square().inverse().matrix();
    if (is_diagonal())
        return inverse.asDiagonal();
    return *B * inverse.asDiagonal() * B->transpose();
}

Matrix Metric::inv_root_C() const
{
    const Vector inverse = d.cwiseInverse();
    if (is_diagonal())
        return inverse.asDiagonal();
    return *B * inverse.asDiagonal() * B->transpose();
}

void decompose_sampled_covariance(const Matrix &s, Vector &axes, Matrix &vectors)
{
    const auto n = s.rows();
    Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver;
    lapack::Workspace lapack_workspace;
    if (!s.allFinite() || !symmetric_eigen(s, axes, vectors, eigen_solver, lapack_workspace, false))
    {
        axes.setConstant(n, std::numeric_limits<Float>::quiet_NaN());
        vectors.setConstant(n, n, std::numeric_limits<Float>::quiet_NaN());
        return;
    }
    // S = F F^T is positive semi-definite: a negative eigenvalue is a rounding error
    axes = axes.cwiseMax(Float{0}).cwiseSqrt();
}

/**
 * The symmetric square root of C0 and its inverse (start_from of MATRIX and NATURAL_GRADIENT,
 * #89): C0^1/2 = B diag(l)^1/2 B^T and C0^-1/2 = B diag(l)^-1/2 B^T for C0 = B diag(l) B^T. It is
 * the one factor of C0 that is symmetric, so it does not depend on the signs and the order of the
 * eigenvectors that the eigensolver returns.
 */
static void symmetric_root(const Matrix &C0, Matrix &root, Matrix &inverse_root)
{
    Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver;
    lapack::Workspace lapack_workspace;
    Vector values;
    Matrix vectors;
    if (!symmetric_eigen(C0, values, vectors, eigen_solver, lapack_workspace, false) ||
        !values.allFinite() || values.minCoeff() <= 0.0)
        throw std::invalid_argument("Settings: C0 must be positive definite");
    const Vector axes = values.cwiseSqrt();
    root.noalias() = vectors * axes.asDiagonal() * vectors.transpose();
    inverse_root.noalias() = vectors * axes.cwiseInverse().asDiagonal() * vectors.transpose();
}

//! The lower Cholesky factor of C0, its strictly upper part zero (start_from of the adaptations
//! that sample with it, #89): the factorization their updates use
static void cholesky_of(const Matrix &C0, Matrix &L)
{
    if (!lapack::cholesky_factor(C0, L))
        throw std::invalid_argument("Settings: C0 must be positive definite");
}

bool CovarianceAdaptation::perform_eigendecomposition(const Settings &settings)
{
    ++factor_version;
    if (!symmetric_eigen(C, d, B, eigen_solver, lapack_workspace, settings.verbose))
        return false;

    // A zero eigenvalue is rejected as a negative one (pycma): adapt_ps and invert_y divide by d
    if (!d.allFinite() || !B.allFinite() || d.minCoeff() <= 0.0)
    {
        if (settings.verbose)
        {
            std::cout << "Non-positive or non-finite eigenvalues after decomposition, we need to "
                         "restart.\n";
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

void CovarianceAdaptation::start_from(const Matrix &C0, const Settings &settings)
{
    Adaptation::start_from(C0, settings);
    C = C0;
    // The decomposition of the update: the eigendecomposition (A = B D) for COVARIANCE, the
    // Cholesky factor for COVARIANCE_NO_EIGV
    if (!perform_eigendecomposition(settings))
        throw std::invalid_argument("Settings: C0 must be positive definite");
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
        ps.norm() / sqrt(1.0 - pow(1.0 - w.cs_path, 2.0 * static_cast<Float>(stats.t_run + 1)));

    hs = actual_ps_length < w.expected_length_ps;
    pc = (1.0 - w.cc) * pc + (hs * w.sqrt_cc_mueff) * dm;
}

bool SeparableAdaptation::adapt_matrix_inner(const parameters::Weights &w,
                                             const Population &pop,
                                             size_t mu,
                                             const parameters::Settings &settings,
                                             parameters::Stats &stats)
{
    ++factor_version;
    stats.last_update = stats.t;
    stats.n_updates++;

    const auto dhs = (1 - hs) * w.cc * (2.0 - w.cc);

    if (settings.modules.active)
        w.for_population_into(pop.n, weights_scratch, pop.ranked());
    else
        weights_scratch = w.positive;
    Vector &weights = weights_scratch;
    const auto popY = pop.Y.leftCols(settings.modules.active ? pop.n : mu);
    // The decay with the sum of the weights before the rescaling, negative ones included (Hansen
    // 2016, Eq. (47); pycma's GaussDiagonalSampler.update)
    const auto decay_c = (1 + (w.c1 * dhs) - w.c1 - (w.cmu * weights.sum()));

    // The negative weights rescaled by d / ||D^-1 y_i||^2, the Mahalanobis length in the diagonal
    // C (as COVARIANCE; pycma's GaussDiagonalSampler.update, sampler.py:770-778). They were used
    // as they are (#73, F7)
    if (settings.modules.active)
        rescale_negative_weights(weights, count_non_negative(weights), dd, [&](Eigen::Index i) {
            return (popY.col(i).array() / d.array()).square().sum();
        });

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

void SeparableAdaptation::start_from(const Matrix &C0, const parameters::Settings &settings)
{
    // Only the diagonal: the variances of C0 (open decision of #89: or reject a C0 that is not
    // diagonal). For the C0 of warm_start, sigma0^2 diag(C0) is the diagonal covariance of the
    // paper's WS-sep-CMA-ES (Nomura et al. 2021, Eq. 6).
    Adaptation::start_from(C0, settings);
    c = C0.diagonal();
    d = c.cwiseSqrt();
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
    ++factor_version;
    stats.last_update = stats.t;
    stats.n_updates++;

    if (settings.modules.active)
        w.for_population_into(pop.n, weights_scratch, pop.ranked());
    else
        weights_scratch = w.positive;
    Vector &weights = weights_scratch;
    const auto popZ = pop.Z.leftCols(settings.modules.active ? pop.n : mu);
    const auto popY = pop.Y.leftCols(settings.modules.active ? pop.n : mu);

    // MA-ES (Beyer & Sendhoff, Simplify Your Covariance Matrix Adaptation Evolution Strategy, IEEE
    // TEVC 2017; Beyer, Design Principles for MA-ES, tutorial 2020, (M13): M <- M [I + 1/tau_1
    // (s s^T - I) + 1/tau_M (<z z^T> - I)], with 1/tau = c/2 from the square root of C's update):
    // M <- M (I + c1/2 (ps ps^T - I) + cmu/2 (sum w z z^T - I)), in O(d^2 lambda).
    // MA-ES has no active update; with `active` it is the active CMA-ES update (Hansen
    // 2016, Eq. (47)) in M, to first order as MA-ES derives its own: with C = M M^T, y = M z,
    // C <- (1 - c1 - cmu sum(w)) C + c1 pc pc^T + cmu sum w°_i y_i y_i^T becomes
    // M <- M (I + c1/2 (ps ps^T - I) + cmu/2 (sum w°_i z_i z_i^T - sum(w) I)), with w° the weights
    // with the negative ones rescaled by d / ||z_i||^2, and sum(w) over the unscaled ones. Before
    // #73 (F7) the decay assumed sum(w) = 1 and the negative weights were not rescaled.
    const Float tau_1 = w.c1 / 2.0;
    const Float tau_m = w.cmu / 2.0;
    // sum(w) = 1 for the positive weights alone
    const Float decay_m =
        (1.0 - tau_1 - tau_m * (settings.modules.active ? weights.sum() : Float{1.0}));
    // ||M^-1 y_i|| = ||z_i||: z is the sample of y = M z (the elites' and the injected points' z are
    // recomputed with invert_y)
    if (settings.modules.active)
        rescale_negative_weights(weights, count_non_negative(weights), dd, [&](Eigen::Index i) {
            return popZ.col(i).squaredNorm();
        });

    // M = decay M + tau_1 (M ps) ps^T + Y diag(tau_m w) Z^T, in place
    batch_scratch.noalias() = popY * (tau_m * weights).asDiagonal();
    scratch_vector.noalias() = M * ps;
    M *= decay_m;
    M.noalias() += (tau_1 * scratch_vector) * ps.transpose();
    if (lapack::use_blas(M.rows()))
        lapack::gemm(1.0, batch_scratch, false, popZ, true, 1.0, M);
    else
        M.noalias() += batch_scratch * popZ.transpose();

    // The inverse is computed again when invert_y needs it (elitism, injected and repaired points).
    // With elitism it was updated incrementally, and drifted from the inverse of M (a relative
    // error of 0.98 after 60 generations), so the elites' z were wrong (#74, F18)
    outdated_M_inv = true;

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

void MatrixAdaptation::start_from(const Matrix &C0, const Settings &settings)
{
    Adaptation::start_from(C0, settings);
    symmetric_root(C0, M, M_inv);
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
    ++factor_version;
    stats.last_update = stats.t;
    stats.n_updates++;

    // The weights: of all evaluated points with `active` (UTILITY sets it; its utilities sum to
    // zero), of the mu best otherwise
    if (settings.modules.active)
        w.for_population_into(pop.n, weights_scratch, pop.ranked());
    else
        weights_scratch = w.positive;
    Vector &weights = weights_scratch;
    const Eigen::Index n_positive = count_non_negative(weights);

    // C <- (1 - c1 - cmu sum(w)) C + c1 pc pc^T + cmu sum_i w°_i y_i y_i^T (Hansen 2016, Eq. (47))
    // on the factor, as the Cholesky-CMA-ES (Krause, Arbonès & Igel, CMA-ES with Optimal
    // Covariance Update and Storage Complexity, NeurIPS 2016): A scaled by the square root of the
    // decay, then a rank-one update of A per term. sum(w) over the unscaled weights, 1 without
    // `active`; w° has the negative weights rescaled by d / ||A^-1 y_i||^2 with the A that sampled
    // them, as COVARIANCE. Before #73 (F7) the decay with `active` ignored the negative weights
    // (except with UTILITY), and they were not rescaled.
    const Float sum_w = settings.modules.active ? weights.sum() : Float{1.0};
    const bool has_negative = n_positive < weights.size();
    if (has_negative)
    {
        scratch_vector.resize(A.rows()); // invert_y_into writes through a Ref: it cannot resize
        rescale_negative_weights(weights, n_positive, dd, [&](const Eigen::Index i) {
            invert_y_into(pop.Y.col(i), scratch_vector);
            return scratch_vector.squaredNorm();
        });
    }

    A *= std::sqrt(1 - w.c1 - w.cmu * sum_w);
    Eigen::internal::llt_rank_update_lower(A, pc, w.c1);
    for (Eigen::Index i = 0; i < n_positive; i++)
        if (weights(i) > 0.0)
            Eigen::internal::llt_rank_update_lower(A, pop.Y.col(i), w.cmu * weights(i));

    // The negative terms, as rank-one downdates. A downdate fails when its result would not be
    // positive definite; Eigen's llt_rank_update_lower then returns the column where it stopped and
    // leaves A partly updated (it was used so, #74, F5). The rescaled weights keep the result
    // positive definite in exact arithmetic (Hansen 2016, Eq. (53)), not UTILITY's, which are not
    // limited for that, nor rounding. Then A from before the downdates is restored and they are
    // applied again without the failing one, which is counted in skipped_downdates
    if (has_negative)
    {
        square_scratch = A;
        for (;;)
        {
            Eigen::Index failed = -1;
            for (Eigen::Index i = n_positive; i < weights.size() && failed < 0; i++)
                if (weights(i) != 0.0 && Eigen::internal::llt_rank_update_lower(
                                             A, pop.Y.col(i), w.cmu * weights(i)) >= 0)
                    failed = i;
            if (failed < 0)
                break;
            A = square_scratch;
            weights(failed) = 0.0;
            ++skipped_downdates;
        }
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

void CholeskyAdaptation::start_from(const Matrix &C0, const parameters::Settings &settings)
{
    Adaptation::start_from(C0, settings);
    cholesky_of(C0, A);
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
    ++factor_version;
    stats.last_update = stats.t;
    stats.n_updates++;

    const Float tc = 1.0 + (dd * (dd + 1)) / (2.0 * w.mueff);
    const Float tc_inv = 1.0 / tc;

    if (settings.modules.active)
        w.for_population_into(pop.n, weights_scratch, pop.ranked());
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

void SelfAdaptation::start_from(const Matrix &C0, const parameters::Settings &settings)
{
    Adaptation::start_from(C0, settings);
    C = C0;
    cholesky_of(C0, A);
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
    ++factor_version;
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
        w.for_population_into(pop.n, weights_scratch, pop.ranked());
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
    ++factor_version;
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

void NaturalGradientAdaptation::start_from(const Matrix &C0, const parameters::Settings &settings)
{
    Adaptation::start_from(C0, settings);
    symmetric_root(C0, A, A_inv);
    outdated_A_inv = false;
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

// -----------------------------------------------------------------------------
// metric() of the adaptations without a decomposition (#123): C = F F^T of the sampling factor F,
// decomposed on request and reused while factor_version is unchanged
// -----------------------------------------------------------------------------

Metric MatrixAdaptation::metric()
{
    return computed_metric(metric_d, metric_B, [this](Matrix &s) { gram(M, s); });
}

Metric CholeskyAdaptation::metric()
{
    return computed_metric(metric_d, metric_B, [this](Matrix &s) {
        gram(Matrix(A.triangularView<Eigen::Lower>()), s);
    });
}

Metric SelfAdaptation::metric()
{
    return computed_metric(metric_d, metric_B, [this](Matrix &s) { s = C; });
}

Metric CovarianceNoEigvAdaptation::metric()
{
    // The C of the last factorization, A A^T, which the samples come from (C can be ahead)
    return computed_metric(
        d, B, [this](Matrix &s) { gram(Matrix(A.triangularView<Eigen::Lower>()), s); });
}

Metric NaturalGradientAdaptation::metric()
{
    return computed_metric(metric_d, metric_B, [this](Matrix &s) { gram(A, s); });
}
} // namespace matrix_adaptation
