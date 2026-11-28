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
 * The weights of the matrix update, into out: with `active` those of all n points of the
 * population (Weights::for_population_into), otherwise the mu positive ones. With pairwise
 * selection the losers get no negative weight. With elitism the negative weights go to the points
 * that the selection dropped (active_steps, #161): the weight of rank i weighs column i + dropped
 * of the sorted pool, so the losers there are the columns from the pool's n_ranked on.
 */
static void update_weights(const Weights &w, const Population &pop, const bool active, Vector &out)
{
    if (!active)
    {
        out = w.positive;
        return;
    }
    const size_t dropped = pop.n_dropped();
    const size_t ranked = std::min(pop.n_ranked, pop.n + dropped);
    w.for_population_into(pop.n, out, ranked > dropped ? ranked - dropped : 0);
}

/**
 * The steps that the n weights of the matrix update weigh (#161), from steps (pop.Y or pop.Z):
 * the first n columns, except with elitism, where the population is the n best of the sorted
 * (mu + lambda) pool. Then the negative weights, from rank n_positive on, go to the worst points of
 * the pool, which the selection dropped (columns pool - (n - n_positive)..pool - 1): the framework
 * paper (de Nobel et al., Sec. "Active Update") gives them to "the lambda - mu worst individuals",
 * as Jastrebski & Arnold (2006) use "information about unsuccessful offspring", and pycma with
 * CMA_elitist gives a population longer than lambda zeros between the positive and the negative
 * weights (recombination_weights.py:202-203), so the negative ones stay on the worst offspring.
 * Ranks mu+1..lambda of the pool are parents and good offspring: their negative weights removed
 * variance along the flat directions (COVARIANCE with elitism and active solved BBOB f10 and f12
 * at d = 10 in 0 of 25 runs). The steps are gathered into scratch only then; returns whether.
 */
static bool active_steps(const Matrix &steps,
                         const Population &pop,
                         const Eigen::Index n,
                         const Eigen::Index n_positive,
                         Matrix &scratch)
{
    const auto dropped = static_cast<Eigen::Index>(pop.n_dropped());
    if (dropped == 0 || n_positive >= n)
        return false;
    scratch.resize(steps.rows(), n);
    scratch.leftCols(n_positive) = steps.leftCols(n_positive);
    scratch.rightCols(n - n_positive) = steps.middleCols(n_positive + dropped, n - n_positive);
    return true;
}

/**
 * h_sigma (Hansen, The CMA Evolution Strategy: A Tutorial, 2016, after Eq. (45)): whether ||ps||,
 * corrected for the generations of the run, is below (1.4 + 2 / (d + 1)) E||N(0, I)||. When it is
 * not, the rank-one update is stalled: this "prevents a too fast increase of axes of C in a linear
 * surrounding, i.e. when the step-size is far too small". The correction counts the generations of
 * the current run, g + 1 in generation g (pycma's countiter).
 */
static bool short_path(const Vector &ps, const Weights &w, const Stats &stats)
{
    const Float actual_ps_length =
        ps.norm() / sqrt(1.0 - pow(1.0 - w.cs_path, 2.0 * static_cast<Float>(stats.t_run + 1)));
    return actual_ps_length < w.expected_length_ps;
}

/**
 * With StepSizeAdaptation XNES and CMA-ES's weights (DEFAULT, EQUAL, EXPONENTIAL), h_sigma = 0
 * also stalls the negative weights of the active update (#175, the maintainer's decision): the
 * matrix update of that generation takes the first n_positive (non-negative) weights of w, copied
 * into positive (a buffer of their own: resizing w allocated in every such generation), and is the
 * update without `active`. Returns whether; not without `active`, when h_sigma = 1, or with
 * UTILITY.
 *
 * h_sigma = 0 marks "a linear surrounding, i.e. when the step-size is far too small" (Hansen, The
 * CMA Evolution Strategy: A Tutorial, 2016, after Eq. (45)). There the ranking reflects the sign
 * of a step along the gradient, not the curvature: the worst steps are as long along the gradient
 * as the best ones, and the negative weights (heaviest on the worst ranks; their sum is
 * -(1 + c1 / cmu), -1.76 at d = 10, against +1) took back most of the growth of C towards the
 * optimum that the positive ones gave, while h_sigma = 0 stalled the rank-one term as well.
 * xNES SA keeps sigma too small (Krause, Glasmachers & Igel, FOGA 2017, Sec. 6): h_sigma was 0 in
 * 89-99 % of the generations at d >= 10, and COVARIANCE + XNES + active solved the ellipsoid
 * (d = 5, 10, 20) in 20, 1 and 0 of 20 runs; with this rule in 20 of 20 at each, in fewer
 * evaluations than without `active`. UTILITY's negative weights are flat (-1 / lambda) and weigh
 * as much as its positive ones (Glasmachers et al. 2010): it solved the ellipsoid without the
 * rule, and with it needed 15-20 % more evaluations (11 of 20 at d = 20). This is our design, not
 * a published rule: the tutorial and pycma 4.6.0 keep the negative weights for every h_sigma, and
 * so does this code with the other step-size rules, which keep sigma near its optimum and h_sigma
 * at 1 most of the time.
 */
static bool stall_negative_weights(const Vector &w,
                                   const Eigen::Index n_positive,
                                   const Modules &m,
                                   const bool hs,
                                   Vector &positive)
{
    if (hs || m.ssa != StepSizeAdaptation::XNES || m.weights == RecombinationWeights::UTILITY ||
        n_positive >= w.size())
        return false;
    positive = w.head(n_positive);
    return true;
}

/**
 * The weights of the matrix update of a generation (update_weights, into all) and the number of
 * non-negative ones among them, n_positive: the negative ones follow them, as the weights decrease
 * with the rank (mu for the usual weights, fewer for UTILITY, whose last parents can have negative
 * utilities). In a generation of XNES with h_sigma = 0 the non-negative ones alone, copied into
 * positive (stall_negative_weights, #175). Returns the weights; the update weighs as many steps
 * (weighed_steps).
 */
static Vector &matrix_update_weights(const Weights &w,
                                     const Population &pop,
                                     const Modules &m,
                                     const bool hs,
                                     Vector &all,
                                     Vector &positive,
                                     Eigen::Index &n_positive)
{
    update_weights(w, pop, m.active, all);
    n_positive = count_non_negative(all);
    return stall_negative_weights(all, n_positive, m, hs, positive) ? positive : all;
}

/**
 * The steps that the n weights of the matrix update weigh, from steps (pop.Y or pop.Z): its first
 * n columns, or with elitism the steps that active_steps gathers into scratch (#161).
 */
static Eigen::Ref<const Matrix> weighed_steps(const Matrix &steps,
                                              const Population &pop,
                                              const Eigen::Index n,
                                              const Eigen::Index n_positive,
                                              Matrix &scratch)
{
    if (active_steps(steps, pop, n, n_positive, scratch))
        return Eigen::Ref<const Matrix>(scratch);
    return Eigen::Ref<const Matrix>(steps.leftCols(n));
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

/**
 * The paths of the adaptations with a rank-one update (COVARIANCE, SEPARABLE, CHOLESKY): ps
 * (adapt_ps), then h_sigma into hs (short_path; true when hs_always), then
 * pc <- (1 - cc) pc + h_sigma sqrt(cc (2 - cc) mueff) dm (Hansen, The CMA Evolution Strategy: A
 * Tutorial, 2016, Eq. (45)). The (1+1)-ES updates only pc, after a success.
 */
static void adapt_rank_one_paths(Adaptation &a,
                                 Vector &pc,
                                 bool &hs,
                                 const bool hs_always,
                                 const Population &pop,
                                 const Weights &w,
                                 const Stats &stats,
                                 const parameters::Settings &settings)
{
    if (settings.one_plus_one)
    {
        one_plus_one_path_update(pc, stats, w.cc, w.sqrt_cc_mueff, pop.Y.col(0));
        return;
    }

    a.adapt_ps(w);
    // The bias correction of h_sigma counts the generations of the current run; it counted
    // evaluations / lambda over all runs
    hs = hs_always || short_path(a.ps, w, stats);
    pc = (1.0 - w.cc) * pc + (hs * w.sqrt_cc_mueff) * a.dm;
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
    adapt_rank_one_paths(*this, pc, hs, false, pop, w, stats, settings);
}

void CovarianceAdaptation::adapt_covariance_matrix(const Weights &w,
                                                   const Modules &m,
                                                   const Population &pop,
                                                   const size_t mu)
{
    const auto dhs = (1.0 - hs) * w.cc * (2.0 - w.cc);

    Eigen::Index n_positive = 0;
    Vector &rank_mu_w =
        matrix_update_weights(w, pop, m, hs, weights_scratch, positive_scratch, n_positive);
    const Eigen::Ref<const Matrix> popY =
        weighed_steps(pop.Y, pop, rank_mu_w.size(), n_positive, steps_scratch);

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
 * eigenvectors as columns into vectors. LAPACK or Eigen, as constants::eigen_solver chooses
 * (lapack::solver: 1 dsyevd, 2 dsyevr). Returns false when the solver fails.
 */
static bool symmetric_eigen(const Matrix &s,
                            Vector &values,
                            Matrix &vectors,
                            Eigen::SelfAdjointEigenSolver<Matrix> &eigen_solver,
                            lapack::Workspace &lapack_workspace,
                            const bool verbose)
{
    const int solver = lapack::solver(constants::eigen_solver, s.rows());
    if (solver != 0)
    {
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

//! inverse = F^-1 for the dense factor F of MATRIX and NATURAL_GRADIENT: an LU inverse (LAPACK)
//! from blas_min_dim on, the pseudo-inverse below or when F is singular
static void invert_factor(const Matrix &F, Matrix &inverse)
{
    if (!(lapack::use_blas(F.rows()) && lapack::inverse(F, inverse)))
        inverse = F.completeOrthogonalDecomposition().pseudoInverse();
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
    adapt_rank_one_paths(*this, pc, hs, false, pop, w, stats, settings);
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

    // With XNES no negative weights in a generation with h_sigma = 0, as COVARIANCE (#175)
    Eigen::Index n_positive = 0;
    Vector &weights = matrix_update_weights(
        w, pop, settings.modules, hs, weights_scratch, positive_scratch, n_positive);
    const Eigen::Ref<const Matrix> popY =
        weighed_steps(pop.Y, pop, weights.size(), n_positive, steps_scratch);
    // The decay with the sum of the weights before the rescaling, negative ones included (Hansen
    // 2016, Eq. (47); pycma's GaussDiagonalSampler.update)
    const auto decay_c = (1 + (w.c1 * dhs) - w.c1 - (w.cmu * weights.sum()));

    // The negative weights rescaled by d / ||D^-1 y_i||^2, the Mahalanobis length in the diagonal
    // C (as COVARIANCE; pycma's GaussDiagonalSampler.update, sampler.py:770-778). They were used
    // as they are (#73, F7)
    if (settings.modules.active)
        rescale_negative_weights(weights, n_positive, dd, [&](Eigen::Index i) {
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
    // h_sigma stalls the rank-one term, which reads ps, as in COVARIANCE (#160), with every
    // step-size rule but CSA. MA-ES (Beyer & Sendhoff 2017; Beyer 2020, (M12)-(M14)) has no
    // h_sigma: CSA updates sigma from the same path, so a long path raises sigma and shortens
    // itself, and MA-ES with CSA stays as published. A rule that keeps sigma too small (XNES, SA)
    // leaves the path long: the rank-one term then stretched M along the recent mean shifts, the
    // selected z became shorter and XNES shrank sigma further (MATRIX + XNES solved the ellipsoid
    // in 0 of 20 runs at d = 5, 10, 20; with h_sigma 17, 20, 20). Not for the (1+1)-ES, whose path
    // moves only on a success (CSA's h_sigma is not defined for it either)
    hs = settings.one_plus_one || settings.modules.ssa == StepSizeAdaptation::CSA ||
         short_path(ps, w, stats);
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

    // With XNES no negative weights in a generation with h_sigma = 0, as COVARIANCE (#175)
    Eigen::Index n_positive = 0;
    Vector &weights = matrix_update_weights(
        w, pop, settings.modules, hs, weights_scratch, positive_scratch, n_positive);
    const auto n = weights.size();
    const Eigen::Ref<const Matrix> popZ = weighed_steps(pop.Z, pop, n, n_positive, steps_scratch);
    const Eigen::Ref<const Matrix> popY = weighed_steps(pop.Y, pop, n, n_positive, steps_scratch2);

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
    // h_sigma = 0 (a step-size rule other than CSA, #160): no rank-one term in this generation
    const Float tau_1 = hs ? w.c1 / 2.0 : 0.0;
    const Float tau_m = w.cmu / 2.0;
    // sum(w) = 1 for the positive weights alone
    const Float decay_m =
        (1.0 - tau_1 - tau_m * (settings.modules.active ? weights.sum() : Float{1.0}));
    // ||M^-1 y_i|| = ||z_i||: z is the sample of y = M z (the elites' and the injected points' z are
    // recomputed with invert_y)
    if (settings.modules.active)
        rescale_negative_weights(weights, n_positive, dd, [&](Eigen::Index i) {
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

    // With elitism, invert_y recomputes the z of the elites every generation: M_inv is kept with M,
    // M_inv <- F^-1 M_inv by the Woodbury identity (update_inverse). The former incremental update,
    // M_inv <- a M_inv + tau_1 ps (ps^T M_inv) + Y diag(tau_m w) (Z^T M_inv), is not the inverse of
    // M F and drifted from it (a relative error of 0.98 after 60 generations), so the elites' z
    // were wrong (#74, F18; the maintainer's decision on #159). The Woodbury update accumulates
    // rounding errors with the active update at small d (above 1e-6 after 1500 generations at
    // d = 10), so up to d = inverse_recompute_max_dim M_inv is computed exactly from M after every
    // inverse_recompute_interval generations of the run (the maintainer's choice, #159; the
    // measurements are in the header). TPA_INPLACE needs M^-1 dm every generation as well (#173).
    // Otherwise, and when the update cannot be applied, the inverse is computed when invert_y needs
    // it (injected and repaired points).
    const bool keep_inverse = settings.modules.elitist ||
                              settings.modules.ssa == StepSizeAdaptation::TPA_INPLACE;
    if (keep_inverse && !settings.one_plus_one)
    {
        if (static_cast<size_t>(M.rows()) <= inverse_recompute_max_dim &&
            (stats.t_run + 1) % inverse_recompute_interval == 0)
            recompute_inverse();
        else if (!outdated_M_inv)
            outdated_M_inv = !update_inverse(popZ, weights, tau_1, tau_m, decay_m);
    }
    else
        outdated_M_inv = true;

    coordinate_wise_variances.noalias() = (M.array().square().rowwise().sum()).matrix();
    return true;
}

bool MatrixAdaptation::update_inverse(Eigen::Ref<const Matrix> Z,
                                      const Vector &w,
                                      const Float tau_1,
                                      const Float tau_m,
                                      const Float a)
{
    // The system is badly conditioned when F is close to singular (the negative weights of the
    // active update can make it so): its error would enter M_inv, so M_inv is then computed from M
    // instead. The test is the ratio of the smallest to the largest pivot of the elimination with
    // partial pivoting (Eigen's rcond() estimate allocates, and costs more than the update at
    // d = 10)
    constexpr Float min_pivot_ratio = 1e-8;
    const auto fallback = [this] {
        ++inverse_fallbacks;
        return false;
    };
    if (!(a > 0.0))
        return fallback();
    const Eigen::Index d = M.rows();
    const Eigen::Index k = Z.cols() + 1;
    const bool blas = lapack::use_blas(d);
    wood_U.resize(d, k);
    wood_U.col(0) = ps;
    wood_U.rightCols(k - 1) = Z;
    wood_D.resize(k);
    wood_D(0) = tau_1;
    wood_D.tail(k - 1) = tau_m * w;

    if (k >= d)
    {
        // At least as many terms as dimensions (large populations): F itself is the smaller
        // system, M_inv <- F^-1 M_inv with an LU of F, O(d^2 k + d^3), not more than the O(k d^2)
        // of the former incremental update
        wood_W.noalias() = wood_U * wood_D.asDiagonal();
        wood_F.resize(d, d);
        if (blas)
            lapack::gemm(1.0, wood_W, false, wood_U, true, 0.0, wood_F);
        else
            wood_F.noalias() = wood_W * wood_U.transpose();
        wood_F.diagonal().array() += a;
        wood_lu.compute(wood_F);
        const auto pivots = wood_lu.matrixLU().diagonal().cwiseAbs();
        if (!(pivots.minCoeff() > min_pivot_ratio * pivots.maxCoeff()))
            return fallback();
        wood_P = wood_lu.solve(M_inv);
        if (!wood_P.allFinite())
            return fallback();
        M_inv.swap(wood_P);
        return true;
    }

    // K = a I_k + U^T U D (k < d), inverted by Gauss-Jordan elimination with partial pivoting:
    // O(k^3), for the k of a population cheaper than a factorization with d right-hand sides
    wood_K.resize(k, k);
    wood_K.noalias() = wood_U.transpose() * wood_U;
    wood_K.array().rowwise() *= wood_D.transpose().array();
    wood_K.diagonal().array() += a;
    // Scalar loops: k is small, and expressions per row cost more than the arithmetic
    wood_Kinv.setIdentity(k, k);
    Float *K = wood_K.data();
    Float *Kinv = wood_Kinv.data();
    const auto at = [k](const Eigen::Index r, const Eigen::Index c) { return r + c * k; };
    Float min_pivot = std::numeric_limits<Float>::infinity(), max_pivot = 0.0;
    for (Eigen::Index c = 0; c < k; ++c)
    {
        Eigen::Index p = c;
        for (Eigen::Index r = c + 1; r < k; ++r)
            if (std::abs(K[at(r, c)]) > std::abs(K[at(p, c)]))
                p = r;
        const Float pivot = std::abs(K[at(p, c)]);
        if (!(pivot > 0.0) || !std::isfinite(pivot))
            return fallback();
        min_pivot = std::min(min_pivot, pivot);
        max_pivot = std::max(max_pivot, pivot);
        if (p != c)
            for (Eigen::Index j = 0; j < k; ++j)
            {
                std::swap(K[at(c, j)], K[at(p, j)]);
                std::swap(Kinv[at(c, j)], Kinv[at(p, j)]);
            }
        const Float inverse_pivot = 1.0 / K[at(c, c)];
        for (Eigen::Index j = 0; j < k; ++j)
        {
            K[at(c, j)] *= inverse_pivot;
            Kinv[at(c, j)] *= inverse_pivot;
        }
        for (Eigen::Index r = 0; r < k; ++r)
        {
            const Float factor = K[at(r, c)];
            if (r == c || factor == 0.0)
                continue;
            for (Eigen::Index j = c; j < k; ++j)
                K[at(r, j)] -= factor * K[at(c, j)];
            for (Eigen::Index j = 0; j < k; ++j)
                Kinv[at(r, j)] -= factor * Kinv[at(c, j)];
        }
    }
    if (!(min_pivot > min_pivot_ratio * max_pivot) || !wood_Kinv.allFinite())
        return fallback();

    // M_inv <- (1/a) (M_inv - (U D K^-1) (U^T M_inv)): two products of O(k d^2), as the former
    // incremental update
    wood_P.resize(k, d);
    if (blas)
        lapack::gemm(1.0, wood_U, true, M_inv, false, 0.0, wood_P);
    else
        wood_P.noalias() = wood_U.transpose() * M_inv;
    wood_U.array().rowwise() *= wood_D.transpose().array(); // U D
    wood_W.noalias() = wood_U * wood_Kinv;
    if (blas)
        lapack::gemm(-1.0, wood_W, false, wood_P, false, 1.0, M_inv);
    else
        M_inv.noalias() -= wood_W * wood_P;
    M_inv *= 1.0 / a;
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
        recompute_inverse();
    z.noalias() = M_inv * y;
}

void MatrixAdaptation::recompute_inverse()
{
    invert_factor(M, M_inv);
    outdated_M_inv = false;
}

void CholeskyAdaptation::adapt_evolution_paths_inner(const Population &pop,
                                                     const parameters::Weights &w,
                                                     const parameters::Stats &stats,
                                                     const parameters::Settings &settings)
{
    // h_sigma stalls the update of pc, as in COVARIANCE (#160), with every step-size rule but CSA
    // (Hansen 2016, after Eq. (45); Krause et al. 2016 adapt sigma by CSA). With XNES the path
    // stayed long, the rank-one term stretched A, and CHOLESKY + XNES solved the ellipsoid in 3, 0
    // and 0 of 20 runs at d = 5, 10, 20 (with h_sigma 20 of 20 at each)
    const bool hs_always = settings.modules.ssa == StepSizeAdaptation::CSA;
    adapt_rank_one_paths(*this, pc, hs, hs_always, pop, w, stats, settings);
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
    // zero), of the mu best otherwise; with elitism the negative ones weigh the worst points of the
    // pool (#161); with XNES none in a generation with h_sigma = 0, as COVARIANCE (#175)
    Eigen::Index n_positive = 0;
    Vector &weights = matrix_update_weights(
        w, pop, settings.modules, hs, weights_scratch, positive_scratch, n_positive);
    const Eigen::Ref<const Matrix> popY =
        weighed_steps(pop.Y, pop, weights.size(), n_positive, steps_scratch);

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
            invert_y_into(popY.col(i), scratch_vector);
            return scratch_vector.squaredNorm();
        });
    }

    // 1 + c1 delta(h_sigma) - c1 - cmu sum(w): the variance that a stalled pc loses is compensated,
    // as in COVARIANCE (#160)
    const Float dhs = (1.0 - hs) * w.cc * (2.0 - w.cc);
    A *= std::sqrt(1 + (w.c1 * dhs) - w.c1 - w.cmu * sum_w);
    Eigen::internal::llt_rank_update_lower(A, pc, w.c1);
    for (Eigen::Index i = 0; i < n_positive; i++)
        if (weights(i) > 0.0)
            Eigen::internal::llt_rank_update_lower(A, popY.col(i), w.cmu * weights(i));

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
                                             A, popY.col(i), w.cmu * weights(i)) >= 0)
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

    update_weights(w, pop, settings.modules.active, weights_scratch);
    const Eigen::Index n_positive = count_non_negative(weights_scratch);
    const Eigen::Ref<const Matrix> popY =
        weighed_steps(pop.Y, pop, weights_scratch.size(), n_positive, steps_scratch);

    // C = (1 - 1/tc) C + (1/tc) Y diag(w) Y^T, as symmetric rank updates
    symmetric_rank_update(C,
                          1.0 - tc_inv,
                          nullptr,
                          0.0,
                          popY,
                          weights_scratch,
                          n_positive,
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
    update_weights(w, pop, settings.modules.active, weights_scratch);
    const Vector &weights = weights_scratch;
    const auto n = weights.rows();
    const Eigen::Index n_positive = count_non_negative(weights);

    // G = sum_i w_i (z_i z_i^T - I) = Z diag(w) Z^T - (sum_i w_i) I, kept as its factor; with
    // elitism the negatively weighted samples are the worst points of the pool (#161)
    if (!active_steps(pop.Z, pop, n, n_positive, gradient_samples))
        gradient_samples = pop.Z.leftCols(n);
    gradient_weights = weights;

    if (dense_update(n))
    {
        // G formed as a matrix, for the dense matrix exponential
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
        invert_factor(A, A_inv);
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
