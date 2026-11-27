#pragma once

#include "lapack.hpp"

#include "population.hpp"
#include "settings.hpp"
#include "stats.hpp"
#include "weights.hpp"

namespace matrix_adaptation
{
/**
 * Shorten the step of a solution that was not sampled from the current distribution (a point given
 * to tell) to a Mahalanobis length of at most sqrt(d) + 2d / (d + 2) (Hansen 2011, "Injecting
 * external solutions into CMA-ES"; pycma). Returns whether it was shortened. The caller moves the
 * point used for the update accordingly (pycma clips the solution itself), so the mean and the
 * paths stay consistent; the evaluated point and its value are kept for the statistics.
 */
inline bool clip_injected_step(Eigen::Ref<Vector> y, Eigen::Ref<Vector> z)
{
    const Float d = static_cast<Float>(z.size());
    const Float max_length = std::sqrt(d) + 2.0 * d / (d + 2.0);
    const Float length = z.norm();
    if (length <= max_length)
        return false;
    const Float factor = max_length / length;
    z *= factor;
    y *= factor;
    return true;
}

/**
 * The metric of a search distribution: its shape without the step size (#123). A sample is
 * x = m + sigma B (d * z), z ~ N(0, I), so the samples have the covariance sigma^2 C with
 * C = B diag(d)^2 B^T. It describes the distribution that is sampled: for COVARIANCE and
 * COVARIANCE_NO_EIGV the C of the last decomposition, which can be up to lazy_update_interval
 * generations behind their C. Strictly, the metric (the matrix of the Mahalanobis distance) is
 * C^-1; this holds the decomposition, from which C and C^-1 follow.
 *
 * A view, on the adaptation's own vectors (COVARIANCE, SEPARABLE, NONE) or on the metric it
 * computed (the others): valid until the next update of the matrix, restart, or metric() of the
 * adaptation.
 */
struct Metric
{
    //! The axis lengths: the square roots of the eigenvalues of C (pycma's D)
    const Vector &d;
    //! The eigenvectors of C as columns, in the order of d; null when C is diagonal (B = I)
    const Matrix *B;

    [[nodiscard]] bool is_diagonal() const { return B == nullptr; }

    //! The eigenvalues of C, d^2, in the order of d
    [[nodiscard]] Vector eigenvalues() const { return d.array().square().matrix(); }

    //! C^-1 = B diag(d)^-2 B^T, (d, d)
    [[nodiscard]] Matrix inv_C() const;

    //! C^-1/2 = B diag(d)^-1 B^T, (d, d)
    [[nodiscard]] Matrix inv_root_C() const;

    //! An owned copy (what a run's record and Python keep)
    [[nodiscard]] MetricCopy copy() const
    {
        return {d, B != nullptr ? std::optional<Matrix>(*B) : std::nullopt};
    }

    //! A view on an owned copy
    static Metric of(const MetricCopy &copy) { return {copy.d, copy.B ? &*copy.B : nullptr}; }
};

/**
 * The eigendecomposition of a sampled covariance S = F F^T (metric() of the adaptations without a
 * decomposition): axes = the square roots of its eigenvalues (rounding errors below zero clipped
 * to 0), vectors = its eigenvectors. NaN when S is not finite or the eigensolver fails.
 */
void decompose_sampled_covariance(const Matrix &s, Vector &axes, Matrix &vectors);

struct Adaptation
{
    Vector m, m_old, dm, ps, dz;
    Vector coordinate_wise_variances;

    Float dd;

    // Buffers reused by the updates, so that a generation does not allocate
    Vector scratch_vector;
    Vector weights_scratch;
    Matrix square_scratch;
    Matrix batch_scratch;  // d x n
    Matrix batch_scratch2; // n x d

    Adaptation(const size_t dim, const Vector &x0) :
        m(x0),
        m_old(dim),
        dm(Vector::Zero(dim)),
        ps(Vector::Zero(dim)),
        coordinate_wise_variances(Vector::Ones(dim)),
        dd(static_cast<Float>(dim))
    {
    }

    virtual ~Adaptation() = default;

    virtual void adapt_ps(const parameters::Weights &w);

    void adapt_evolution_paths(const Population &pop,
                               const parameters::Weights &w,
                               const parameters::Stats &stats,
                               const parameters::Settings &settings,
                               size_t mu);

    virtual void adapt_evolution_paths_inner(const Population &pop,
                                             const parameters::Weights &w,
                                             const parameters::Stats &stats,
                                             const parameters::Settings &settings) = 0;

    bool adapt_matrix(const parameters::Weights &w,
                      const Population &pop,
                      size_t mu,
                      const parameters::Settings &settings,
                      parameters::Stats &stats)
    {
        if (settings.one_plus_one and !stats.has_improved)
            return true;
        const bool success = adapt_matrix_inner(w, pop, mu, settings, stats);
        // Every matrix adaptation derives coordinate_wise_variances from all entries of its
        // matrix (row sums of squares, or the diagonal of C, whose decomposition is checked
        // separately), so a NaN or inf anywhere in the matrix shows up here. Without this check
        // a run continued with NaN samples until its budget was used up.
        return success && coordinate_wise_variances.allFinite();
    }

    virtual bool adapt_matrix_inner(const parameters::Weights &w,
                                    const Population &pop,
                                    size_t mu,
                                    const parameters::Settings &settings,
                                    parameters::Stats &stats) = 0;

    //! y = A z for the sampling transformation A, written into y (no allocation)
    virtual void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) = 0;

    //! Y = A Z for a block of samples (the columns of Z): one matrix product (T2.8)
    virtual void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y)
    {
        for (Eigen::Index j = 0; j < Z.cols(); ++j)
            compute_y_into(Z.col(j), Y.col(j));
    }

    //! y = A z as a new vector
    [[nodiscard]] Vector compute_y(const Vector &z)
    {
        Vector y(z.size());
        compute_y_into(z, y);
        return y;
    }

    //! y = (x - m) / sigma, written into y
    void invert_x_into(Eigen::Ref<const Vector> x,
                       Eigen::Ref<const Vector> sigma,
                       Eigen::Ref<Vector> y) const
    {
        y = ((x - m).array() / sigma.array()).matrix();
    }

    [[nodiscard]] Vector invert_x(const Vector &x, const Vector &sigma) const
    {
        Vector y(x.size());
        invert_x_into(x, sigma, y);
        return y;
    }

    //! z = A^-1 y for the sampling transformation A, written into z (no allocation)
    virtual void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) = 0;

    [[nodiscard]] Vector invert_y(const Vector &y)
    {
        Vector z(y.size());
        invert_y_into(y, z);
        return z;
    }

    virtual void restart(const parameters::Settings &settings)
    {
        m = settings.x0.value_or(settings.center);
        m_old.setZero();
        dm.setZero();
        ps.setZero();
        dz.setZero();
        coordinate_wise_variances.setOnes();
        ++factor_version;
    }

    /**
     * Start from the covariance matrix C0 instead of I (Settings::C0, warm starting, #89): after
     * the constructor and after restart(). Each adaptation sets its sampling factor F to a factor
     * of C0, F F^T = C0, as its own update keeps it: COVARIANCE the eigendecomposition (F = B D),
     * COVARIANCE_NO_EIGV, CHOLESKY and CMSA the lower Cholesky factor, MATRIX and NATURAL_GRADIENT
     * the symmetric square root C0^1/2, SEPARABLE the square roots of the diagonal of C0; NONE
     * rejects C0 (std::invalid_argument). Two factors of C0 differ by a rotation of z, so the
     * sampled distribution is N(m, sigma^2 C0) with each. C0 is positive definite
     * (Settings::set_C0).
     */
    virtual void start_from(const Matrix &C0, const parameters::Settings &settings)
    {
        coordinate_wise_variances = C0.diagonal();
        ++factor_version;
    }

    Float distance(const Vector u, const Vector &v)
    {
        const auto &delta = u - v;
        return invert_y(delta).norm();
    }

    Float distance_from_center(const Vector &xi) { return distance(m, xi); }

    /**
     * The metric of the sampled distribution (#123). COVARIANCE, SEPARABLE and NONE return a view
     * on what they keep, at no cost. The others decompose C = F F^T of their sampling factor F on
     * request, O(d^3), and reuse it while factor_version is unchanged: nothing is computed unless
     * something asks.
     */
    virtual Metric metric() = 0;

    /**
     * Whether metric() is a view on what the adaptation keeps up to date (COVARIANCE, SEPARABLE,
     * NONE), so that reading it costs nothing. The others decompose their factor on request,
     * O(d^3): the restart criteria that read the metric are not created for them by default
     * (#80).
     */
    [[nodiscard]] virtual bool keeps_metric() const { return false; }

    //! The evolution path p_c of the rank-one update, or null for the adaptations without one
    //! (MATRIX, CMSA, NATURAL_GRADIENT, NONE); TolX reads it
    [[nodiscard]] virtual const Vector *rank_one_path() const { return nullptr; }

    //! Increased wherever the sampling factor changes: an update of the matrix, a decomposition,
    //! a restart. A computed metric is reused while it is unchanged; a write to the matrices from
    //! Python does not increase it.
    size_t factor_version = 0;

  protected:
    /**
     * metric() of the adaptations without a decomposition: the decomposition of the sampled
     * covariance S = F F^T, formed by form_s, into axes and vectors; done only when the factor
     * changed since the last request.
     */
    template <typename FormS> Metric computed_metric(Vector &axes, Matrix &vectors, FormS &&form_s)
    {
        if (metric_version != factor_version)
        {
            Matrix s;
            form_s(s);
            decompose_sampled_covariance(s, axes, vectors);
            metric_version = factor_version;
        }
        return {axes, &vectors};
    }

    //! The storage of a computed metric, allocated on the first request
    Vector metric_d;
    Matrix metric_B;

  private:
    //! The factor_version of the computed metric (none at the start)
    size_t metric_version = std::numeric_limits<size_t>::max();
};

struct None final : Adaptation
{
    //! The axes of the metric: C = I
    Vector ones;

    None(const size_t dim, const Vector &x0) :
        Adaptation(dim, x0),
        ones(Vector::Ones(dim))
    {
    }

    Metric metric() override { return {ones, nullptr}; }

    [[nodiscard]] bool keeps_metric() const override { return true; }

    //! C stays I: std::invalid_argument (Settings::set_C0 rejects C0 with NONE already)
    void start_from(const Matrix &C0, const parameters::Settings &settings) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const Population &pop,
                            const size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override
    {
        return true;
    }

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;
};

struct CovarianceAdaptation : Adaptation
{
    Vector pc, d;
    Matrix B, C;
    Matrix A;
    bool hs = true;

    //! B and d hold the eigendecomposition of C, kept up to date by the update (not for
    //! COVARIANCE_NO_EIGV, which factorizes C with a Cholesky decomposition and fills B and d
    //! only when metric() is asked)
    [[nodiscard]] bool keeps_metric() const override { return true; }

    //! A view on d and B, which the decomposition updates in place (#123)
    Metric metric() override { return {d, &B}; }

    [[nodiscard]] const Vector *rank_one_path() const override { return &pc; }

    // Reused by the matrix update: the eigensolver with its workspace
    Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver;
    lapack::Workspace lapack_workspace;

    CovarianceAdaptation(const size_t dim, const Vector &x0) :
        Adaptation(dim, x0),
        pc(Vector::Zero(dim)),
        d(Vector::Ones(dim)),
        B(Matrix::Identity(dim, dim)),
        C(Matrix::Identity(dim, dim)),
        A(Matrix::Identity(dim, dim))
    {
    }

    void adapt_covariance_matrix(const parameters::Weights &w,
                                 const parameters::Modules &m,
                                 const Population &pop,
                                 size_t mu);

    virtual bool perform_eigendecomposition(const parameters::Settings &settings);

    void adapt_ps(const parameters::Weights &w) override;

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings) override;

    void start_from(const Matrix &C0, const parameters::Settings &settings) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;
};

struct SeparableAdaptation : Adaptation
{
    Vector pc, d, c;
    bool hs;

    SeparableAdaptation(const size_t dim, const Vector &x0) :
        Adaptation(dim, x0),
        pc(Vector::Zero(dim)),
        d(Vector::Ones(dim)),
        c(Vector::Ones(dim)),
        hs(true)
    {
    }

    //! A view on d, which the update keeps (C is diagonal)
    Metric metric() override { return {d, nullptr}; }

    [[nodiscard]] bool keeps_metric() const override { return true; }

    [[nodiscard]] const Vector *rank_one_path() const override { return &pc; }

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings) override;

    void start_from(const Matrix &C0, const parameters::Settings &settings) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;
};

struct MatrixAdaptation final : Adaptation
{
    Matrix M;
    Matrix M_inv;

    MatrixAdaptation(const size_t dim, const Vector &x0) :
        Adaptation(dim, x0),
        M(Matrix::Identity(dim, dim)),
        M_inv(Matrix::Identity(dim, dim)),
        outdated_M_inv(false)
    {
    }

    //! Computed on request from C = M M^T
    Metric metric() override;

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings) override;

    void start_from(const Matrix &C0, const parameters::Settings &settings) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;

    //! The generations with elitism in which the low-rank (Woodbury) update of M_inv could not be
    //! applied (the decay a <= 0, or a singular or badly conditioned k x k system), since the
    //! adaptation was created; M_inv is then computed when invert_y needs it (#74)
    size_t inverse_fallbacks = 0;

    /**
     * With elitism and d <= inverse_recompute_max_dim, M_inv is computed exactly from M after
     * every this many generations of a run (the generations since the last restart), and updated
     * by the Woodbury identity in between. The maintainer's choice on #74 (#159). The update's
     * rounding errors grow with the active update at small d: max |M M_inv - I| over 5000
     * generations without the recomputation (rotated ellipsoid, condition 1e6, 5 seeds): with
     * active 2.4 to 26 at d = 10 (above 1e-10 from generation 785 to 950 on), 1.3e-8 to 6.4e-8 at
     * d = 20, at most 4.3e-11 at d = 30, 6.3e-13 at d = 40, below 1e-13 at d = 50, 100 and 200;
     * without active at most 4.5e-12 at every d. With it, at d = 10 and active at most 3.7e-7.
     */
    static constexpr size_t inverse_recompute_interval = 1000;

    /**
     * The largest dimension with the exact recomputation of M_inv (inverse_recompute_interval):
     * the smallest d at which the drift without it stayed below 1e-10 over 5000 generations in
     * every run measured (see above). The maintainer's choice on #74 (#159). One recomputation
     * costs about 1.3 to 2.9 elitist generations at d = 100 to 1000 (0.07, 0.50, 6.7 ms against
     * 55, 282, 2306 us), 0.1 to 0.3 % per 1000 generations.
     */
    static constexpr size_t inverse_recompute_max_dim = 30;

  private:
    bool outdated_M_inv;

    //! M_inv = M^-1: an LU inverse (LAPACK) from blas_min_dim on, the pseudo-inverse below or
    //! when M is singular
    void recompute_inverse();

    /**
     * M_inv <- F^-1 M_inv for the update M <- M F, F = a I + U D U^T with U = [ps, Z] and
     * D = diag(tau_1, tau_m w): by the Woodbury identity F^-1 = (1/a) [I - U D (a I_k + U^T U
     * D)^-1 U^T], which has no D^-1 and holds for zero and negative weights. O(k d^2) + O(k^3).
     * Returns false (M_inv unchanged) when it cannot be applied.
     */
    bool update_inverse(Eigen::Ref<const Matrix> Z,
                        const Vector &w,
                        Float tau_1,
                        Float tau_m,
                        Float a);

    // Buffers of update_inverse
    Matrix wood_U, wood_K, wood_Kinv, wood_W, wood_P, wood_F;
    Vector wood_D;
    Eigen::PartialPivLU<Matrix> wood_lu;
};

struct CholeskyAdaptation final : Adaptation
{
    Matrix A;
    Vector pc;
    /*
    Only a triangular matrix is stored, so the storage complexity is optimal (Krause et al. 2016).
    The diagonal of A gives det C = prod(A_ii)^2, but not the eigenvalues of C: they are the
    squared singular values of A (for A = [[1, 0], [1, 1]]: 2.618 and 0.382, while A_ii = 1).
    metric() computes them.*/

    //! The rank-one downdates of the active update that were skipped since the adaptation was
    //! created (restarts included), because their result would not be positive definite (#74)
    size_t skipped_downdates = 0;

    CholeskyAdaptation(const size_t dim, const Vector &x0) :
        Adaptation(dim, x0),
        A(Matrix::Identity(dim, dim)),
        pc(Vector::Zero(dim))
    {
    }

    //! Computed on request from C = A A^T
    Metric metric() override;

    [[nodiscard]] const Vector *rank_one_path() const override { return &pc; }

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings) override;

    void start_from(const Matrix &C0, const parameters::Settings &settings) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;
};

struct SelfAdaptation final : Adaptation
{
    Matrix A;
    Matrix C;

    SelfAdaptation(const size_t dim, const Vector &x0) :
        Adaptation(dim, x0),
        A(Matrix::Identity(dim, dim)),
        C(Matrix::Identity(dim, dim))
    {
    }

    //! Computed on request from C, which the update factorizes every generation (C = A A^T)
    Metric metric() override;

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings) override;

    void start_from(const Matrix &C0, const parameters::Settings &settings) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;
};

struct CovarianceNoEigvAdaptation : CovarianceAdaptation
{
    using CovarianceAdaptation::CovarianceAdaptation;

    void adapt_ps(const parameters::Weights &w) override;

    bool perform_eigendecomposition(const parameters::Settings &settings) override;

    //! Its metric is decomposed on request: B and d are not kept up to date
    [[nodiscard]] bool keeps_metric() const override { return false; }

    //! Computed on request from A A^T, the C of the last factorization, into d and B (which this
    //! adaptation does not use otherwise)
    Metric metric() override;

    //! A is the lower triangular Cholesky factor here
    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;
};

struct NaturalGradientAdaptation final : Adaptation
{
    Matrix A;
    Matrix A_inv;
    Float sigma_g;

    /**
     * The natural gradient G = Z diag(u) Z^T - alpha I, kept as its low-rank factor: the samples Z
     * (d x k, k <= lambda), their weights u and the shift alpha. The update A <- A expm(eta G) then
     * costs O(d^2 k) instead of a dense O(d^3) matrix exponential.
     */
    Matrix gradient_samples;
    Vector gradient_weights;
    Float gradient_shift = 0.0;
    //! G as a dense matrix, formed only when the update uses the dense matrix exponential
    Matrix gradient_matrix;

    //! Whether the update uses the dense matrix exponential (lambda >= d, or d < 16)
    [[nodiscard]] bool dense_update(Eigen::Index n_samples) const;

    //! G = Z diag(u) Z^T - alpha I as a dense matrix (computed; the updates do not form it)
    [[nodiscard]] Matrix G() const
    {
        const auto d = A.rows();
        Matrix g = -gradient_shift * Matrix::Identity(d, d);
        if (gradient_samples.cols() > 0)
            g.noalias() += gradient_samples * gradient_weights.asDiagonal() *
                           gradient_samples.transpose();
        return g;
    }

    NaturalGradientAdaptation(const size_t dim, const Vector &x0) :
        Adaptation(dim, x0),
        // x = m + sigma A z with A = I at the start: the step size sigma sets the spread (xNES:
        // A = sigma B with det B = 1, Glasmachers et al. 2010)
        A(Matrix::Identity(dim, dim)),
        A_inv(Matrix::Identity(dim, dim)),
        sigma_g(0),
        gradient_samples(dim, 0),
        gradient_matrix(Matrix::Zero(dim, dim)),
        outdated_A_inv(false)
    {
    }

    //! Computed on request from C = A A^T
    Metric metric() override;

    void compute_gradients(const Population &pop,
                           const parameters::Weights &w,
                           const parameters::Settings &settings);

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings) override;

    void start_from(const Matrix &C0, const parameters::Settings &settings) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;

  private:
    bool outdated_A_inv;
};

inline std::shared_ptr<Adaptation>
get(const parameters::Modules &m, const size_t dim, const Vector &x0)
{
    using namespace parameters;
    switch (m.matrix_adaptation)
    {
    case MatrixAdaptationType::MATRIX:
        return std::make_shared<MatrixAdaptation>(dim, x0);
    case MatrixAdaptationType::NONE:
        return std::make_shared<None>(dim, x0);
    case MatrixAdaptationType::SEPARABLE:
        return std::make_shared<SeparableAdaptation>(dim, x0);
    case MatrixAdaptationType::CHOLESKY:
        return std::make_shared<CholeskyAdaptation>(dim, x0);
    case MatrixAdaptationType::CMSA:
        return std::make_shared<SelfAdaptation>(dim, x0);
    case MatrixAdaptationType::COVARIANCE_NO_EIGV:
        return std::make_shared<CovarianceNoEigvAdaptation>(dim, x0);
    case MatrixAdaptationType::NATURAL_GRADIENT:
        return std::make_shared<NaturalGradientAdaptation>(dim, x0);
    default:
    case MatrixAdaptationType::COVARIANCE:
        return std::make_shared<CovarianceAdaptation>(dim, x0);
    }
}
} // namespace matrix_adaptation
