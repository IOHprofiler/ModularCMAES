#pragma once

#include "lapack.hpp"

#include "mutation.hpp"
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

struct Adaptation
{
    Vector m, m_old, dm, ps, dz;
    Vector coordinate_wise_variances;

    Float dd;
    Float expected_length_z;

    // Buffers reused by the updates, so that a generation does not allocate
    Vector scratch_vector;
    Vector weights_scratch;
    Matrix square_scratch;
    Matrix batch_scratch;  // d x n
    Matrix batch_scratch2; // n x d

    Adaptation(const size_t dim,
               const Vector &x0,
               const Vector &ps,
               const Float expected_length_z) :
        m(x0),
        m_old(dim),
        dm(Vector::Zero(dim)),
        ps(ps),
        coordinate_wise_variances(Vector::Ones(dim)),
        dd(static_cast<Float>(dim)),
        expected_length_z(expected_length_z)
    {
    }

    virtual void adapt_ps(const parameters::Weights &w);

    void adapt_evolution_paths(const Population &pop,
                               const parameters::Weights &w,
                               const parameters::Stats &stats,
                               const parameters::Settings &settings,
                               size_t mu,
                               size_t lambda);

    virtual void adapt_evolution_paths_inner(const Population &pop,
                                             const parameters::Weights &w,
                                             const parameters::Stats &stats,
                                             const parameters::Settings &settings,
                                             size_t mu,
                                             size_t lambda) = 0;

    bool adapt_matrix(const parameters::Weights &w,
                      const parameters::Modules &m,
                      const Population &pop,
                      size_t mu,
                      const parameters::Settings &settings,
                      parameters::Stats &stats)
    {
        if (settings.one_plus_one and !stats.has_improved)
            return true;
        const bool success = adapt_matrix_inner(w, m, pop, mu, settings, stats);
        // Every matrix adaptation derives coordinate_wise_variances from all entries of its
        // matrix (row sums of squares, or the diagonal of C, whose decomposition is checked
        // separately), so a NaN or inf anywhere in the matrix shows up here. Without this check
        // a run continued with NaN samples until its budget was used up.
        return success && coordinate_wise_variances.allFinite();
    }

    virtual bool adapt_matrix_inner(const parameters::Weights &w,
                                    const parameters::Modules &m,
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

    virtual void restart(const parameters::Settings &settings, const Float sigma)
    {
        m = settings.x0.value_or(settings.center);
        m_old.setZero();
        dm.setZero();
        ps.setZero();
        dz.setZero();
        coordinate_wise_variances.setOnes();
    }

    Float distance(const Vector u, const Vector &v)
    {
        const auto &delta = u - v;
        return invert_y(delta).norm();
    }

    Float distance_from_center(const Vector &xi) { return distance(m, xi); }
};

struct None final : Adaptation
{
    None(const size_t dim, const Vector &x0, const Float expected_length_z) :
        Adaptation(dim, x0, Vector::Zero(dim), expected_length_z)
    {
    }

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const parameters::Modules &m,
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
                                     const parameters::Settings &settings,
                                     size_t mu,
                                     size_t lambda) override;

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

    //! Whether B and d hold the eigendecomposition of C (false for COVARIANCE_NO_EIGV, which
    //! factorizes C with a Cholesky decomposition and leaves B and d at their initial values)
    [[nodiscard]] virtual bool has_eigendecomposition() const { return true; }

    //! C^-1/2 = B D^-1 B^T. Not kept up to date (an O(d^3) product per eigendecomposition): the
    //! updates use B and d directly; this computes it.
    [[nodiscard]] Matrix inv_root_C() const { return B * d.cwiseInverse().asDiagonal() * B.transpose(); }

    // Reused by the matrix update: the eigensolver with its workspace
    Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver;
    lapack::Workspace lapack_workspace;

    CovarianceAdaptation(const size_t dim, const Vector &x0, const Float expected_length_z) :
        Adaptation(dim, x0, Vector::Zero(dim), expected_length_z),
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
                                     const parameters::Settings &settings,
                                     size_t mu,
                                     size_t lambda) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const parameters::Modules &m,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings, const Float sigma) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;
};

struct SeparableAdaptation : Adaptation
{
    Vector pc, d, c;
    bool hs;

    SeparableAdaptation(const size_t dim, const Vector &x0, const Float expected_length_z) :
        Adaptation(dim, x0, Vector::Zero(dim), expected_length_z),
        pc(Vector::Zero(dim)),
        d(Vector::Ones(dim)),
        c(Vector::Ones(dim)),
        hs(true)
    {
    }

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings,
                                     size_t mu,
                                     size_t lambda) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const parameters::Modules &m,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings, const Float sigma) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;
};

struct MatrixAdaptation final : Adaptation
{
    Matrix M;
    Matrix M_inv;

    MatrixAdaptation(const size_t dim, const Vector &x0, const Float expected_length_z) :
        Adaptation(dim, x0, Vector::Zero(dim), expected_length_z),
        M(Matrix::Identity(dim, dim)),
        M_inv(Matrix::Identity(dim, dim)),
        outdated_M_inv(false)
    {
    }

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings,
                                     size_t mu,
                                     size_t lambda) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const parameters::Modules &m,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings, const Float sigma) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;

  private:
    bool outdated_M_inv;
};

struct CholeskyAdaptation final : Adaptation
{
    Matrix A;
    Vector pc;
    /*
    First, as only triangular matrices have to be stored, the storage complexity is optimal.
    Second, the diagonal elements of a triangular Cholesky factor are the square roots of the
    eigenvalues of the factorized matrix, that is, we get the eigenvalues of the covariance matrix
    for free*/

    CholeskyAdaptation(const size_t dim, const Vector &x0, const Float expected_length_z) :
        Adaptation(dim, x0, Vector::Zero(dim), expected_length_z),
        A(Matrix::Identity(dim, dim)),
        pc(Vector::Zero(dim))
    {
    }

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings,
                                     size_t mu,
                                     size_t lambda) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const parameters::Modules &m,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings, const Float sigma) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;
};

struct SelfAdaptation final : Adaptation
{
    Matrix A;
    Matrix C;

    SelfAdaptation(const size_t dim, const Vector &x0, const Float expected_length_z) :
        Adaptation(dim, x0, Vector::Zero(dim), expected_length_z),
        A(Matrix::Identity(dim, dim)),
        C(Matrix::Identity(dim, dim))
    {
    }

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings,
                                     size_t mu,
                                     size_t lambda) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const parameters::Modules &m,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings, const Float sigma) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;
};

struct CovarianceNoEigvAdaptation : CovarianceAdaptation
{
    using CovarianceAdaptation::CovarianceAdaptation;

    void adapt_ps(const parameters::Weights &w) override;

    bool perform_eigendecomposition(const parameters::Settings &settings) override;

    bool has_eigendecomposition() const override { return false; }

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

    NaturalGradientAdaptation(const size_t dim,
                              const Vector &x0,
                              const Float expected_length_z,
                              [[maybe_unused]] const Float sigma0) :
        Adaptation(dim, x0, Vector::Zero(dim), expected_length_z),
        // x = m + sigma A z with A = I at the start: sigma0 sets the spread (xNES: A = sigma B
        // with det B = 1, Glasmachers et al. 2010)
        A(Matrix::Identity(dim, dim)),
        A_inv(Matrix::Identity(dim, dim)),
        sigma_g(0),
        gradient_samples(dim, 0),
        gradient_matrix(Matrix::Zero(dim, dim)),
        outdated_A_inv(false)
    {
    }

    void compute_gradients(const Population &pop,
                           const parameters::Weights &w,
                           const parameters::Stats &stats,
                           const parameters::Settings &settings,
                           size_t mu,
                           size_t lambda);

    void adapt_evolution_paths_inner(const Population &pop,
                                     const parameters::Weights &w,
                                     const parameters::Stats &stats,
                                     const parameters::Settings &settings,
                                     size_t mu,
                                     size_t lambda) override;

    bool adapt_matrix_inner(const parameters::Weights &w,
                            const parameters::Modules &m,
                            const Population &pop,
                            size_t mu,
                            const parameters::Settings &settings,
                            parameters::Stats &stats) override;

    void restart(const parameters::Settings &settings, const Float sigma) override;

    void compute_y_into(Eigen::Ref<const Vector> z, Eigen::Ref<Vector> y) override;

    void compute_y_batch(Eigen::Ref<const Matrix> Z, Eigen::Ref<Matrix> Y) override;

    void invert_y_into(Eigen::Ref<const Vector> y, Eigen::Ref<Vector> z) override;

  private:
    bool outdated_A_inv;
};

inline std::shared_ptr<Adaptation> get(const parameters::Modules &m,
                                       const size_t dim,
                                       const Vector &x0,
                                       const Float expected_z,
                                       const Float sigma0)
{
    using namespace parameters;
    switch (m.matrix_adaptation)
    {
    case MatrixAdaptationType::MATRIX:
        return std::make_shared<MatrixAdaptation>(dim, x0, expected_z);
    case MatrixAdaptationType::NONE:
        return std::make_shared<None>(dim, x0, expected_z);
    case MatrixAdaptationType::SEPARABLE:
        return std::make_shared<SeparableAdaptation>(dim, x0, expected_z);
    case MatrixAdaptationType::CHOLESKY:
        return std::make_shared<CholeskyAdaptation>(dim, x0, expected_z);
    case MatrixAdaptationType::CMSA:
        return std::make_shared<SelfAdaptation>(dim, x0, expected_z);
    case MatrixAdaptationType::COVARIANCE_NO_EIGV:
        return std::make_shared<CovarianceNoEigvAdaptation>(dim, x0, expected_z);
    case MatrixAdaptationType::NATURAL_GRADIENT:
        return std::make_shared<NaturalGradientAdaptation>(dim, x0, expected_z, sigma0);
    default:
    case MatrixAdaptationType::COVARIANCE:
        return std::make_shared<CovarianceAdaptation>(dim, x0, expected_z);
    }
}
} // namespace matrix_adaptation
