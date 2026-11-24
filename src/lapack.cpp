#include "lapack.hpp"

#include <atomic>

#ifdef MODCMA_LAPACK
// The linked LAPACK (Accelerate on macOS, OpenBLAS or the reference LAPACK elsewhere), Fortran
// interface with 32-bit integers. The hidden lengths of the character arguments are left out, as
// all single-character arguments are.
extern "C"
{
    void dsyevd_(char *jobz,
                 char *uplo,
                 int *n,
                 double *a,
                 int *lda,
                 double *w,
                 double *work,
                 int *lwork,
                 int *iwork,
                 int *liwork,
                 int *info);

    void dsyevr_(char *jobz,
                 char *range,
                 char *uplo,
                 int *n,
                 double *a,
                 int *lda,
                 double *vl,
                 double *vu,
                 int *il,
                 int *iu,
                 double *abstol,
                 int *m,
                 double *w,
                 double *z,
                 int *ldz,
                 int *isuppz,
                 double *work,
                 int *lwork,
                 int *iwork,
                 int *liwork,
                 int *info);

    void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
    void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
    void dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
    void dgemm_(char *transa,
                char *transb,
                int *m,
                int *n,
                int *k,
                double *alpha,
                double *a,
                int *lda,
                double *b,
                int *ldb,
                double *beta,
                double *c,
                int *ldc);
}
#endif

namespace lapack
{
namespace
{
#ifdef MODCMA_LAPACK
const Routines linked_routines{&dsyevd_, &dsyevr_, &dpotrf_, &dgetrf_, &dgetri_, &dgemm_};
constexpr bool linked = true;
#else
const Routines linked_routines{};
constexpr bool linked = false;
#endif
// Set once, at import of the Python package, before any optimizer runs; published with `ready`
Routines routines = linked_routines;
std::atomic<bool> ready{linked};
std::string routines_source = linked ? "linked" : "none";

const Routines &get() { return routines; }
} // namespace

bool available() { return ready.load(std::memory_order_acquire); }

std::string source() { return routines_source; }

void set_routines(const Routines &r, const std::string &from)
{
    if (linked || r.syevd == nullptr || r.syevr == nullptr || r.potrf == nullptr ||
        r.getrf == nullptr || r.getri == nullptr || r.gemm == nullptr)
        return; // a build with MODCMA_LAPACK uses its own LAPACK; all routines or none
    routines = r;
    routines_source = from;
    ready.store(true, std::memory_order_release);
}

namespace
{
//! Size the workspace with a LAPACK workspace query, once per dimension and method
bool prepare(int n, const Method method, Workspace &ws)
{
    if (ws.n == n && ws.method == method)
        return true;

    double work_size = 0;
    int iwork_size = 0, info = 0, query = -1;
    char jobz = 'V', uplo = 'L', range = 'A';
    ws.a.resize(n, n);
    Vector w(n);
    if (method == Method::SYEVD)
    {
        get().syevd(&jobz, &uplo, &n, ws.a.data(), &n, w.data(), &work_size, &query, &iwork_size,
                     &query, &info);
    }
    else
    {
        double zero = 0.0;
        int one = 1, m = 0;
        Matrix z(n, n);
        ws.isuppz.resize(2 * static_cast<size_t>(n));
        get().syevr(&jobz, &range, &uplo, &n, ws.a.data(), &n, &zero, &zero, &one, &one, &zero,
                     &m, w.data(), z.data(), &n, ws.isuppz.data(), &work_size, &query,
                     &iwork_size, &query, &info);
    }
    if (info != 0)
        return false;
    ws.work.resize(static_cast<Eigen::Index>(work_size));
    ws.iwork.resize(static_cast<size_t>(iwork_size));
    ws.n = n;
    ws.method = method;
    return true;
}
} // namespace

bool symmetric_eigen(
    const Matrix &C, Vector &eigenvalues, Matrix &eigenvectors, const Method method, Workspace &ws)
{
    if (!available())
        return false;

    int n = static_cast<int>(C.rows());
    if (!prepare(n, method, ws))
        return false;

    char jobz = 'V', uplo = 'L', range = 'A';
    int lwork = static_cast<int>(ws.work.size());
    int liwork = static_cast<int>(ws.iwork.size());
    int info = 0;
    eigenvalues.resize(n);
    if (method == Method::SYEVD)
    {
        // In place: the eigenvectors overwrite the copy of C
        eigenvectors = C;
        get().syevd(&jobz, &uplo, &n, eigenvectors.data(), &n, eigenvalues.data(),
                     ws.work.data(), &lwork, ws.iwork.data(), &liwork, &info);
    }
    else
    {
        double zero = 0.0;
        int one = 1, m = 0;
        ws.a = C;
        eigenvectors.resize(n, n);
        get().syevr(&jobz, &range, &uplo, &n, ws.a.data(), &n, &zero, &zero, &one, &one, &zero,
                     &m, eigenvalues.data(), eigenvectors.data(), &n, ws.isuppz.data(),
                     ws.work.data(), &lwork, ws.iwork.data(), &liwork, &info);
        if (m != n)
            return false;
    }
    return info == 0;
}

bool cholesky(const Matrix &C, Matrix &L)
{
    if (!available())
        return false;
    int n = static_cast<int>(C.rows()), info = 0;
    char uplo = 'L';
    L = C;
    get().potrf(&uplo, &n, L.data(), &n, &info);
    if (info != 0)
        return false;
    L.triangularView<Eigen::StrictlyUpper>().setZero();
    return true;
}

bool inverse(const Matrix &M, Matrix &inverse)
{
    if (!available())
        return false;
    int n = static_cast<int>(M.rows()), info = 0;
    std::vector<int> pivots(static_cast<size_t>(std::max(n, 1)));
    inverse = M;
    get().getrf(&n, &n, inverse.data(), &n, pivots.data(), &info);
    if (info != 0)
        return false; // singular
    double work_size = 0;
    int query = -1;
    get().getri(&n, inverse.data(), &n, pivots.data(), &work_size, &query, &info);
    int lwork = std::max(1, static_cast<int>(work_size));
    Vector work(lwork);
    get().getri(&n, inverse.data(), &n, pivots.data(), work.data(), &lwork, &info);
    return info == 0 && inverse.allFinite();
}

bool use_blas(const Eigen::Index d)
{
    if (constants::blas == 0 || !available())
        return false;
    return constants::blas == 1 || static_cast<size_t>(d) >= constants::blas_min_dim;
}

void gemm(const double alpha,
          Eigen::Ref<const Matrix> A,
          const bool transpose_a,
          Eigen::Ref<const Matrix> B,
          const bool transpose_b,
          const double beta,
          Eigen::Ref<Matrix> C)
{
    char ta = transpose_a ? 'T' : 'N', tb = transpose_b ? 'T' : 'N';
    int m = static_cast<int>(C.rows()), n = static_cast<int>(C.cols());
    int k = static_cast<int>(transpose_a ? A.rows() : A.cols());
    if (m == 0 || n == 0)
        return;
    int lda = static_cast<int>(std::max<Eigen::Index>(1, A.outerStride()));
    int ldb = static_cast<int>(std::max<Eigen::Index>(1, B.outerStride()));
    int ldc = static_cast<int>(std::max<Eigen::Index>(1, C.outerStride()));
    double a = alpha, b = beta;
    get().gemm(&ta, &tb, &m, &n, &k, &a, const_cast<double *>(A.data()), &lda,
               const_cast<double *>(B.data()), &ldb, &b, C.data(), &ldc);
}

bool cholesky_factor(const Matrix &C, Matrix &L)
{
    const int solver =
        constants::cholesky_solver >= 0
            ? constants::cholesky_solver
            : (available() && static_cast<size_t>(C.rows()) >= constants::lapack_min_dim ? 1 : 0);
    if (solver == 1)
        return cholesky(C, L);

    const Eigen::LLT<Matrix> llt(C);
    if (llt.info() != Eigen::Success)
        return false;
    L = llt.matrixL();
    return true;
}
} // namespace lapack
