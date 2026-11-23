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
}
#endif

namespace lapack
{
namespace
{
#ifdef MODCMA_LAPACK
std::atomic<syevd_fn> syevd{&dsyevd_};
std::atomic<syevr_fn> syevr{&dsyevr_};
std::atomic<potrf_fn> potrf{&dpotrf_};
constexpr bool linked = true;
#else
std::atomic<syevd_fn> syevd{nullptr};
std::atomic<syevr_fn> syevr{nullptr};
std::atomic<potrf_fn> potrf{nullptr};
constexpr bool linked = false;
#endif
// Written once, at import of the Python package, before any optimizer runs
std::string routines_source = linked ? "linked" : "none";
} // namespace

bool available()
{
    return syevd.load() != nullptr && syevr.load() != nullptr && potrf.load() != nullptr;
}

std::string source() { return routines_source; }

void set_routines(const syevd_fn d, const syevr_fn r, const potrf_fn p, const std::string &from)
{
    if (linked || d == nullptr || r == nullptr || p == nullptr)
        return; // a build with MODCMA_LAPACK uses its own LAPACK
    syevd = d;
    syevr = r;
    potrf = p;
    routines_source = from;
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
        syevd.load()(&jobz, &uplo, &n, ws.a.data(), &n, w.data(), &work_size, &query, &iwork_size,
                     &query, &info);
    }
    else
    {
        double zero = 0.0;
        int one = 1, m = 0;
        Matrix z(n, n);
        ws.isuppz.resize(2 * static_cast<size_t>(n));
        syevr.load()(&jobz, &range, &uplo, &n, ws.a.data(), &n, &zero, &zero, &one, &one, &zero,
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
        syevd.load()(&jobz, &uplo, &n, eigenvectors.data(), &n, eigenvalues.data(),
                     ws.work.data(), &lwork, ws.iwork.data(), &liwork, &info);
    }
    else
    {
        double zero = 0.0;
        int one = 1, m = 0;
        ws.a = C;
        eigenvectors.resize(n, n);
        syevr.load()(&jobz, &range, &uplo, &n, ws.a.data(), &n, &zero, &zero, &one, &one, &zero,
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
    potrf.load()(&uplo, &n, L.data(), &n, &info);
    if (info != 0)
        return false;
    L.triangularView<Eigen::StrictlyUpper>().setZero();
    return true;
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
