#pragma once

#include <string>

#include "common.hpp"

/**
 * LAPACK and BLAS for the dense linear algebra in high dimension: the eigendecomposition of the
 * covariance matrix, Cholesky factorizations, the inverse of M (MA-ES) and A (xNES), and matrix
 * products. They are several times faster than Eigen's kernels in high dimension (docs/performance.md),
 * so they are used from a threshold dimension on; below, Eigen is used and the results are
 * unchanged.
 *
 * Where the routines come from:
 *  - "linked": a build with MODCMA_LAPACK (the default on macOS: Apple Accelerate);
 *  - "scipy": otherwise, the Python package gives the routines of scipy.linalg.cython_lapack and
 *    cython_blas at import (scipy is a dependency of modcma), so the Linux and Windows wheels need
 *    no LAPACK of their own;
 *  - "none": Eigen only (a C++ program built without MODCMA_LAPACK).
 */
namespace lapack
{
// The Fortran interfaces (32-bit integers; as scipy's cython_lapack/cython_blas declare them: no
// hidden string lengths)
using syevd_fn = void (*)(char *, char *, int *, double *, int *, double *, double *, int *, int *,
                          int *, int *);
using syevr_fn = void (*)(char *, char *, char *, int *, double *, int *, double *, double *,
                          int *, int *, double *, int *, double *, double *, int *, int *, double *,
                          int *, int *, int *, int *);
using potrf_fn = void (*)(char *, int *, double *, int *, int *);
using getrf_fn = void (*)(int *, int *, double *, int *, int *, int *);
using getri_fn = void (*)(int *, double *, int *, int *, double *, int *, int *);
using gemm_fn = void (*)(char *, char *, int *, int *, int *, double *, double *, int *, double *,
                         int *, double *, double *, int *);

//! The routines the library uses (all of them, or none)
struct Routines
{
    syevd_fn syevd = nullptr;
    syevr_fn syevr = nullptr;
    potrf_fn potrf = nullptr;
    getrf_fn getrf = nullptr;
    getri_fn getri = nullptr;
    gemm_fn gemm = nullptr;
};

//! Whether LAPACK and BLAS routines are available
bool available();

//! "linked", "scipy" (or what was given to set_routines) or "none"
std::string source();

//! Use these routines (from Python: scipy's), unless the library was built with MODCMA_LAPACK
void set_routines(const Routines &routines, const std::string &source);

//! The LAPACK routine
enum class Method
{
    SYEVD, //!< divide and conquer (as numpy.linalg.eigh)
    SYEVR  //!< relatively robust representations (MRRR)
};

//! The workspace of a decomposition, kept between calls
struct Workspace
{
    Vector work;
    std::vector<int> iwork;
    std::vector<int> isuppz;
    Matrix a;
    int n = -1;
    Method method = Method::SYEVD;
};

/**
 * Eigenvalues (ascending) and orthonormal eigenvectors (columns) of the symmetric matrix C (its
 * lower triangle is read), as Eigen's SelfAdjointEigenSolver returns them. Returns false when
 * LAPACK reports an error or is not available.
 */
bool symmetric_eigen(
    const Matrix &C, Vector &eigenvalues, Matrix &eigenvectors, Method method, Workspace &ws);

/**
 * The lower triangular Cholesky factor L of the symmetric positive definite matrix C (its lower
 * triangle is read), C = L L^T, with dpotrf; the strictly upper triangle of L is zero, as
 * Eigen::LLT::matrixL(). Returns false when C is not positive definite or LAPACK is not
 * available.
 */
bool cholesky(const Matrix &C, Matrix &L);

/**
 * The Cholesky factor with LAPACK or Eigen, as constants::cholesky_solver chooses (automatic:
 * LAPACK from constants::lapack_min_dim on, when available). Returns false when C is not positive
 * definite.
 */
bool cholesky_factor(const Matrix &C, Matrix &L);

//! The inverse of the square matrix M with an LU factorization (dgetrf, dgetri). Returns false
//! when M is singular or LAPACK is not available.
bool inverse(const Matrix &M, Matrix &inverse);

//! Whether matrix products and the LU inverse use BLAS/LAPACK for dimension d
//! (constants::blas: -1 automatic, from constants::blas_min_dim on; 0 never; 1 always)
bool use_blas(Eigen::Index d);

/**
 * C = alpha op(A) op(B) + beta C with dgemm, for (blocks of) column-major matrices; op(X) is X or
 * X^T. The caller checks use_blas first.
 */
void gemm(double alpha,
          Eigen::Ref<const Matrix> A,
          bool transpose_a,
          Eigen::Ref<const Matrix> B,
          bool transpose_b,
          double beta,
          Eigen::Ref<Matrix> C);
} // namespace lapack
