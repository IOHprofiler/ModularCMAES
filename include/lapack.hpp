#pragma once

#include <string>

#include "common.hpp"

/**
 * The eigendecomposition of a symmetric matrix with LAPACK, for the covariance adaptation.
 * LAPACK's blocked routines are several times faster than Eigen's SelfAdjointEigenSolver above
 * d ~ 100 (PERFORMANCE.md).
 *
 * Where LAPACK comes from:
 *  - "linked": a build with MODCMA_LAPACK (the default on macOS: Apple Accelerate);
 *  - "scipy": otherwise, the Python package gives the routines of scipy.linalg.cython_lapack at
 *    import (scipy is a dependency of modcma), so the Linux and Windows wheels need no LAPACK
 *    of their own;
 *  - "none": Eigen only (a C++ program built without MODCMA_LAPACK).
 */
namespace lapack
{
//! The Fortran interface of ?syevd and ?syevr (32-bit integers; as scipy's cython_lapack declares
//! them: no hidden string lengths)
using syevd_fn = void (*)(char *, char *, int *, double *, int *, double *, double *, int *, int *,
                          int *, int *);
using syevr_fn = void (*)(char *, char *, char *, int *, double *, int *, double *, double *,
                          int *, int *, double *, int *, double *, double *, int *, int *, double *,
                          int *, int *, int *, int *);
using potrf_fn = void (*)(char *, int *, double *, int *, int *);

//! Whether LAPACK routines are available
bool available();

//! "linked", "scipy" (or what was given to set_routines) or "none"
std::string source();

//! Use these routines (from Python: scipy's), unless the library was built with MODCMA_LAPACK
void set_routines(syevd_fn syevd, syevr_fn syevr, potrf_fn potrf, const std::string &source);

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
} // namespace lapack
