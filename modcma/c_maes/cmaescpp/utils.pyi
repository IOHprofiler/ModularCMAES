"""
Helpers: the global random generator, build information, the normal CDF and its inverse, the Sobol sequence, the LAPACK/BLAS routines the library uses, and the permutation generators of the samplers.
"""

from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


def build_info() -> dict[str, str]:
    """
    How the extension was built.

    Returns
    -------
    dict[str, str]
        'compiler': the compiler and its version, e.g. 'gcc 13.3.0';
        'fma': 'yes' when the code may use fused multiply-add instructions (which change
        the last bits of results), 'no' otherwise;
        'lapack': where LAPACK and BLAS come from: 'linked' (built with them, Accelerate
        on macOS), 'scipy' (scipy's, handed over at import) or 'none' (Eigen only).
    """

def cdf(x: float) -> float:
    """
    The cumulative distribution function of the standard normal distribution at x.
    """

def ppf(x: float) -> float:
    """
    The inverse of `cdf` (percent point function): the standard normal quantile of the probability x in (0, 1).

    The GAUSSIAN sample transformation maps uniform numbers to normal ones with it.
    """

def i8_sobol(dim_num: int, seed: int, quasi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> int:
    """
    Write the next point of the Sobol sequence into quasi and return the next seed.

    Parameters
    ----------
    dim_num : int
        The dimension of the point.
    seed : int
        The position in the sequence (the `sampling.Sobol` sampler starts at 2).
    quasi : numpy.ndarray
        A contiguous float64 array of at least dim_num numbers, overwritten with the point.

    Returns
    -------
    int
        The seed of the point after this one.
    """

def compute_ert(running_times: Sequence[int], budget: int) -> tuple[float, int]:
    """
    The expected running time (ERT) of a set of runs.

    Parameters
    ----------
    running_times : list[int]
        The evaluations each run used; a run succeeded when it used fewer than budget.
    budget : int
        The budget of each run.

    Returns
    -------
    tuple[float, int]
        The sum of all running times divided by the number of successful runs (inf when no run succeeded), and the number of successful runs.
    """

def set_seed(seed: int) -> None:
    """
    Seed the global random generator (mt19937).

    An optimizer without `Settings.seed` copies the global generator when it is created and advances it, so that after set_seed(s) the optimizers created in the same order draw the same numbers. Samplers, bound corrections and functions of this module that are used on their own draw from the global generator directly.
    """

def lapack_eigh(C: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], method: int = 1) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
    """
    Eigenvalues and eigenvectors of the symmetric matrix C with LAPACK, as the covariance adaptation computes them.

    Parameters
    ----------
    C : numpy.ndarray
        A symmetric matrix (its lower triangle is read).
    method : int
        1: dsyevd (divide and conquer, default), 2: dsyevr (MRRR).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The eigenvalues in ascending order and the orthonormal eigenvectors (columns).

    Raises
    ------
    RuntimeError
        When LAPACK fails or is not available.
    """

def lapack_cholesky(C: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
    """
    The lower triangular Cholesky factor L of C (C = L L^T) with LAPACK dpotrf, as CMSA and COVARIANCE_NO_EIGV compute it.

    Raises RuntimeError when C is not positive definite or LAPACK is not available.
    """

def lapack_inverse(M: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
    """
    The inverse of the square matrix M with an LU factorization (LAPACK dgetrf, dgetri), as MA-ES and xNES compute it.

    Raises RuntimeError when M is singular or LAPACK is not available.
    """

def blas_gemm(A: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], B: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], transpose_a: bool = False, transpose_b: bool = False) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
    """
    The product op(A) op(B) with BLAS dgemm, as the matrix adaptations compute it (for tests).

    op(X) is X, or X^T when transpose_a (transpose_b) is True. Raises RuntimeError when BLAS is not available.
    """

def random_uniform() -> float:
    """
    A uniform random number in the open interval (0, 1) from the global generator.
    """

def random_normal() -> float:
    """
    A standard normal random number from the global generator (Box-Muller).
    """

class Shuffler:
    """
    A random permutation of the integers in [start, stop), generated one at a time.

    A linear congruential generator, x <- (multiplier x + offset) mod modulus, with a random seed and offset; values >= n are skipped. The first n values of `next` are a permutation of the range, after which it repeats. Used to draw cached samples (Sobol) in a random order.
    """

    @overload
    def __init__(self, start: int, stop: int) -> None:
        """
        A permutation of [start, stop); seed and offset are drawn from the global generator.
        """

    @overload
    def __init__(self, stop: int) -> None:
        """A permutation of [0, stop)."""

    def next(self) -> int:
        """The next integer of the permutation."""

    @property
    def start(self) -> int:
        """The first integer of the range."""

    @start.setter
    def start(self, arg: int, /) -> None: ...

    @property
    def stop(self) -> int:
        """The end of the range (exclusive)."""

    @stop.setter
    def stop(self, arg: int, /) -> None: ...

    @property
    def n(self) -> int:
        """The length of the range, stop - start."""

    @n.setter
    def n(self, arg: int, /) -> None: ...

    @property
    def seed(self) -> int:
        """
        The state of the generator: start + seed is the value returned last (before the first call: the first value).
        """

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

    @property
    def offset(self) -> int:
        """The increment of the generator (odd)."""

    @offset.setter
    def offset(self, arg: int, /) -> None: ...

    @property
    def multiplier(self) -> int:
        """The multiplier of the generator, 4 floor(n / 4) + 1."""

    @multiplier.setter
    def multiplier(self, arg: int, /) -> None: ...

    @property
    def modulus(self) -> int:
        """The modulus of the generator, the smallest power of 2 >= n."""

    @modulus.setter
    def modulus(self, arg: int, /) -> None: ...

    @property
    def found(self) -> int:
        """The number of integers returned so far."""

    @found.setter
    def found(self, arg: int, /) -> None: ...

class CachedShuffleSequence:
    """
    A cache of points, returned in a random order (a `Shuffler` over the points).

    Holds n points of dimension dim, with n = max(constants.cache_min_samples, the largest power of 2 <= constants.cache_max_doubles / dim). The Sobol sampler keeps its points in one.
    """

    def __init__(self, dim: int) -> None:
        """A cache of points of dimension dim, all zero until `fill`."""

    def fill(self, c: Sequence[float]) -> None:
        """
        Copy the numbers c into the cache, point after point (dim numbers each).

        c must not hold more than n * dim numbers.
        """

    def get_index(self, index: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """A copy of point index of the cache (index < n; not checked)."""

    def next(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """The next point in the random order."""
