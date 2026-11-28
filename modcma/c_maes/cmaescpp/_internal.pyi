"""
Internals of the library, exposed for the tests: the LAPACK/BLAS routines, the Sobol sequence and the permutation generators of the samplers. Not part of the API: they may change without notice.
"""

from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


def i8_sobol(dim_num: int, seed: int, quasi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> int:
    """
    Write the next point of the Sobol sequence into quasi and return the next seed.

    The routine keeps its tables in static variables: calls from several threads at once are not safe.

    Parameters
    ----------
    dim_num : int
        The dimension of the point, 1 to 1111 (the size of the table of direction numbers; outside this range the routine ends the process).
    seed : int
        The position in the sequence (the `sampling.Sobol` sampler starts at 2).
    quasi : numpy.ndarray
        A contiguous float64 array of at least dim_num numbers, overwritten with the point.

    Returns
    -------
    int
        The seed of the point after this one.
    """

def set_lapack(capsules: dict, source: str) -> None:
    """
    Use these LAPACK/BLAS routines (internal).

    capsules maps the names dsyevd, dsyevr, dpotrf, dgetrf, dgetri and dgemm to the PyCapsules of scipy.linalg.cython_lapack and cython_blas; source is what build_info()['lapack'] reports afterwards. Ignored when the library has its own (build_info()['lapack'] == 'linked'). modcma.c_maes calls it at import.
    """

def lapack_eigh(C: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], method: int = 1) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]]:
    """
    Eigenvalues and eigenvectors of the symmetric matrix C with LAPACK, as the covariance adaptation computes them.

    Parameters
    ----------
    C : numpy.ndarray
        A symmetric matrix (its lower triangle is read).
    method : int
        1: dsyevd (divide and conquer, default), 2: dsyevr (MRRR); any other value: dsyevd.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The eigenvalues in ascending order and the orthonormal eigenvectors (columns).

    Raises
    ------
    ValueError
        When C is not square.
    RuntimeError
        When LAPACK fails or is not available.
    """

def lapack_cholesky(C: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
    """
    The lower triangular Cholesky factor L of C (C = L L^T) with LAPACK dpotrf, as CMSA and COVARIANCE_NO_EIGV compute it.

    Raises ValueError when C is not square, RuntimeError when C is not positive definite or LAPACK is not available.
    """

def lapack_inverse(M: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
    """
    The inverse of the square matrix M with an LU factorization (LAPACK dgetrf, dgetri), as MA-ES and xNES compute it.

    Raises ValueError when M is not square, RuntimeError when M is singular or LAPACK is not available.
    """

def blas_gemm(A: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], B: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], transpose_a: bool = False, transpose_b: bool = False) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
    """
    The product op(A) op(B) with BLAS dgemm, as the matrix adaptations compute it (for tests).

    op(X) is X, or X^T when transpose_a (transpose_b) is True. Raises ValueError when the inner dimensions of op(A) and op(B) differ, RuntimeError when BLAS is not available.
    """

class Shuffler:
    """
    A random permutation of the integers in [start, stop), generated one at a time.

    A linear congruential generator, x <- (multiplier x + offset) mod modulus, with a random seed, offset and multiplier; values >= n are skipped. The first n values of `next` are a permutation of the range, after which it repeats. Used to draw cached samples (Sobol) in a random order.
    """

    @overload
    def __init__(self, start: int, stop: int) -> None:
        """
        A permutation of [start, stop); seed, offset and multiplier are drawn from the global generator.
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
        """
        The multiplier of the generator, drawn at random with multiplier = 5 (mod 8) (1 when modulus <= 4): the full period, and not a constant stride.
        """

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
        """A copy of point index of the cache (index < n, otherwise IndexError)."""

    def next(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """The next point in the random order."""
