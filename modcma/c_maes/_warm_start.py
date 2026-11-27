"""Warm starting: the initial search distribution from the solutions of a related task (#89)."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def warm_start(
    solutions: Sequence[tuple[Any, float]], gamma: float = 0.1, alpha: float = 0.1
) -> dict[str, Any]:
    """The initial distribution of a run, estimated from the solutions of a related task.

    WS-CMA-ES (Nomura, Watanabe, Akimoto, Ozaki and Onishi, "Warm starting CMA-ES for
    hyperparameter optimization", AAAI 2021, Section 3.3): the best ``floor(gamma * N)`` of the
    N solutions define a promising distribution, a mixture of Gaussians N(x_i, alpha^2 I) (Eq. 3).
    The Gaussian closest to it in KL divergence has the mean and covariance

    - m = (1 / n) sum_i x_i (Eq. 4),
    - Sigma = alpha^2 I + (1 / n) sum_i (x_i - m)(x_i - m)^T (Eq. 5),

    over the n selected solutions. Sigma = sigma0^2 C0 is split as in ``cmaes``
    (``get_warm_start_mgd`` in ``cmaes/_warm_start.py``, by the paper's authors):
    sigma0 = det(Sigma)^(1 / (2 d)) and C0 = Sigma / sigma0^2, so that det(C0) = 1; the paper
    leaves the split open. The result agrees with ``cmaes`` (0.13.1) to rounding (tested to
    1e-12), with two differences in how it is computed, not in what:

    - Sigma is summed over the centered solutions, as Eq. 5 writes it; ``cmaes`` computes the
      equal (1 / n) sum_i x_i x_i^T - m m^T, which loses the digits of a small spread far from
      the origin (solutions at 1e6 with a spread of 1e-3: C off by 4e-2).
    - The determinant comes from the Cholesky factor of Sigma, as a logarithm; the
      ``numpy.linalg.det`` of ``cmaes`` underflows in high dimension (for Sigma close to
      alpha^2 I with alpha = 0.1 it is subnormal from d = 154 on and 0 from d = 162 on, and
      ``cmaes`` then divides by zero).

    Parameters
    ----------
    solutions : sequence of (array_like, float)
        Pairs (x, f(x)) from a related task (the source task), e.g. every point that a run on it
        evaluated: x a (d,) point, f(x) its objective value (smaller is better). Solutions with
        equal values keep their order; a NaN value counts as the worst.
    gamma : float, default 0.1
        The fraction of the solutions that is selected, in (0, 1]: the best floor(gamma * N).
        At least one solution must be selected.
    alpha : float, default 0.1
        The prior: the standard deviation of each Gaussian of the promising distribution, in the
        units of x; at least 0. It keeps C0 positive definite when fewer than d + 1 solutions are
        selected. The paper uses 0.1 for variables in [0, 1] (Appendix A): scale it with the
        range of the variables.

    Returns
    -------
    dict
        The arguments of `Settings` for the new task: ``x0`` ((d,) array, the mean m),
        ``sigma0`` (float) and ``C0`` ((d, d) array, symmetric positive definite, det 1), so
        that ``Settings(d, **warm_start(solutions), budget=...)`` starts from N(m, Sigma).

    Raises
    ------
    ValueError
        No solutions, points of different lengths or not finite, gamma outside (0, 1] or
        selecting no solution, a negative alpha, or a Sigma that is not positive definite
        (alpha = 0 and the selected solutions do not span the space).

    Examples
    --------
    >>> import numpy as np
    >>> import modcma
    >>> rng = np.random.default_rng(1)
    >>> X = rng.uniform(0, 1, size=(100, 3))
    >>> solutions = [(x, float(np.sum((x - 0.6) ** 2))) for x in X]  # the source task
    >>> settings = modcma.Settings(3, **modcma.warm_start(solutions), budget=1000)

    See also
    --------
    Settings.C0 : how each matrix adaptation starts from C0.
    """
    if not 0 < gamma <= 1:
        raise ValueError(f"warm_start: gamma must be in (0, 1], got {gamma}")
    if not alpha >= 0:
        raise ValueError(f"warm_start: alpha must be at least 0, got {alpha}")
    if len(solutions) == 0:
        raise ValueError("warm_start: solutions must contain at least one (x, f) pair")
    points = [np.asarray(x, dtype=float).reshape(-1) for x, _ in solutions]
    dim = len(points[0])
    if dim == 0 or any(len(x) != dim for x in points):
        raise ValueError("warm_start: every x must have the same length d >= 1")
    X = np.array(points)
    if not np.all(np.isfinite(X)):
        raise ValueError("warm_start: every x must be finite")
    f = np.array([float(value) for _, value in solutions])

    # The best floor(gamma N) (cmaes: math.floor(len(solutions) * gamma)); a stable sort, as
    # Python's sorted in cmaes, keeps the order of equal values
    n = math.floor(len(solutions) * gamma)
    if n < 1:
        raise ValueError(
            f"warm_start: gamma * len(solutions) = {gamma * len(solutions)} selects no solution"
        )
    top = X[np.argsort(f, kind="stable")[:n]]

    # Nomura et al. (2021), Eqs. 4 and 5
    mean = top.mean(axis=0)
    centered = top - mean
    cov = alpha**2 * np.eye(dim) + (centered.T @ centered) / n
    cov = (cov + cov.T) / 2

    # Sigma = sigma0^2 C0 with det(C0) = 1 (cmaes): log det(Sigma) = 2 sum_i log L_ii
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError(
            "warm_start: the covariance of the selected solutions is not positive definite; "
            "use alpha > 0, or select at least d + 1 solutions that span the space"
        ) from None
    log_det = 2.0 * float(np.sum(np.log(np.diag(factor))))
    sigma0 = math.exp(log_det / (2 * dim))
    return {"x0": mean, "sigma0": sigma0, "C0": cov / sigma0**2}
