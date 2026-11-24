"""Deprecated: use :func:`modcma.set_seed` and :func:`modcma.build_info`.

modcma 1.x had the helpers of the C++ library in ``modcma.c_maes.utils``. In 2.0 the two that are
meant for users are in the main namespace; this module forwards to them with a
``DeprecationWarning`` and will be removed in a later release. The other names of 1.x were
removed (see the migration guide).
"""

import warnings
from typing import Dict

from .cmaescpp import build_info as _build_info, set_seed as _set_seed

_REMOVED = {
    "cdf": "scipy.stats.norm.cdf",
    "ppf": "scipy.stats.norm.ppf",
    "random_uniform": "numpy.random.default_rng().uniform",
    "random_normal": "numpy.random.default_rng().normal",
    "compute_ert": "IOHanalyzer or iohinspector",
    "i8_sobol": "scipy.stats.qmc.Sobol",
    "Shuffler": None,
    "CachedShuffleSequence": None,
    "lapack_eigh": "numpy.linalg.eigh",
    "lapack_cholesky": "numpy.linalg.cholesky",
    "lapack_inverse": "numpy.linalg.inv",
    "blas_gemm": "numpy.matmul",
}


def _deprecated(name: str) -> None:
    warnings.warn(
        f"modcma.c_maes.utils.{name} is deprecated; use modcma.{name} (or modcma.c_maes.{name})",
        DeprecationWarning,
        stacklevel=3,
    )


def set_seed(seed: int) -> None:
    """Deprecated alias of :func:`modcma.set_seed`."""
    _deprecated("set_seed")
    _set_seed(seed)


def build_info() -> Dict[str, str]:
    """Deprecated alias of :func:`modcma.build_info`."""
    _deprecated("build_info")
    return _build_info()


def __getattr__(name: str):
    if name in _REMOVED:
        instead = _REMOVED[name]
        hint = f"; use {instead} instead" if instead else " (an internal of the samplers)"
        raise AttributeError(f"modcma.c_maes.utils.{name} was removed in modcma 2.0{hint}")
    raise AttributeError(f"module 'modcma.c_maes.utils' has no attribute {name!r}")
