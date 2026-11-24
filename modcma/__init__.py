"""The Modular CMA-ES: one CMA-ES with interchangeable modules, implemented in C++.

The API is that of :mod:`modcma.c_maes`, which stays available under that name::

    import modcma

    xopt, fopt, evaluations, es = modcma.fmin(func, x0, sigma0=0.5, budget=10_000)

The pure-Python implementation of modcma 1.x (``AskTellCMAES``, ``evaluate_bbob``, ...) was
removed in 2.0; it remains in modcma 1.x (``pip install "modcma<2"``).
"""

from importlib.metadata import PackageNotFoundError, version as _version

from . import c_maes
from .c_maes import (
    Modules,
    ModularCMAES,
    Parameters,
    Population,
    Settings,
    bounds,
    center,
    constants,
    es,
    fmin,
    get_all_module_options,
    get_configspace,
    mutation,
    options,
    parameters,
    repelling,
    restart,
    sampling,
    selection,
    settings_from_config,
    settings_from_dict,
    set_seed,
    build_info,
    utils,  # deprecated
)

try:
    __version__ = _version("modcma")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "unknown"

# Names of the pure-Python implementation (modcma 1.x) that 2.0 no longer has
_REMOVED = frozenset(
    {
        "AskTellCMAES",
        "BIPOPParameters",
        "evaluate_bbob",
        "timeit",
        "ert",
        "gaussian_sampling",
        "sobol_sampling",
        "halton_sampling",
        "mirrored_sampling",
        "orthogonal_sampling",
        "Halton",
        "Sobol",
    }
)


def __getattr__(name: str):
    if name in _REMOVED:
        raise AttributeError(
            f"modcma.{name} belonged to the pure-Python implementation, which modcma 2.0 no "
            "longer contains; see https://modularcmaes.readthedocs.io/en/latest/migration/. It "
            'remains in modcma 1.x: pip install "modcma<2".'
        )
    raise AttributeError(f"module 'modcma' has no attribute {name!r}")


__all__ = (
    "c_maes",
    "fmin",
    "ModularCMAES",
    "Settings",
    "Modules",
    "Parameters",
    "Population",
    "options",
    "constants",
    "set_seed",
    "build_info",
    "parameters",
    "sampling",
    "mutation",
    "selection",
    "bounds",
    "restart",
    "repelling",
    "center",
    "es",
    "get_configspace",
    "get_all_module_options",
    "settings_from_config",
    "settings_from_dict",
)
