import warnings
from enum import Enum
from typing import Union

import numpy as np
from ConfigSpace import (
    ConfigurationSpace,
    Configuration,
    Categorical,
    ForbiddenGreaterThanRelation,
    UniformIntegerHyperparameter,
    NormalFloatHyperparameter,
)

from .cmaescpp import (
    constants,
    utils,  # pyright: ignore[reportMissingModuleSource]
    sampling,  # pyright: ignore[reportMissingModuleSource]
    mutation,  # pyright: ignore[reportMissingModuleSource]
    selection,  # pyright: ignore[reportMissingModuleSource]
    parameters,  # pyright: ignore[reportMissingModuleSource]
    bounds,  # pyright: ignore[reportMissingModuleSource]
    restart,  # pyright: ignore[reportMissingModuleSource]
    options,  # pyright: ignore[reportMissingModuleSource]
    repelling,  # pyright: ignore[reportMissingModuleSource]
    Population,
    Parameters,
    ModularCMAES,
    center,  # pyright: ignore[reportMissingModuleSource]
    es,  # pyright: ignore[reportMissingModuleSource]
)

from .cmaescpp.parameters import (  # pyright: ignore[reportMissingModuleSource]
    Settings,
    Modules,
)  # pyright: ignore[reportMissingModuleSource]


def _use_scipy_lapack() -> None:
    """Give the C++ library scipy's LAPACK for the eigendecomposition in high dimension.

    The macOS wheels link Apple Accelerate themselves; the Linux and Windows wheels have no LAPACK
    of their own and use scipy's (a dependency of modcma, already imported by modcma). Without it
    the library falls back to Eigen's eigensolver. See docs/performance.md.
    """
    if utils.build_info()["lapack"] != "none":
        return
    try:
        from scipy.linalg import cython_blas, cython_lapack

        lapack, blas = cython_lapack.__pyx_capi__, cython_blas.__pyx_capi__
        capsules = {name: lapack[name] for name in ("dsyevd", "dsyevr", "dpotrf", "dgetrf", "dgetri")}
        capsules["dgemm"] = blas["dgemm"]
        utils._set_lapack(capsules, "scipy")
    except Exception as error:  # pragma: no cover - depends on the scipy build
        warnings.warn(f"modcma: scipy's LAPACK is not usable ({error}); using Eigen", stacklevel=2)


_use_scipy_lapack()


def _get_module_options(name: str) -> tuple:
    if not hasattr(Modules, name):
        raise NameError(f"Modules has no member {name}")

    default_value = getattr(Modules(), name)
    if isinstance(default_value, bool):
        return (default_value, not default_value)

    module_class = default_value.__class__
    if issubclass(module_class, Enum):
        other_values = [
            x.name for x in module_class.__members__.values() if x is not default_value
        ]
        return tuple([default_value.name] + other_values)
    raise TypeError(f"{name} has a unparsable type {type(default_value)}")


def get_all_module_options() -> dict:
    """The options of every module of `Modules`.

    Returns
    -------
    dict
        Maps the name of every `Modules` attribute to a tuple of its options, the default first.
        A bool module gives `(default, not default)`; an enum module gives the names (str) of
        its options in `modcma.c_maes.options`, e.g. `"COVARIANCE"`.
    """
    return {
        name: _get_module_options(name)
        for name in dir(Modules)
        if not name.startswith("_")
    }


def _make_numeric_parameter(
    name: str, dim: int, lb: float, ub: float
) -> Union[UniformIntegerHyperparameter, NormalFloatHyperparameter]:

    settings = Parameters(Settings(dim))
    default = getattr(settings.weights, name, None)
    if default is None:
        default = getattr(settings.settings, name)

    if isinstance(default, int):
        return UniformIntegerHyperparameter(name, lb, ub, default, log=True)

    elif isinstance(default, float):
        db = min(default - lb, ub - default)
        return NormalFloatHyperparameter(name, default, 0.3 * db, lb, ub)

    raise TypeError(
        f"default value for {name} ({default}) "
        f"has an unparsable type {type(default)}"
    )


def get_configspace(
    dim: int = None,
    add_popsize: bool = True,
    add_sigma: bool = True,
    add_learning_rates: bool = True,
) -> ConfigurationSpace:
    """A ConfigSpace configuration space of the modules and the numeric settings, for tuning.

    Every module of `Modules` is a categorical hyperparameter with its options from
    `get_all_module_options` (default: the default of `Modules`). The numeric hyperparameters
    have the defaults of `Settings(dim)` with the default modules:

    - `lambda0` and `mu0`: integers in [1, 50 * dim], log scale, with the condition
      mu0 <= lambda0;
    - `sigma0`: a normal distribution around its default, truncated to [1e-15, 1e15];
    - `cs`, `cc`, `cmu`, `c1` in [0, 1] and `damps` in [0, 10]: normal distributions around
      their defaults (the values of `Weights`).

    The normal distributions have standard deviation 0.3 * min(default - lower, upper - default).
    Turn the result into `Settings` with `settings_from_config`.

    Parameters
    ----------
    dim : int, optional
        The dimension, for the numeric defaults. Without it, and with any numeric
        hyperparameter, a warning is issued and dim = 2 is used.
    add_popsize : bool, default True
        Add `lambda0` and `mu0`.
    add_sigma : bool, default True
        Add `sigma0`.
    add_learning_rates : bool, default True
        Add `cs`, `cc`, `cmu`, `c1` and `damps`.

    Returns
    -------
    ConfigSpace.ConfigurationSpace
        The configuration space.
    """
    cspace = ConfigurationSpace()
    for name, options in get_all_module_options().items():
        cspace.add(Categorical(name, options, default=options[0]))

    if dim is None and (add_popsize or add_sigma or add_learning_rates):
        warnings.warn(
            "Filling configspace with default numeric values for dim=2, since no dim was "
            "provided and add_popsize, add_sigma or add_learning_rates is True"
        )
        dim = 2

    if add_popsize:
        cspace.add(_make_numeric_parameter("lambda0", dim, 1, 50 * dim))
        cspace.add(_make_numeric_parameter("mu0", dim, 1, 50 * dim))
        cspace.add(ForbiddenGreaterThanRelation(cspace["mu0"], cspace["lambda0"]))

    if add_sigma:
        cspace.add(_make_numeric_parameter("sigma0", dim, 1e-15, 1e15))

    if add_learning_rates:
        cspace.add(_make_numeric_parameter("cs", dim, 0, 1.0))
        cspace.add(_make_numeric_parameter("cc", dim, 0, 1.0))
        cspace.add(_make_numeric_parameter("cmu", dim, 0, 1.0))
        cspace.add(_make_numeric_parameter("c1", dim, 0, 1.0))
        cspace.add(_make_numeric_parameter("damps", dim, 0, 10.0))

    return cspace


def set_module(modules: Modules, name: str, value: Union[str, bool, Enum]) -> bool:
    """Set one module of a `Modules` object by name, if it is a module.

    Parameters
    ----------
    modules : Modules
        The configuration to change (in place).
    name : str
        The name of a `Modules` attribute, e.g. `"matrix_adaptation"`.
    value : str, enum member or bool
        For an enum module, the option, by name (e.g. `"MATRIX"`) or as a member of its enum
        (e.g. `options.MatrixAdaptationType.MATRIX`); for a bool module, a bool.

    Returns
    -------
    bool
        True when `name` is a module and was set, False otherwise (nothing changes).
    """
    if hasattr(modules, name):
        attr_class = type(getattr(modules, name))
        if issubclass(attr_class, Enum) and not isinstance(value, attr_class):
            value = getattr(attr_class, value)
        setattr(modules, name, value)
        return True
    return False


def settings_from_dict(dim: int, **config: dict) -> Settings:
    """`Settings` from keyword arguments that mix module options and settings.

    A keyword that names a `Modules` attribute sets that module (an enum option by its name,
    e.g. `matrix_adaptation="MATRIX"`); every other keyword goes to the `Settings` constructor
    (e.g. `budget`, `lb`, `lambda0`, `seed`).

    Parameters
    ----------
    dim : int
        The dimension.
    **config
        Module options and `Settings` keyword arguments.

    Returns
    -------
    Settings
        The settings, with the modules set.

    Examples
    --------
    >>> settings = settings_from_dict(10, active=True, matrix_adaptation="MATRIX", budget=5000)
    """
    modules = Modules()
    via_settings = {}
    for name, value in dict(config).items():
        if set_module(modules, name, value):
            continue
        via_settings[name] = value
    settings = Settings(dim, modules, **via_settings)
    return settings


def settings_from_config(dim: int, config: Configuration, **kwargs) -> Settings:
    """`Settings` from a configuration of the space of `get_configspace`.

    The module hyperparameters set the modules. A numeric hyperparameter is passed to
    `Settings` only when it differs from its default in `get_configspace(dim)`; otherwise
    `Settings` computes the default itself, for the chosen modules.

    Parameters
    ----------
    dim : int
        The dimension.
    config : ConfigSpace.Configuration
        The configuration, e.g. from a tuner or `get_configspace(dim).sample_configuration()`.
    **kwargs
        Further `Settings` keyword arguments, e.g. `lb`, `ub` or `budget`.

    Returns
    -------
    Settings
        The settings of the configuration.
    """
    via_settings = kwargs
    default_config = get_configspace(dim).get_default_configuration()
    modules = Modules()
    for name, value in dict(config).items():
        if set_module(modules, name, value):
            continue
        if default_config[name] != value:
            via_settings[name] = value

    settings = Settings(dim, modules, **via_settings)
    return settings


def fmin(func: callable, x0: np.ndarray, sigma0: float, budget: int, **kwargs):
    """Minimize a function with the modular CMA-ES.

    Builds `Settings` with `settings_from_dict`, runs a `ModularCMAES` until a break condition
    holds and returns the best solution found.

    Parameters
    ----------
    func : callable
        The objective function, called with a (d,) numpy array; returns a float.
    x0 : array_like
        The starting point; its length is the dimension d.
    sigma0 : float
        The initial step size (rule of thumb: 0.3 * (ub - lb)).
    budget : int
        The maximum number of function evaluations.
    **kwargs
        Module options and `Settings` keyword arguments, as for `settings_from_dict`, e.g.
        `active=True`, `matrix_adaptation="MATRIX"`, `lb=..., ub=...`, `target=1e-8`.

    Returns
    -------
    xopt : numpy.ndarray
        (d,) the best point found.
    fopt : float
        The objective value of xopt.
    evals : int
        The number of evaluations used.
    es : ModularCMAES
        The optimizer, with its state in `es.p`.

    Examples
    --------
    >>> import numpy as np
    >>> from modcma.c_maes import fmin
    >>> xopt, fopt, evals, es = fmin(lambda x: float(np.sum(x**2)), np.ones(5), 0.5, 2000)
    """
    x0 = np.asarray(x0, dtype=float)
    settings = settings_from_dict(len(x0), x0=x0, sigma0=sigma0, budget=budget, **kwargs)
    es = ModularCMAES(settings)
    es(func)

    return (
        es.p.stats.global_best.x,
        es.p.stats.global_best.y,
        es.p.stats.evaluations,
        es,
    )


__all__ = (
    "fmin",
    "settings_from_dict",
    "settings_from_config",
    "get_configspace",
    "get_all_module_options",
    "Settings",
    "Modules",
    "constants",
    "utils",
    "sampling",
    "mutation",
    "selection",
    "parameters",
    "bounds",
    "restart",
    "options",
    "repelling",
    "Population",
    "Parameters",
    "ModularCMAES",
    "center",
    "es",
)
