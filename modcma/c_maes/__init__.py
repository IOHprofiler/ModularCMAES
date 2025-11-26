import warnings
from enum import Enum
from typing import Union

import numpy as np
from ConfigSpace import (
    ConfigurationSpace,
    Configuration,
    Categorical,
    CategoricalHyperparameter,
    ForbiddenGreaterThanRelation,
    UniformIntegerHyperparameter,
    NormalFloatHyperparameter,
    EqualsCondition,
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

from .cmaescpp.parameters import ( # pyright: ignore[reportMissingModuleSource]
    Settings,
    Modules,
)  # pyright: ignore[reportMissingModuleSource]


def _get_module_options(name: str) -> tuple:
    if not hasattr(Modules, name):
        raise NameError(f"Modules has no member {name}")

    default_value = getattr(Modules(), name)
    if isinstance(default_value, bool):
        return (default_value, not default_value)

    module_class = default_value.__class__
    if issubclass(module_class, Enum):
        other_values = [
            x.name for x in module_class.__members__.values() 
            if x is not default_value
        ]
        return tuple([default_value.name] + other_values)
    raise TypeError(f"{name} has a unparsable type {type(default_value)}")


def get_all_module_options() -> dict:
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
    add_learning_rates: bool = True
) -> ConfigurationSpace:
    cspace = ConfigurationSpace()
    for name, options in get_all_module_options().items():
        cspace.add(Categorical(name, options, default=options[0]))

    if dim is None and (add_popsize or add_sigma or add_learning_rates):
        warnings.warn(
            "Filling configspace with default numeric values for dim=2, "
            "since no dim was provided and only_modules was set to False"
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


def set_module(modules: Modules, name: str, value: Enum) -> bool:
    if hasattr(modules, name):
        attr_class = type(getattr(modules, name))
        if issubclass(attr_class, Enum):
            value = getattr(attr_class, value)
        setattr(modules, name, value)
        return True
    return False    


def settings_from_dict(dim: int, **config: dict) -> Settings:
    modules = Modules()
    via_settings = {}
    for name, value in dict(config).items():
        if set_module(modules, name, value):
            continue
        via_settings[name] = value
    settings = Settings(dim, modules, **via_settings)
    return settings


def settings_from_config(
    dim: int, 
    config: Configuration, 
    **kwargs
) -> Settings:
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
    """Minimize a function using the modular CMA-ES.

    Parameters
    ----------
    func: callable
        The objective function to be minimized.
    x0 np.ndarray: 
        The first solution estimate
    sigma0: float
        The estimate of the stepsize (rule of thumb: 0.3 * (ub - lb))
    budget: int
        Maximum number of function evaluations to make.
    **kwargs
        These are directly passed into the instance of ModularCMAES,
        in this manner parameters can be specified for the optimizer.
    Returns
    -------
    xopt
        The variables which minimize the function during this run
    fopt
        The value of function at found xopt
    evals
        The number of evaluations performed
    es
        The ModularCMAES instance
    """
    settings = settings_from_dict(len(x0), sigma0=sigma0, budget=budget, **kwargs)
    es = ModularCMAES(settings)
    es(func)
    
    return es.p.stats.global_best.x, es.p.stats.global_best.y, es.p.stats.evaluations, es
    
__all__ = (
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
