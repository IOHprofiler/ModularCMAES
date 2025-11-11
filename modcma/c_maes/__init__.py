import warnings
from enum import Enum

import numpy as np
from ConfigSpace import (
    ConfigurationSpace,
    Configuration,
    Categorical,
    CategoricalHyperparameter,
    ForbiddenGreaterThanRelation,
    UniformIntegerHyperparameter,
    NormalFloatHyperparameter,
    EqualsCondition
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

from .cmaescpp.parameters import (
    Settings,
    Modules,
)  # pyright: ignore[reportMissingModuleSource]




def get_module_options(name: str) -> tuple:
    if not hasattr(Modules, name):
        raise NameError(f"Modules has no member {name}")

    default_value = getattr(Modules(), name)
    if isinstance(default_value, bool):
        return (default_value, not default_value)

    module_class = default_value.__class__
    if issubclass(module_class, Enum):
        other_values = [
            x for x in module_class.__members__.values() if x is not default_value
        ]
        return tuple([default_value] + other_values)
    raise TypeError(f"{name} has a unparsable type {type(default_value)}")


def get_all_module_options() -> dict:
    return {
        name: get_module_options(name)
        for name in dir(Modules)
        if not name.startswith("_")
    }


def _make_numeric_parameter(
    name: str, dim: int, lb: float, ub: float
) -> UniformIntegerHyperparameter | NormalFloatHyperparameter:

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


def _get_numeric_config(
    dim: int,
) -> ConfigurationSpace:
    cs = ConfigurationSpace(
        {
            "lambda0": _make_numeric_parameter("lambda0", dim, 1, 50 * dim),
            "mu0": _make_numeric_parameter("mu0", dim, 1, 50 * dim),
            
            "sigma0": _make_numeric_parameter("sigma0", dim, 1e-15, 1e15), # TODO: should be based on lb-ub
            "cs": _make_numeric_parameter("cs", dim, 0, 1.0),
            "cc": _make_numeric_parameter("cc", dim, 0, 1.0),
            "cmu": _make_numeric_parameter("cmu", dim, 0, 1.0),
            "c1": _make_numeric_parameter("c1", dim, 0, 1.0),
            "damps": _make_numeric_parameter("damps", dim, 0, 10.0),
        }
    )
    cs.add(ForbiddenGreaterThanRelation(cs["mu0"], cs["lambda0"]))

    return cs


def get_configspace(dim: int = None, only_modules: bool = False) -> ConfigurationSpace:
    cspace = ConfigurationSpace()
    for name, options in get_all_module_options().items():
        cspace.add(Categorical(name, options, default=options[0]))

    if only_modules:
        return cspace

    if dim is None:
        warnings.warn(
            "Filling configspace with default numeric values for dim=2, "
            "since no dim was provided and only_modules was set to False"
        )
        dim = 2

    cspace.add_configuration_space("", _get_numeric_config(dim), delimiter="")
    return cspace


def settings_from_config(
    dim: int, 
    config: Configuration, 
    lb: np.ndarray = None, 
    ub: np.ndarray = None
) -> Settings:
    modules = Modules()
    via_setings = {}
    default_config = get_configspace(dim).get_default_configuration()
    for name, value in dict(config).items():
        if hasattr(modules, name):
            setattr(modules, name, value)
            continue
        if default_config[name] != value:
            via_setings[name] = value
    
    settings = Settings(dim, modules, lb=lb, ub=ub, **via_setings) 
    return settings
