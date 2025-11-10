from enum import Enum
from dataclasses import dataclass

from .cmaescpp import (
    constants,
    utils,      # pyright: ignore[reportMissingModuleSource]
    sampling,   # pyright: ignore[reportMissingModuleSource]
    mutation,   # pyright: ignore[reportMissingModuleSource]
    selection,  # pyright: ignore[reportMissingModuleSource]
    parameters, # pyright: ignore[reportMissingModuleSource]
    bounds,     # pyright: ignore[reportMissingModuleSource]
    restart,    # pyright: ignore[reportMissingModuleSource]
    options,    # pyright: ignore[reportMissingModuleSource]
    repelling,  # pyright: ignore[reportMissingModuleSource]
    Population,
    Parameters,
    ModularCMAES,
    center,     # pyright: ignore[reportMissingModuleSource]
    es          # pyright: ignore[reportMissingModuleSource]
)

from .cmaescpp.parameters import Settings, Modules # pyright: ignore[reportMissingModuleSource]

from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float, NormalIntegerHyperparameter, NormalFloatHyperparameter


def get_module_options(name: str) -> tuple:
    if not hasattr(Modules, name):
        raise NameError(f"Modules has no member {name}")
    
    default_value = getattr(Modules(), name)
    if isinstance(default_value, bool):
        return (default_value, not default_value)
    
    module_class = default_value.__class__
    if issubclass(module_class, Enum):
        other_values = [
            x for x in module_class.__members__.values() 
            if x is not default_value
        ]
        return tuple([default_value] + other_values)
    raise TypeError(f"{name} has a unparsable type {type(default_value)}")

def get_all_module_options() -> dict:
    return {
        name: get_module_options(name) 
        for name in dir(Modules) 
        if not name.startswith("_")
    }

def make_numeric_parameter(name: str, dim: int, lb: float = 0, ub: float = float("inf")) -> NormalIntegerHyperparameter | NormalFloatHyperparameter:
    settings = Parameters(Settings(dim))
    default = getattr(settings.weights, name, None)
    print(name, default)

    


def get_all_numeric_options(dim: int) -> dict[str, NormalIntegerHyperparameter | NormalFloatHyperparameter]:
    return {
        "sigma0": make_numeric_parameter("sigma0", dim),
        "lambda0": make_numeric_parameter("lambda0", dim, 4),
        "mu0": make_numeric_parameter("mu0", dim, 4),
        "cs": make_numeric_parameter("cs",  dim, 0, 1.0),
        "cc": make_numeric_parameter("cc", dim, 0, 1.0),
        "cmu": make_numeric_parameter("cmu", dim, 0, 1.0),
        "c1": make_numeric_parameter("c1", dim, 0, 1.0),
        "damps": make_numeric_parameter("damps", dim, 0, 10.0),
        "acov": make_numeric_parameter("acov", dim, 0, 10.0),
    }

def get_configspace(dim: int = None, only_modules: bool = False) -> ConfigurationSpace:
    cspace = ConfigurationSpace()
    for name, options in get_all_module_options().items():
        cspace.add(Categorical(name, options, default=options[0]))

    if only_modules:
        return cspace
    
    if dim is None:
        print("warning")
        dim = 1
        
    for name, options in get_all_numeric_options(dim).items():
        cspace.add()

    return cspace