from enum import Enum

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


def get_module_options(name: str) -> tuple:
    if not hasattr(Modules, name):
        raise NameError(f"Modules has no member {name}")
    
    default_value = getattr(Modules(), name)
    if isinstance(default_value, bool):
        return (default_value, not default_value)
    
    if issubclass(default_value.__class__, Enum):
        other_values = [
            x for x in default_value.__members__.items() 
            if x is not default_value
        ]
        return tuple([default_value] + other_values)
    breakpoint()
    raise TypeError(f"{name} has a unparsable type {type(default_value)}")

def get_all_module_options() -> dict:
    return {
        name: get_module_options(name) 
        for name in dir(Modules) 
        if not name.startswith("_")
    }
