# Settings validation

Where the settings are checked (`include/settings.hpp`): the constructor of `Settings`, the
setters, which check only their own field, and `require_valid`, which `Parameters` calls before it
builds the modules. What is rejected (`std::invalid_argument`, a `ValueError` in Python) and what is
adjusted, such as the (1+1)-ES and PAIRWISE with an odd λ. The principle: reject only what crashes
or is invalid, and allow the rest as an experiment, documented on the
[compatibility](../guide/compatibility.md) page. How to add a setting or a check, and the Python
side (`settings_from_dict`, the conditions of `get_configspace`).

!!! note "Planned"
    This page is an outline. Its three parts are written in step 2 of
    [#186](https://github.com/IOHprofiler/ModularCMAES/issues/186).

## How it works

## How to change it

## When it breaks
