# The bindings

The nanobind code in `bindings/`: the order of the definitions in `module.cpp`, and the patterns it
uses: the caster for Python objectives, the setters of bool members, the conversion of sequences to
vectors, `def_pickle`, the trampoline for restart criteria written in Python (`NB_TRAMPOLINE`), the
sampler written in Python, and the restart criteria as an opaque list. The docstring rule. The
generated files and their checks: `tools/gen_stubs.py` (the stubs), `tools/gen_docs.py` (the option
tables of the guide), `tests/test_c_stubs.py`, and `tests/api_dump.py` with its `ACCEPTED` list;
deprecated aliases (`warn_deprecated`).

!!! note "Planned"
    This page is an outline. Its three parts are written in step 2 of
    [#186](https://github.com/IOHprofiler/ModularCMAES/issues/186).

## How it works

## How to change it

## When it breaks
