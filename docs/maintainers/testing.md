# The test tiers

The checks a change must pass, and how to read their output. The golden fixtures (`tests/golden.py`,
`tests/data/golden/`): what a run stores, the pairwise covering set of the module options and why a
new enum value changes it, the drivers (the ways of running a generation, and the resume drivers
that pickle the optimizer), the three toolchains and how to regenerate the fixture of each, and how
to debug a mismatch: which run, which driver, the first generation that differs. The statistical
tier (`tests/statistical.py`): how to run master against a branch on BBOB, and how to read the
comparison. The C++ tests (`tests/cpp/`, `ctest`), the tests of the documentation
(`tests/test_c_docs.py`), and the other Python tests.

!!! note "Planned"
    This page is an outline. Its three parts are written in step 2 of
    [#186](https://github.com/IOHprofiler/ModularCMAES/issues/186).

## How it works

## How to change it

## When it breaks
