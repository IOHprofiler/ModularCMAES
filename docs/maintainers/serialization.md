# Saving and resuming runs

How the state of an optimizer becomes bytes and back (#91, #183; `include/serialization.hpp`,
`src/serialization.cpp`). The functions `put` and `make` of each class, which list its members in
the order they are written; the type tags of the modules, by which the reader creates the right
class; the random generators, saved once each, so that the objects that share one still share it
after loading; the header, the format version and the CRC-32 checksum. How to add a member (a line
in `put` and in `make`, or a "Not saved" comment with the reason), when `format_version`
increases, and which older states `load()` still reads. The resume check,
`python -m tests.golden check --drivers resume`, and how to find the member that a failing run
does not save. What is saved, seen from the user, is on the page
[Saving and resuming a run](../guide/checkpointing.md).

!!! note "Planned"
    This page is an outline. Its three parts are written in step 2 of
    [#186](https://github.com/IOHprofiler/ModularCMAES/issues/186).

## How it works

## How to change it

## When it breaks
