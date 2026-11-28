# Random numbers

Where the random numbers of a run come from: the global generator (`rng::GENERATOR`, `set_seed`)
and the generator of each optimizer (`Parameters::generator`, from `Settings.seed` or
`rng::fork()`), and which objects draw from it. The draws that give the same numbers with every C++
standard library (`rng::below`, `rng::shuffle`, `rng::uniform`; #116, #164), and the xoshiro256++
generator of `fast_sampling`. What is reproducible across platforms and what is not: results are
bit-exact only with the same compiler and libraries, hence three golden fixtures. What a user can
rely on is on the page [Reproducibility and threads](../guide/reproducibility.md).

!!! note "Planned"
    This page is an outline. Its three parts are written in step 2 of
    [#186](https://github.com/IOHprofiler/ModularCMAES/issues/186).

## How it works

## How to change it

## When it breaks
