# CI and the release

The GitHub workflows in `.github/workflows/`: `Unittest` (`python-package.yml`), its jobs and its
matrix (which operating systems and Python versions run on a pull request and on master), the
compile cache (sccache), and the golden fixture that CI writes as an artifact; `Wheels`
(`python-publish.yml`), paused during the 2.0 development (#154). The steps of a release (#93).
What to do when a job fails, including the 503 errors of GitHub's API.

!!! note "Planned"
    This page is an outline. Its three parts are written in step 2 of
    [#186](https://github.com/IOHprofiler/ModularCMAES/issues/186).

## How it works

## How to change it

## When it breaks
