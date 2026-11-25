# Contributing

Contributions are welcome: bug reports and questions as
[issues](https://github.com/IOHprofiler/ModularCMAES/issues), changes as pull requests.

## Development setup

A C++17 compiler and CMake ≥ 3.18 are needed.

```bash
git clone https://github.com/IOHprofiler/ModularCMAES.git
cd ModularCMAES
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .                 # builds the extension into build/{wheel_tag}
python -m unittest discover -s tests -t .  # the Python tests
```

After a change to the C++ code, run `python -m pip install --no-build-isolation -e .` again: it
recompiles only what changed.

The C++ library, its unit tests and the benchmark harness build with CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/modcma_bench --dims 2,10,100
```

## Layout

| path | contents |
|---|---|
| `include/`, `src/` | the C++ library |
| `bindings/` | the Python bindings (nanobind), with the docstrings of the whole API |
| `modcma/` | the Python package; `modcma/c_maes/cmaescpp/*.pyi` are the generated type stubs |
| `tests/` | Python tests (`test_c_*.py`), C++ tests (`tests/cpp`), golden trajectories, the statistical comparison |
| `docs/`, `zensical.toml` | the documentation site ([Zensical](https://zensical.org)) |
| `examples/` | the examples of the documentation, run by the tests |
| `bench/`, `scripts/performance/` | benchmarks ([the benchmarks](https://modularcmaes.readthedocs.io/en/latest/performance/)) |
| `scripts/` | experiment scripts of the papers ([reproducing the paper](https://modularcmaes.readthedocs.io/en/latest/reproducibility/)) |

## What a change needs

* **Tests.** `python -m unittest discover -s tests -t .` and `ctest` pass.
* **Results.** `tests/golden.py` keeps the trajectories of 260 configurations per compiler
  (`tests/data/golden/`). A change that is not meant to change results must leave them bit-exact
  (`python -m tests.golden check --drivers all`). A change that does change results (a bug fix, a
  new default) needs a reason in the pull request and a statistical comparison on BBOB
  (`python tests/statistical.py`, which needs `pip install ioh`); the fixtures are then regenerated
  (`python -m tests.golden write`), and CI produces the fixtures of the other toolchains as
  artifacts.
* **Docstrings.** Every name bound in `bindings/*.cpp` has a docstring (numpy style). After a
  change of the bindings, regenerate the stubs with `python tools/gen_stubs.py` and commit them;
  `tests/test_c_stubs.py` checks that nothing is undocumented.
* **Documentation.** Every Python block in `docs/` and `README.md` is run by `tests/test_c_docs.py`
  (skip one with `<!-- no-test -->` on the line before it); the examples in `examples/` too. Build
  the site with `pip install -r docs/requirements.txt && zensical serve`. The option tables of
  the guide are generated from the stubs: `python tools/gen_docs.py` after `tools/gen_stubs.py`.
* **Performance.** Changes in the hot path report before/after timings of `modcma_bench` (min of at
  least 5 repetitions).
* **Style.** C++: `.clang-format` (100 columns). Python: PEP 8, 100 columns.
* **Changelog.** User-visible changes get a line in [`CHANGELOG.md`](https://github.com/IOHprofiler/ModularCMAES/blob/master/CHANGELOG.md).

## Rules that hold for every change

* **The golden fixtures are never regenerated to make a test pass.**
  * Regenerate them only from the commit before the change, on the same machine, with the
    maintainer's approval.
  * The pull request gives the reason.
* **Ask-tell gives the same results as `run()`**, bit for bit, for every module. A new module that
  needs evaluations of its own does them as a phase of the generation, as TPA does.
* **A new module is off by default**, and leaves the results of every existing configuration
  unchanged.
* **Signatures take mu before lambda**, in C++ and in Python.
* **The Python API stays source-compatible.** A name that is renamed or removed keeps a deprecated
  alias, or raises an error that says what to use instead. The change gets a line in the
  changelog.

## A module that evaluates the objective

No module calls the objective, and no module queues points itself. `ModularCMAES`
(`include/c_maes.hpp`, `src/c_maes.cpp`) holds the whole flow: `advance()` runs the steps of a
generation in order (`BEGIN`, `RESTART`, `SAMPLE`, `POPULATION`, `SELECT`, `UPDATE`) and is the
only place that queues points; `answer()` counts every evaluation and passes each request's values
to `dispatch()`, a `switch` over the request's source that calls the module. Adding a module that
evaluates takes three changes, all in `ModularCMAES`:

* **A source:** a value of `Source`, and its `case` in `dispatch()`, which passes the values (`f`,
  the points `X`, and whether `tell(f, X)` gave them) to the module through `p`.
* **A call at its step** in `advance()`, which asks the module for its points and queues them
  (`queue(X, source)`). A module whose next point depends on the previous value answers
  "do I need another point, and which?" (as repelling's `has_query()` and `query_point()`); its
  step then runs again until the module is done, as `RESTART` does for the hill-valley tests and
  `POPULATION` for the batches of sequential selection. A module that needs the objective at a new
  moment of the generation adds a step.
* **The budget:** a fixed number of points per generation, the same in every generation of a
  configuration, goes into `Parameters::reserved_points()`, which the budget checks reserve. A
  module with a variable number of points stops asking when the next one would not leave room for
  the rest of the generation: the hill-valley tests of repelling check `Parameters::restart_fits()`
  before each point.

**Random numbers** are drawn when the module computes its points, never later. Ask–tell then stays
identical to `run()`.

## Planned work

Planned work is tracked in the [issues](https://github.com/IOHprofiler/ModularCMAES/issues), by label:

| Label | What | Detail of the issue |
|---|---|---|
| [`bug`](https://github.com/IOHprofiler/ModularCMAES/labels/bug) | Known bugs | Each issue is one pull request. |
| [`module`](https://github.com/IOHprofiler/ModularCMAES/labels/module) | Planned modules | The algorithm, the API, an implementation plan, the interactions with the other modules, and the validation. |
| [`framework`](https://github.com/IOHprofiler/ModularCMAES/labels/framework) | Framework features | |
| [`decision`](https://github.com/IOHprofiler/ModularCMAES/labels/decision) | Design questions for the maintainer | |

The record of the 2.0 overhaul (work packages WP0–WP8, the reviews, the decisions) is
[`dev/overhaul-2.0.md`](https://github.com/IOHprofiler/ModularCMAES/blob/master/dev/overhaul-2.0.md).
