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
* **Results.** `tests/golden.py` keeps the trajectories of 258 configurations per compiler
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

The plan of the larger work packages is [`TASKS.md`](https://github.com/IOHprofiler/ModularCMAES/blob/master/TASKS.md).
