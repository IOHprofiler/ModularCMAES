# Changelog

## 2.0.0 (unreleased)

modcma is now the C++ implementation only. Upgrading from 1.x: see the
[migration guide](https://modularcmaes.readthedocs.io/en/latest/migration/).

### Removed

* The pure-Python implementation (`modcma.ModularCMAES(func, d, ...)`, `AskTellCMAES`,
  `evaluate_bbob`, `Parameters`/`BIPOPParameters` of `modcma.parameters`, the samplers of
  `modcma.sampling`, `modcma.utils`, the `python -m modcma` command line). It had not been developed
  for several releases; it remains in modcma 1.x (`pip install "modcma<2"`) and on the branch
  `legacy-python`. Its names raise an error that says where they went.
* `ioh` is no longer a dependency (only the Python implementation used it).

### Added

* `import modcma` gives the C++ API (`modcma.fmin`, `modcma.ModularCMAES`, `modcma.Settings`, ...);
  `modcma.c_maes` stays as an alias. `modcma.__version__`.
* **Ask–tell for every module**: `ask()` returns the points to evaluate as a `(k, d)` array,
  `tell(f)` takes their values; the results are identical to `run()`, bit for bit. `tell(f, X)`
  evaluates repaired or injected points instead of the asked ones.
* **Vectorized objectives**: `run(f, vectorized=True)` and `step(f, vectorized=True)` evaluate a
  whole batch per call.
* **Per-optimizer random generators**: `Settings(seed=...)`; without a seed an optimizer copies the
  global generator (`utils.set_seed`) when it is created. Optimizers run independently in parallel
  threads; the module supports free-threaded CPython 3.14t (no GIL).
* `Settings(fast_sampling=True)`: xoshiro256++ and a ziggurat for the normal samples, up to twice
  as fast for the cheap matrix adaptations (a different random stream; off by default).
* **LAPACK and BLAS** in high dimension: the eigendecomposition, Cholesky factorizations, inverses
  and matrix products, from d = 100 to 120 (`constants.eigen_solver`, `cholesky_solver`, `blas`,
  `lapack_min_dim`, `blas_min_dim`). Apple Accelerate on macOS, scipy's LAPACK and BLAS on Linux and
  Windows. `utils.build_info()`.
* `Modules.repelling_metric` (`EUCLIDEAN`, the previous behaviour; `CURRENT`; `ARCHIVED`).
* Wheels for Linux aarch64, Windows ARM64 and macOS x86_64 as well; CPython 3.10–3.14 and 3.14t,
  one abi3 wheel for 3.12+.
* Docstrings for the whole API (`help()`, editors), type stubs, and a new documentation site.
* `PERFORMANCE.md`: benchmarks against pycma, `cmaes`, evosax and EvoTorch.

### Changed

* Requires Python ≥ 3.10 and numpy ≥ 2.0 (was Python ≥ 3.8).
* The bindings use nanobind instead of pybind11; the Python API of `modcma.c_maes` is unchanged
  (checked against a dump of the 1.x API).
* Much faster: an in-place population and allocation-free sampling, batched linear algebra, LAPACK
  and BLAS in high dimension, an exact low-rank update for `NATURAL_GRADIENT` (80× faster at
  d = 1000) and an O(1) stagnation criterion. A generation is 20–46× faster than pycma up to d = 10.
* Results differ from 1.x in the last bits (batched linear algebra, LAPACK in high dimension), and
  where bugs were fixed (below); the differences are statistically indistinguishable on BBOB.

### Fixed

* `integer_variables` rounded the coordinates 0, 1, ... instead of the listed indices.
* Active update: the negative weights of the default µ = λ/2 were not rescaled.
* `NATURAL_GRADIENT` recomputed a pseudo-inverse on every call of `invert_y`.
* Repelling restarts used the identity instead of the covariance of the taboo points (now the
  `repelling_metric` option, Euclidean by default as before).
* Uniform random numbers could be exactly 0 or 1, giving infinite normal samples.
* Shared state between optimizers in the normal sampler (Box–Muller) and the generators.
* A non-finite covariance matrix counted as a successful adaptation, so a run could evaluate NaN
  until the budget was used up; now it restarts.
* Sequential selection kept unevaluated columns in the population.
* `COVARIANCE_NO_EIGV`: the restart criteria that read the eigenvalues saw the initial ones.
* `MaxDSigma`, `MinDSigma` and `NoEffectAxis` used the fourth root of the eigenvalues instead of
  the square root (now as pycma).
* The stagnation criterion took O(t) per generation (5× slower long BIPOP runs).
* `fmin` ignored `x0` (it used only its length) and started at the origin.
* `str(Modules)` printed `center_placement: unknown` for `CENTER`.
* Linux wheels were compiled for the CPU of the build machine (`-march=native`).
