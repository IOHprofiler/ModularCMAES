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
* `c_maes.utils`: `set_seed`, `build_info` and `compute_ert` moved to the main namespace
  (`modcma.set_seed`, ...; the old names still work, with a `DeprecationWarning`). Removed: `cdf`,
  `ppf` (use `scipy.stats.norm`), `random_uniform`, `random_normal`; the internals of
  the Sobol sampler (`i8_sobol`, `Shuffler`, `CachedShuffleSequence`, `sampling.Sobol.cache`) are
  no longer exposed. A removed name raises an error that says what to use instead.

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
* Docstrings for the whole API (`help()`, editors), type stubs (with `py.typed`), and a new
  documentation site.
* `ModularCMAES.xfavorite`: the mean of the search distribution as a solution, as pycma's
  `xfavorite`; with noise and a large population often closer to the optimum than the best
  evaluated point, whose recorded value is biased low (#24).
* `RecombinationWeights.UTILITY`: the utilities of NES/xNES (summing to zero) for the mean and every
  matrix update; with `NATURAL_GRADIENT` and `ssa=XNES` it is xNES (the same evaluations as
  `cmaes`' XNES within 1 %).
* Correctly spelled names next to the old ones: `selection.Elitism`, `NoElitism`,
  `repelling.euclidean`, `mahalanobis`, `Adaptation.adapt_evolution_paths_inner`.
* `docs/performance.md`: benchmarks against pycma, `cmaes`, evosax and EvoTorch.

### Changed

* Requires Python ≥ 3.10 and numpy ≥ 2.0 (was Python ≥ 3.8).
* The bindings use nanobind instead of pybind11; the Python API of `modcma.c_maes` is unchanged
  (checked against a dump of the 1.x API).
* Much faster: an in-place population and allocation-free sampling, batched linear algebra, LAPACK
  and BLAS in high dimension, an exact low-rank update for `NATURAL_GRADIENT` (80× faster at
  d = 1000) and an O(1) stagnation criterion. A generation is 20–47× faster than pycma up to d = 10.
* Results differ from 1.x in the last bits (batched linear algebra, LAPACK in high dimension), and
  where bugs were fixed (below); the differences are statistically indistinguishable on BBOB.

### Fixed

Checked against the original sources (Hansen's tutorial, pycma, the papers of each module); most
change results, compared on BBOB before and after (`tests/statistical.py`):

* The covariance update had the h_sigma correction with the wrong sign (Hansen 2016, pycma).
* With a lazy eigendecomposition (large d), the covariance matrix itself was only updated every few
  generations, losing the updates in between; now every generation, as pycma.
* The negative weights of the active update were scaled with 1 + c1 / mu instead of 1 + c1 / cmu.
* `NATURAL_GRADIENT` ignored `sigma0` (A started at I / sigma) and its learning rate was
  (9 + 3 + ln d) / (5 d sqrt d) instead of xNES's (9 + 3 ln d) / (5 d sqrt d).
* BIPOP's small populations were at most the initial lambda (Hansen 2009: up to half of the large
  one); IPOP's growth limit never applied, and a non-integer `ipop_factor` acted as its integer part.
* Orthogonal sampling did not produce orthogonal samples (the norms were read while overwritten),
  and with lambda > d it now orthogonalizes groups of d.
* With mirrored or orthogonal sampling CSA normalized with sqrt(d) instead of E||N(0, I)||.
* `EXPONENTIAL` weights follow the paper, w_i = 2^-i + 1 / (mu 2^mu).
* `TolX` compared with tolerance * sigma0^2 instead of tolerance * sigma0, and used the signed path.
* p_sigma started at ones for MATRIX, CHOLESKY, CMSA, NATURAL_GRADIENT and NONE (zero after a
  restart).
* Without `x0` the first run started at the origin, restarts at the center of the bounds.
* A run stopped one generation early when the budget was a multiple of lambda (budget 5000, lambda
  10: 4990 evaluations); with TPA it could exceed the budget.
* A restart near the end of the budget could exceed it: by up to lambda + 1 (the restart center, and
  the population that IPOP and BIPOP had grown), and by the hill-valley tests of repelling, which had
  no budget check. The first generation after a restart now takes at most the rest of the budget,
  and the hill-valley tests stop when the budget runs short.
* Points given to `tell(f, X)` far outside the distribution enter the update clipped (Hansen 2011,
  as pycma).
* COTN could place a coordinate outside the bounds for a large sigma.
* `OnePlusOneES` checked the bounds of the parent, not of the candidate; `MuCommaLambdaES` shared a
  sampler between instances.
* `Settings`: `mu0 > lambda0` was not corrected with SA or CMSA; the C++ default `verbose = true`
  made `ModularCMAES(d)` print at every restart; changing `lb`/`ub` left `center`, `db`, ... stale.
* `set_module` did not accept enum members; `Population.__add__` changed its left operand;
  `utils.random_normal` could return a number from before `set_seed`; out-of-range indices in
  `CachedShuffleSequence` and an empty `CachedSampler` were not checked.
* Sequential selection with the active update broke the eigendecomposition (#42; fixed by the
  sequential-selection and active-update fixes below, now tested).

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
* `NATURAL_GRADIENT` with `active` used CMA-ES's negative weights, which made σ shrink even on a
  linear function; it now uses xNES's utilities (`weights=UTILITY`).
* `fmin` ignored `x0` (it used only its length) and started at the origin.
* Type hints: `Settings(x0=, lb=, ub=)` accept numpy arrays (the stubs said sequences), and the
  annotations of `fmin`, `settings_from_dict`, `settings_from_config` and `get_configspace` were
  wrong (`**config: dict`); a type check in CI.
* `str(Modules)` printed `center_placement: unknown` for `CENTER`.
* Linux wheels were compiled for the CPU of the build machine (`-march=native`).
