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
* Internal helpers of repelling: `repelling.hill_valley_test`, `TabooPoint.shares_basin` and
  `Repelling.update_archive` (the optimizer runs the hill-valley tests itself, #84); and
  `mutation.Strategy.mutate`.
* `Parameters.start` and `Parameters.perform_restart` (internal): only the optimizer evaluates a
  restart (#84). To force one, add a restart criterion that is met: the next generation starts
  with the restart.
* Unused parameters and members of internals: `expected_length_z` of the matrix adaptations
  (their constructors take `(dimension, x0)`; `Weights.expected_length_z` is the one CSA uses),
  `sigma` of `Adaptation.restart` and of the `NaturalGradientAdaptation` constructor, `modules` of
  `Adaptation.adapt_matrix` (it is `settings.modules`), `stats`, `mu` and `lamb` of
  `NaturalGradientAdaptation.compute_gradients`, `lamb` of `Adaptation.adapt_evolution_paths` and of
  the step-size adaptations' `adapt`, `mu` and `lamb` of `adapt_evolution_paths_inner` (its
  deprecated alias `adapt_evolution_paths_innner` still accepts them), and `mueff` of
  `CoordinateMapping`.
* `c_maes.utils`: `set_seed`, `build_info` and `compute_ert` moved to the main namespace
  (`modcma.set_seed`, ...; the old names still work, with a `DeprecationWarning`). Removed: `cdf`,
  `ppf` (use `scipy.stats.norm`), `random_uniform`, `random_normal`; the internals of
  the Sobol sampler (`i8_sobol`, `Shuffler`, `CachedShuffleSequence`, `sampling.Sobol.cache`) are
  no longer exposed. A removed name raises an error that says what to use instead.

### Added

* `import modcma` gives the C++ API (`modcma.fmin`, `modcma.ModularCMAES`, `modcma.Settings`, ...);
  `modcma.c_maes` stays as an alias. `modcma.__version__`. `matrix_adaptation` is exported with the
  other submodules (1.x had it only as `modcma.c_maes.cmaescpp.matrix_adaptation`).
* **Ask–tell for every module**: `ask()` returns the points to evaluate as a `(k, d)` array,
  `tell(f)` takes their values; the results are identical to `run()`, bit for bit. `tell(f, X)`
  evaluates repaired or injected points instead of the asked ones. The steps of the 1.x API
  (`mutate()`, `select()`, `recombine()`, `adapt()`) still work; `select()`, `recombine()` and
  `adapt()` raise `RuntimeError` while the points of a generation wait for their values (after
  `ask()`, or an interrupted `mutate()` or `step()`).
* **Vectorized objectives**: `run(f, vectorized=True)` and `step(f, vectorized=True)` evaluate a
  whole batch per call.
* **Per-optimizer random generators**: `Settings(seed=...)`; without a seed an optimizer copies the
  global generator (`modcma.set_seed`) when it is created. Optimizers run independently in parallel
  threads; the module supports free-threaded CPython 3.14t (no GIL).
* `Settings(fast_sampling=True)`: xoshiro256++ and a ziggurat for the normal samples, up to twice
  as fast for the cheap matrix adaptations (a different random stream; off by default).
* **LAPACK and BLAS** in high dimension: the eigendecomposition, Cholesky factorizations, inverses
  and matrix products, from d = 100 to 120 (`constants.eigen_solver`, `cholesky_solver`, `blas`,
  `lapack_min_dim`, `blas_min_dim`). Apple Accelerate on macOS, scipy's LAPACK and BLAS on Linux and
  Windows. `modcma.build_info()`.
* `Modules.repelling_metric` (`EUCLIDEAN`, the previous behaviour; `CURRENT`; `ARCHIVED`).
* Wheels for Linux aarch64, Windows ARM64 and macOS x86_64 as well; CPython 3.10–3.14 and 3.14t,
  one abi3 wheel for 3.12+.
* Docstrings for the whole API (`help()`, editors), type stubs (with `py.typed`), and a new
  documentation site.
* One path for every evaluation of the objective (#84): the optimizer runs the fixed steps of a
  generation and queues the points its modules need, and ask–tell, `step()`, `run()` and
  `mutate()` evaluate them the same way. Results are unchanged, but with sequential selection
  TPA's two points are now evaluated with the population's first batch (before: with its last
  batch, or after a population that sequential selection stopped). The order of the objective
  calls changes, and so does the evaluation count recorded with the best solution (`Solution.e`)
  when it is one of TPA's points. Every evaluation is counted in one place, when its value
  arrives.
* `ModularCMAES.xfavorite`: the mean of the search distribution as a solution, as pycma's
  `xfavorite`; with noise and a large population often closer to the optimum than the best
  evaluated point, whose recorded value is biased low (#24).
* `RecombinationWeights.UTILITY`: the utilities of NES/xNES (summing to zero) for the mean and every
  matrix update; with `NATURAL_GRADIENT` and `ssa=XNES` it is xNES (the same evaluations as
  `cmaes`' XNES within 1 %).
* Correctly spelled names: `selection.Elitism`, `NoElitism`, `Strategy.elitism`,
  `repelling.euclidean`, `mahalanobis`, `Adaptation.adapt_evolution_paths_inner`, the keyword
  `seq_cutoff_factor`. The old names still work, with a `DeprecationWarning` (#82).
* Python constructors for the step-size adaptations (`mutation.CSA`, `TPA`, ...) and
  `CovarianceNoEigvAdaptation`.

### Deprecated

* The misspelled names of 1.x (above), and `AdaptiveRepelling.target_rejection_rate` and
  `local_target_rejection_rate`, which never had an effect.
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
* Modules now follow the framework paper and their original papers where the code had diverged
  (#83):
  * **Elitism** is (µ + λ) selection: the µ parents compete with the offspring (it kept all λ
    survivors). Their steps are recomputed with the generation's σ; before, each parent kept the
    step size it was sampled with.
  * **MSR** compares with the 0.3λ-th best of the previous generation (Ait ElHara et al. 2013,
    pycma), not with the median, and centers the count at λ/2, as pycma. It is 2–4× faster on the
    sphere and the ellipsoids, but solves Rosenbrock and the bent cigar somewhat less often within
    the budget. Its damping is `damps`, 2 − 2/d by default; at d = 1 it is 1,
    where it was 0.
  * **PSR** uses the constants of Loshchilov (2014): cs = 0.3 (was 0.9) and damps = 1 (was 2 − 2/d).
  * **SR** counts one success per generation: the generation's best beats the best of the run.
    TPA's points and the restart centers no longer count. The framework paper's reference, the
    previous generation's best, made σ diverge without elitism; the paper is to be corrected.
  * **IPOP:** the population size is limited to 2⁹ λ₀ (was (d λ₀)²).
  * **Pairwise selection with the active update:** the losers of the mirrored pairs get no negative
    weights, as in pycma.

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
* h_σ's bias correction counted evaluations/λ over all runs (restarts and TPA's points included)
  instead of the generations of the current run (Hansen 2016, pycma); `Stats.t_run` (#77).
* Sequential selection compared with the best of all runs, so after a restart it hardly ever
  stopped a generation; it now compares with the best of the previous generation, as the framework
  paper (the first generation of a run is complete) (#77).
* TPA counted a tie of its two points (as after every restart, when the mean has not moved) as a
  success, so σ grew by 16 % after each restart and diverged with elitism; a tie now gives no
  signal, as in pycma (#78).
* Threshold convergence used a threshold in units of x (a tenth of the box diameter) on z, so on
  boxes larger than the default [−5, 5]^d it stretched every sample. The threshold is now a
  multiple of E‖z‖ (`init_threshold` = 1, the same as before on the default box), computed once per
  generation. Past the budget (ask–tell) its remaining budget wrapped around. The internal
  `ThresholdConvergence.scale` is replaced by `threshold()` and `apply()` (#78).
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
* Invalid settings crashed the process or misbehaved silently (#113): `Settings(0)` and
  `lambda0=0` crashed at the first `ask()`, `x0`, `lb` or `ub` of the wrong length corrupted
  memory, and `mu0=0`, a non-positive `sigma0` or `lb > ub` were accepted. They now raise
  `ValueError` (also from the setters of `x0`, `lb`, `ub`, and the internal LAPACK helpers,
  `Population.keep_only` and `Population.n`); `tell(f, X)` with a 3-d `X` raises `ValueError`.
* `run()` and `fmin` ran a generation before they checked the budget, and a finished optimizer
  ran again at every `run()` (#114). The break conditions are now checked before every
  generation. Past the budget, the first generation after a restart no longer turns a (1+1)-ES
  into λ = 2.
* `Solution.x` (and so `fmin`'s `xopt`, `stats.global_best.x`) was a view of the optimizer's
  memory: it changed as the run continued, and a stored `current_best.x` read freed memory after
  the next improvement (#115). It is a copy now. The population arrays stay views; their
  docstrings say how long a view is valid.
* With elitism, `active` and µ = 1 the parent sits at the mean, and its zero step got an infinite
  negative weight: C became NaN and the run restarted about every three generations (#118). Every
  module option now runs with λ = 2 and 3 without NaN; many work poorly there (the compatibility
  page).
* A built-in restart criterion added from Python (`TolX`, `FlatFitness`, ...) crashed or never
  fired: its buffers were sized only at a restart (#119). `FlatFitness` read past the population
  after sequential selection stopped a generation early.
* Small fixes (#117): `CachedSampler` passes a restart's reset to the sampler it wraps; a zero
  eigenvalue is rejected like a negative one; LAPACK asked for in `constants` but missing falls
  back to Eigen; `resize_population` keeps the previous generation; `inv_root_C` of
  `COVARIANCE_NO_EIGV` is computed (it was the identity); `MuCommaLambdaES` stays within its budget
  and sorts NaN last.
