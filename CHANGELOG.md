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
* `CovarianceAdaptation.inv_root_C` (internal): use `es.p.adaptation.metric().inv_root_C()`, which
  every matrix adaptation has (#123).
* `restart.BIPOP.budget`, `budget_small` and `budget_large` (internal): the regimes count the
  evaluations they used, `used_small` and `used_large` (#80).
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

* **Presets** for published algorithms that are a selection of modules: `modcma.preset(name, dim, ...)`,
  `Settings.preset(...)` and `fmin(..., preset=name)`; names: `cma-es`, `active-cma-es`, `sep-cma`,
  `ma-es`, `cholesky-cma`, `cmsa-es`, `xnes`, `ipop`, `bipop`, `(1+1)-cma-es`. Keyword arguments
  override the preset; `modcma.preset_names()` lists the names (#92).
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
* `ModularCMAES.center_of_mass`: the mean of the search distribution as a solution, pycma's
  `xfavorite`; with noise and a large population often closer to the optimum than the best
  evaluated point, whose recorded value is biased low (#24).
* **Every run ends with an evaluation of its center of mass**, the last one too, as pycma's
  `eval_final_mean` (#111). `stats.centers` holds a record of every run, a `Center`: the evaluated
  point (`x`, `y`), the step size `sigma` and the shape `metric` the run ended with, the `reasons` it
  ended (the met restart criteria, `"budget"`, `"target"`, `"max_generations"`) and `std`, the
  longest semi-axis of its one-sigma ellipsoid. `ModularCMAES.stop_conditions()` tells that the
  optimizer should stop; `break_conditions()` holds once the final center of mass is evaluated.
* `RecombinationWeights.UTILITY`: the utilities of NES/xNES (summing to zero) for the mean and every
  matrix update; with `NATURAL_GRADIENT` and `ssa=XNES` it is xNES (the same evaluations as
  `cmaes`' XNES within 1 %).
* Correctly spelled names: `selection.Elitism`, `NoElitism`, `Strategy.elitism`,
  `repelling.euclidean`, `mahalanobis`, `Adaptation.adapt_evolution_paths_inner`, the keyword
  `seq_cutoff_factor`. The old names still work, with a `DeprecationWarning` (#82).
* Python constructors for the step-size adaptations (`mutation.CSA`, `TPA`, ...) and
  `CovarianceNoEigvAdaptation`.
* `Adaptation.metric()`: the shape of the sampled distribution, without σ, for every matrix
  adaptation: a `matrix_adaptation.Metric` with the axis lengths `d`, the eigenvectors `B`,
  `eigenvalues()`, `inv_C()` and `inv_root_C()` (#123). `COVARIANCE`, `SEPARABLE` and `NONE` keep
  it; the others compute it when asked (O(d³)) and reuse it until their factor changes
  (`Adaptation.factor_version`). Results are unchanged.
* **Warm starting** (#89): `Settings(C0=...)`, the initial covariance matrix C (default: the
  identity), for every matrix adaptation (`SEPARABLE` uses its diagonal, `NONE` rejects it); every
  restart starts from it again. `modcma.warm_start(solutions, gamma, alpha)` estimates `x0`,
  `sigma0` and `C0` from the solutions of a related task, as WS-CMA-ES (Nomura et al. 2021) and
  `cmaes`' `get_warm_start_mgd`, and returns them as `Settings` arguments. Results without `C0`
  are unchanged.
* `Adaptation.keeps_metric()`, whether `metric()` costs nothing; `restart.Strategy.max_iter`, the
  generations a restart strategy allows a run; `restart.MaxSigma.start`; `restart.BIPOP.used_small`,
  `used_large` and `large_run` (#80).

* `docs/performance.md`: benchmarks against pycma, `cmaes`, evosax and EvoTorch.
* `tools/count_variants.py` counts the distinct variants from the code, and the guide documents
  the counts (Modules, "How many variants"): 21,430,080 for the 14 modules of the framework paper with today's options (3,061,440 for the 13
  algorithmic modules, without bound correction), 686,685,216 for all 17 fields of `Modules`. The
  paper counted 14,859,936 (#130).

### Deprecated

* The misspelled names of 1.x (above), and `AdaptiveRepelling.target_rejection_rate` and
  `local_target_rejection_rate`, which never had an effect.

### Changed

* The configuration space of `get_configspace` and `settings_from_config` (#81). The numeric
  hyperparameters keep their names, ranges and defaults.
  * Hyperparameters without an effect are inactive (conditions): `active`, `center_placement`,
    `repelling_type`, `repelling_metric`, `cs`, `cc`, `cmu`, `c1`, `damps`, `mu0`, and (for
    lambda0 = 1) `elitist`, `sequential_selection` and `weights`. A sampled configuration no longer
    contains them; the evidence is in the source. A configuration with a value for an inactive
    hyperparameter is not valid for `Configuration(space, ...)`; `settings_from_config` still
    takes it as a dict.
  * `sample_transformation=NONE` is not in the space (`get_all_module_options` still lists it).
  * `COVARIANCE` with another `ssa` than `CSA`, and `IPOP`, `BIPOP` or `mirrored=PAIRWISE` with
    `lambda0 = 1` (#136), are forbidden: `Settings` turns them into another configuration of the
    space.
    `get_configspace(remove_duplicates=False)` keeps them.
  * `settings_from_config` passed `mu0` only when it differed from its default, so a sampled `mu0`
    equal to the default was dropped and `lambda0 = 20, mu0 = 3` ran with `mu0 = 10`. It passes
    `mu0` now whenever `lambda0` is passed.
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
* **The end of a run** (#111): every run ends with an evaluation of its center of mass, also when
  the optimizer stops (it was evaluated only at a restart). The budget always keeps that
  evaluation: a generation runs only when it leaves one, `Settings` rejects a budget smaller than
  the first generation plus one, and when the generations would use the budget up exactly, the last
  one is dropped for the center. `break_conditions()` holds once the final center of mass is
  evaluated: the last `ask()` returns it, a single point, and the last `step()` evaluates only it;
  after it the 1.x steps `select()`, `recombine()` and `adapt()` do nothing. `stats.solutions`, like
  `stats.centers`, has one entry per run, the last one too. Trajectories are unchanged until the
  last generations of a run (all 260 golden configurations).
* **The restart criteria for every matrix adaptation** (#80; Hansen 2016, App. B.3; pycma):
  * `TolX` and `NoEffectCoord` are used with every matrix adaptation, from the diagonal of C (and
    p_c where there is one); `ConditionC` and `NoEffectAxis`, which read `Adaptation.metric()`,
    with `COVARIANCE`, `SEPARABLE` and `NONE`. Before, the four were used only with `COVARIANCE`,
    so not when a step-size adaptation other than CSA turned it into `COVARIANCE_NO_EIGV`.
  * `NoEffectAxis` and `NoEffectCoord` test, as pycma, whether m + 0.1σDᵢbᵢ equals m, and
    mᵢ + 0.2σ√Cᵢᵢ equals mᵢ, in floating point; they compared the step with 10 ε. Their
    `tolerance` is now the step in standard deviations (0.1, 0.2).
  * `MaxSigma` is Hansen's TolXUp: σ max(D) grew by more than 1e4 since the start of the run. It
    compared σ with 1e4, whatever σ₀ and C. `constants.clip_sigma` still keeps σ below 1e4.
  * `MaxSigma`, `MaxDSigma` and `MinDSigma` read max(D) and min(D) from the metric for
    `COVARIANCE`, `SEPARABLE` and `NONE`, and √max Cᵢᵢ and √min Cᵢᵢ for the other matrix
    adaptations (they used D = 1 for every type but `COVARIANCE`).
* **BIPOP** (#80; Hansen 2009, pycma's `fmin`): each regime counts the evaluations of its runs, the
  first run as small (pycma), and a restart uses the regime that has used fewer. A small run takes
  at most half the evaluations of the large regime so far (`ExceededMaxIter`, from
  `Strategy.max_iter`). λ of a small run is rounded down, as in the paper; up to even only with
  `PAIRWISE` (it was always even).
* **Setting a field of `Settings`** (#133, #136) checks that field alone, as the constructor does
  (`lambda0`, `mu0`, `sigma0`, `budget`, `x0`, `lb`, `ub`: a wrong length, NaN, +inf in `lb` or
  -inf in `ub`, a sign or a range). The checks between fields (`lb` <= `ub`, the integer
  variables in the box, `mu0` <= `lambda0`, the budget against the first generation, an odd
  `lambda0` with `PAIRWISE`, `lambda0` changed to or from 1) run in the constructor and again when
  an optimizer is created (`ModularCMAES(settings)`, `Parameters(settings)`), so the fields can be
  set in any order; this replaces the integer check that the setters of `lb` and `ub` made (#79).
  Setting `lambda0` recomputes `mu0`, unless `mu0` was given. `db`, `center`, `diameter` and
  `has_bounds` are read-only: they are derived from `lb` and `ub` (`db` and `center` are read-only
  views); set `lb`, `ub` or `x0` instead. Setting one raises Python's "property 'db' of 'Settings'
  object has no setter". The setters of `x0`, `lb` and `ub` take array_like, as the constructor
  (they took only numpy arrays).
* **`Settings` rejects `mu0` above `lambda0`** (#136) with a `ValueError` that names both; 1.x
  replaced it by `lambda0 // 2` (but not with SA or CMSA). A (1+1)-ES (`lambda0 = 1`) still uses
  µ = 1 whatever `mu0` is. **A NaN `target`** raises `ValueError` too: it never stopped a run.
  `None` is no target.
* **`lambda0 = 1` with `mirrored=PAIRWISE`** (#136, F17 of #75): the (1+1)-ES turns `PAIRWISE`
  into `NONE`. Before, `PAIRWISE` made λ = 2 with µ = 0 and NaN weights. `MIRRORED` stays.

### Fixed

Checked against the original sources (Hansen's tutorial, pycma, the papers of each module); most
change results, compared on BBOB before and after (`tests/statistical.py`):

* The covariance update had the h_sigma correction with the wrong sign (Hansen 2016, pycma).
* With a lazy eigendecomposition (large d), the covariance matrix itself was only updated every few
  generations, losing the updates in between; now every generation, as pycma.
* The negative weights of the active update were scaled with 1 + c1 / mu instead of 1 + c1 / cmu.
* `NATURAL_GRADIENT` ignored `sigma0` (A started at I / sigma) and its learning rate was
  (9 + 3 + ln d) / (5 d sqrt d) instead of xNES's (9 + 3 ln d) / (5 d sqrt d).
* BIPOP's small populations were at most the initial lambda; now they are between it and the last
  large population, as pycma's `fmin` (Hansen 2009's text: half of it; #140). BIPOP's large
  population has no upper limit, as in pycma. IPOP's growth limit never applied, and a non-integer
  `ipop_factor` acted as its integer part.
* BIPOP's budgets left wrapped around (unsigned) when a run used more than its regime had left:
  one regime then ran for the rest of the budget (Rastrigin, d = 2, budget 20000, seed 0) (#80).
* Orthogonal sampling did not produce orthogonal samples (the norms were read while overwritten),
  and with lambda > d it now orthogonalizes groups of d.
* With mirrored or orthogonal sampling CSA normalized with sqrt(d) instead of E||N(0, I)||.
* `EXPONENTIAL` weights follow the paper, w_i = 2^-i + 1 / (mu 2^mu).
* `TolX` compared with tolerance * sigma0^2 instead of tolerance * sigma0, and used the signed path.
* p_sigma started at ones for MATRIX, CHOLESKY, CMSA, NATURAL_GRADIENT and NONE (zero after a
  restart).
* Without `x0` the first run started at the origin, restarts at the center of the bounds.
* A run stopped one generation early when the budget was a multiple of lambda (budget 5000, lambda
  10: 4990 evaluations); with TPA it could exceed the budget. Now a generation runs when it leaves
  one evaluation for the final center of mass (#111): budget 5000, lambda 10 gives 499 generations
  and the center, 4991 evaluations.
* A restart near the end of the budget could exceed it: by up to lambda + 1 (the restart center, and
  the population that IPOP and BIPOP had grown), and by the hill-valley tests of repelling, which had
  no budget check. The first generation after a restart now takes at most the rest of the budget,
  and the hill-valley tests stop when the budget runs short.
* Points given to `tell(f, X)` far outside the distribution enter the update clipped (Hansen 2011,
  as pycma).
* A restart evaluated the raw mean, which can lie outside the box (with `UTILITY` weights, injected
  points, an `x0` outside the box, or integer rounding past an edge): wasted on SBOX-COST and
  archived by repelling. It now evaluates the center of mass, clipped when a bound correction is
  used (#111).
* COTN could place a coordinate outside the bounds for a large sigma.
* `OnePlusOneES` checked the bounds of the parent, not of the candidate; `MuCommaLambdaES` shared a
  sampler between instances.
* `Settings`: `mu0 > lambda0` was not caught with SA or CMSA (it is now rejected, #136); the C++
  default `verbose = true` made `ModularCMAES(d)` print at every restart; changing `lb`/`ub` left
  `center`, `db`, ... stale.
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
* The setters of `Settings` accepted what the constructor rejects (#133, #136): `lambda0 = 0`, or
  `mu0` above `lambda0`, crashed the process when the optimizer was created or run; a NaN `x0`,
  `lb`, `ub` or `sigma0`, a `budget` of 0 and `lb` above `ub` were accepted. They now raise
  `ValueError` (see Changed).
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
  `COVARIANCE_NO_EIGV` was the identity (it is now `metric().inv_root_C()`, computed for every
  adaptation, #123); `MuCommaLambdaES` stays within its budget
  and sorts NaN last.
* Infinite bounds and integer variables in the box (#79). Results with finite bounds, and integer
  bounds on the integer variables, are unchanged.
  * A box with an infinite bound turned the bound correction off in every coordinate, also at the
    finite bounds. Each coordinate and each side is now corrected on its own. `MIRROR`,
    `TOROIDAL`, `COTN` and `UNIFORM_RESAMPLE` need the width ub - lb: in a coordinate with one
    infinite bound they set a value that crosses the finite bound to that bound, as `SATURATE`.
    The final center of mass is clipped to every finite bound. `Settings.has_bounds` (the box is
    finite) no longer switches the correction off.
  * `UNIFORM`, `NOVELTY_WEIGHTED` and `MAXIMIN_TABOO` placed the centers of restarts at NaN when a
    bound was infinite. In such a coordinate they now sample uniformly in x0 ± 2 sigma0, shifted
    inside a finite bound.
  * An integer variable could be rounded out of the box: with lb = 0.3, the value 0.4 was
    evaluated at 0. An integer coordinate in the box is now rounded to the nearest integer in
    [ceil(lb), floor(ub)]. `Settings` reject an integer variable without an integer in [lb, ub]
    (lb = 0.6, ub = 0.9); for `lb` or `ub` set later, the optimizer rejects it when it is created
    (#133).
  * Internals: `CoordinateMapping.transform(x, lb=None, ub=None)` takes the bounds; the
    `correct_x` of every bound correction applies the rule for infinite bounds.
* Degenerate boxes (#133). A fixed coordinate (lb_i == ub_i) made `MIRROR` and `TOROIDAL` evaluate
  NaN points (division by the width 0); every correction now keeps it at lb_i. An empty coordinate
  (lb_i > ub_i, lb_i = +inf, ub_i = -inf, so also lb_i = ub_i = ±inf, or a NaN bound) raises
  `ValueError`, from `Settings` and when the optimizer is created; the setters of `lb` and `ub`
  check only their own field (see Changed). Results of boxes with positive widths are unchanged.
