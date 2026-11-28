# Changelog

## 2.0.0 (unreleased)

modcma is now the C++ implementation only. Upgrading from 1.x: see the
[migration guide](https://modularcmaes.readthedocs.io/en/latest/migration/).

### Removed

* The pure-Python implementation (`modcma.ModularCMAES(func, d, ...)`, `AskTellCMAES`,
  `evaluate_bbob`, `Parameters`/`BIPOPParameters` of `modcma.parameters`, the samplers of
  `modcma.sampling`, `modcma.utils`, the `python -m modcma` command line). It had not been developed
  for several releases; it remains in modcma 1.x (`pip install "modcma<2"`) and on the branch
  `legacy-python`. Its top-level names (`modcma.ModularCMAES(func, d)`, `AskTellCMAES`,
  `evaluate_bbob`, `modcma.utils`) raise an error that says where they went; importing from the
  modules `modcma.parameters`, `modcma.sampling` and `modcma.utils` raises `ModuleNotFoundError`.
* `ioh` is no longer a dependency (only the Python implementation used it).
* `Modules.repelling_restart` (1.2.0): replaced by `Modules.repelling_type` (Added); `True` is
  `COVERAGE`, `False` is `NONE`. The old name raises a plain `AttributeError` (a `TypeError` in
  `settings_from_dict`), without a pointer to the new one.
* Internal helpers of repelling: `repelling.hill_valley_test`, `TabooPoint.shares_basin` and
  `Repelling.update_archive` (the optimizer runs the hill-valley tests itself, #84); and
  `mutation.Strategy.mutate`.
* `CovarianceAdaptation.inv_root_C` (internal): use `es.p.adaptation.metric().inv_root_C()`, which
  every matrix adaptation has (#123).
* `restart.BIPOP.budget`, `budget_small` and `budget_large` (internal): the regimes count the
  evaluations they used, `used_small` and `used_large` (#80).
* `restart.BIPOP.mu_factor` (internal): µ after a restart follows the new λ, as in IPOP (#150).
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
  takes the values of repaired or injected points X instead of the asked ones. The steps of the
  1.x API (`mutate()`, `select()`, `recombine()`, `adapt()`) still work; `select()`,
  `recombine()` and `adapt()` raise `RuntimeError` while the points of a generation wait for their
  values (after `ask()`, or an interrupted `mutate()` or `step()`).
* **Vectorized objectives**: `run(f, vectorized=True)` and `step(f, vectorized=True)` evaluate a
  whole batch per call.
* **Gradient injection** (#90): with the gradient of the objective, every generation one point
  along the negative gradient from the mean, x = m − σ c C g / √(gᵀ C g) with c = E‖z‖ of the
  sampler, replaces a sample (with mirrored sampling, the point and its mirror replace a pair;
  Hansen 2011, "Injecting external solutions into CMA-ES", Eq. 14; libcmaes'
  gradient injection). `run(f, gradient=g)`, `step`, `mutate` and `fmin(..., gradient=g)`; with
  ask–tell, `es.inject_gradient(g)` while `es.accepts_gradient`. Without a gradient nothing
  changes. On BBOB f2, f10 and f12 (d = 10 to 100) the exact gradient saves 8 to 46 % of the
  evaluations; a bad gradient costs about the replaced sample. Guide: "Coming from gradient-based
  optimization". In C++: `GradientType`, `ModularCMAES::inject_gradient`, `accepts_gradient` and
  `Adaptation::apply_transpose_into` (out = Aᵀ v).
* **Per-optimizer random generators**: `Settings(seed=...)`; without a seed an optimizer copies the
  global generator (`modcma.set_seed`) when it is created. Optimizers run independently in parallel
  threads; the module supports free-threaded CPython 3.14t (no GIL).
* `Settings(fast_sampling=True)`: xoshiro256++ and a ziggurat for the normal samples, up to twice
  as fast for the cheap matrix adaptations (a different random stream; off by default).
* **LAPACK and BLAS** in high dimension: the eigendecomposition, Cholesky factorizations, inverses
  and matrix products, from d = 100 to 120 (`constants.eigen_solver`, `cholesky_solver`, `blas`,
  `lapack_min_dim`, `blas_min_dim`). Apple Accelerate on macOS, scipy's LAPACK and BLAS on Linux and
  Windows. `modcma.build_info()`.
* `Modules.repelling_type`, in place of 1.2.0's `repelling_restart`: `NONE` (the default),
  `COVERAGE` (the repelling restarts of de Nobel et al. 2024) and `ADAPTIVE` (experimental, see
  Changed). `CenterPlacement.NOVELTY_WEIGHTED` and `MAXIMIN_TABOO` place the center of a restart
  away from the points of the repelling archive.
* `Modules.repelling_metric`, the distance of repelling: `CURRENT` (the default; the Mahalanobis
  distance in the current distribution, which 1.x used), `EUCLIDEAN` and `ARCHIVED` (the shape,
  scaled to determinant 1, of the run that found the point).
* **`StepSizeAdaptation.TPA_INPLACE`** (#173): two-point step-size adaptation within the
  population, as pycma 4.6.0's `CMAAdaptSigmaTPA` (after Hansen, Atamna & Auger 2014): offspring 1
  and 2 are a mirrored pair along the last mean shift, no extra evaluations; its σ trajectory on
  scripted values is pycma's to rounding. Off by default; appended last to the enum, so the other
  options keep their values. `mutation.TPAInPlace`.
* Wheels for Linux aarch64, Windows ARM64 and macOS x86_64 as well; CPython 3.10–3.14 and 3.14t,
  one abi3 wheel for 3.12+.
* Docstrings for the whole API (`help()`, editors), type stubs (with `py.typed`), and a new
  documentation site.
* `ModularCMAES.center_of_mass`, a (d,) array: the mean of the search distribution as a point of
  the objective (integer variables rounded, inside the bounds with a bound correction), pycma's
  `xfavorite`; with noise and a large population often closer to the optimum than the best
  evaluated point, whose recorded value is biased low (#24).
* `stats.centers` (#111): a record of every run, a `Center`: its evaluated center of mass (`x`,
  `y`), the step size `sigma` and the shape `metric` the run ended with, the `reasons` it ended (the
  met restart criteria, `"budget"`, `"target"`, `"max_generations"`) and `std`, the longest
  semi-axis of its one-sigma ellipsoid. `ModularCMAES.stop_conditions()` tells whether the
  optimizer should stop. Every run now ends with an evaluation of its center of mass (see Changed,
  "The end of a run").
* `RecombinationWeights.UTILITY`: the utilities of NES/xNES (summing to zero) for the mean and every
  matrix update; with `NATURAL_GRADIENT` and `ssa=XNES` it is xNES (median evaluations within 2 %
  of `cmaes`' XNES on the sphere and the ellipsoid, d = 5 to 20).
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
* **Saving and resuming a run** (#91): `pickle`, `copy.deepcopy` and `copy.copy` work for
  `ModularCMAES`, `Parameters`, `Settings`, `Modules` and every module object
  (`__getstate__`/`__setstate__`). The state includes the random generator and the positions of
  the samplers, the generation in progress (the points of `ask()` that wait for `tell()`) and the
  restart criteria defined in Python; a loaded run continues bit for bit. The objective is not
  saved. The state is a versioned binary format with a checksum, the same in C++
  (`include/serialization.hpp`: `serialization::save`, `serialization::load<T>`); a state of a newer
  format, or a corrupted one, raises `ValueError`. A how-to: checkpointing a long run
  ([saving and resuming a run](https://modularcmaes.readthedocs.io/en/latest/guide/checkpointing/)).
  Results are unchanged.
* `Adaptation.keeps_metric()`, whether `metric()` costs nothing; `restart.Strategy.max_iter`, the
  generations a restart strategy allows a run; `restart.MaxSigma.start`; `restart.BIPOP.used_small`,
  `used_large` and `large_run` (#80). `restart.MaxDSigma.sigma_start` and
  `restart.MinDSigma.sigma_start`, the step size at the start of the run (#139).
* `Weights.cs_path`, the learning rate of the path p_σ (#72);
  `CholeskyAdaptation.skipped_downdates`, the downdates of the active update that were skipped;
  `MatrixAdaptation.inverse_fallbacks`, `inverse_recompute_interval` and
  `inverse_recompute_max_dim`, of the update of M⁻¹ with elitism (#74).
* Internals: `Population.n_pool`, the size of the (µ+λ) pool of elitist selection, whose dropped
  points stay after the population (#161); `MatrixAdaptation.hs` and `CholeskyAdaptation.hs` (#160),
  `SeparableAdaptation.hs` (#176); `TPA.sigma_old` (#167); `mutation.Strategy` hooks for samples of
  the step-size rule (#173, C++).
* Also bound: `ModularCMAES.has_pending`; `Settings.integer_variables` (read-only);
  `restart.IPOP.max_lambda_factor` and `mutation.MSR.quantile` (#83);
  `mutation.Strategy.generation_threshold` (#78); `Weights.mueff_mean`, `utility` and `pairwise`;
  `restart.NoImprovement.tolerance`; `sampling.FastGaussian` (`fast_sampling`).
  `NaturalGradientAdaptation.G` is read-only, computed from the new `gradient_samples`,
  `gradient_weights` and `gradient_shift` (internal).
* `docs/performance.md`: benchmarks against pycma, `cmaes`, evosax and EvoTorch.
* `tools/count_variants.py` counts the distinct variants from the code, and the guide documents
  the counts (Modules, "How many variants"): 21,954,240 for the 14 modules of the framework paper
  with today's options (3,136,320 for the 13 algorithmic modules, without bound correction),
  708,676,416 for all 17 fields of `Modules` (with `TPA_INPLACE`, #173). The paper counted
  14,859,936 (#130).

### Deprecated

* The misspelled names of 1.x (above); `modcma.c_maes.utils.set_seed`, `build_info` and
  `compute_ert` (use `modcma.set_seed`, ...); and `AdaptiveRepelling.target_rejection_rate` and
  `local_target_rejection_rate`, which never had an effect.

### Changed

* The configuration space of `get_configspace` and `settings_from_config` (#81). The numeric
  hyperparameters keep their names. Their defaults are those of `Settings(dim)`, so the defaults
  of `cs` and `damps` follow CSA's new rate (#138; at d = 5, cs 0.434 → 0.365). Their ranges are
  unchanged except the lower bound of `cs`, `cc` and `damps` (#169, below).
  * Hyperparameters without an effect are inactive (conditions): `active`, `center_placement`,
    `repelling_type`, `repelling_metric`, `cs`, `cc`, `cmu`, `c1`, `damps`, `mu0`, and (for
    lambda0 = 1) `elitist`, `sequential_selection`, `weights` and `ssa` (#131). A sampled
    configuration no longer contains them; the evidence is in the source. A configuration with a
    value for an inactive hyperparameter is not valid for `Configuration(space, ...)`;
    `settings_from_config` still takes it as a dict.
  * `sample_transformation=NONE` is not in the space (`get_all_module_options` still lists it).
  * `COVARIANCE` with another `ssa` than `CSA`, and `COVARIANCE` (#131), `IPOP`, `BIPOP` or
    `mirrored=PAIRWISE` (#136) with `lambda0 = 1`, are forbidden: `Settings` turns them into
    another configuration of the space.
    `get_configspace(remove_duplicates=False)` keeps them.
  * `settings_from_config` passed `mu0` only when it differed from its default, so a sampled `mu0`
    equal to the default was dropped and `lambda0 = 20, mu0 = 3` ran with `mu0 = 10`. It passes
    `mu0` now whenever `lambda0` is passed.
* Requires Python ≥ 3.10 and numpy ≥ 2.0 (was Python ≥ 3.8).
* The bindings use nanobind instead of pybind11; the Python API of `modcma.c_maes` is unchanged by
  the port (checked against a dump of the pybind11 API before it, `tests/api_dump.py`). The dump
  was taken after the release of 1.2.0, so it does not cover `Modules.repelling_restart` (Removed).
* Much faster: an in-place population and allocation-free sampling, batched linear algebra, LAPACK
  and BLAS in high dimension, an exact low-rank update for `NATURAL_GRADIENT` (80× faster at
  d = 1000) and an O(1) stagnation criterion. A generation is 20–47× faster than pycma up to d = 10.
* Results differ from 1.x in the last bits (batched linear algebra, LAPACK in high dimension), and
  where bugs were fixed (below); the differences are statistically indistinguishable on BBOB.
* One path for every evaluation of the objective (#84): the optimizer runs the fixed steps of a
  generation and queues the points its modules need, and ask–tell, `step()`, `run()` and
  `mutate()` evaluate them the same way. Results are unchanged, but with sequential selection
  TPA's two points are now evaluated with the population's first batch (before: with its last
  batch, or after a population that sequential selection stopped). The order of the objective
  calls changes, and so does the evaluation count recorded with the best solution (`Solution.e`)
  when it is one of TPA's points. Every evaluation is counted in one place, when its value
  arrives.
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
* **CSA's learning rate** is cs = (µ_eff + 2)/(d + µ_eff + 5), as in Hansen's tutorial
  (arXiv:1604.00772v2, Table 1, Eq. (55)), `cmaes` and modcma 1.0. modcma 1.1 and 1.2 had pycma's
  d + µ_eff + 3, without pycma's cap of cs at 1/2 (#138). The rate of the step-size update,
  cs/damps, is 15 %, 12 % and 6 % smaller at d = 2, 5 and 20 with the default λ. Every
  configuration with CSA changes; on BBOB (d = 2, 5, 20) the median number of evaluations changes
  by +1 % (geometric mean), threshold convergence needs 9–52 % fewer evaluations on the unimodal
  functions at d = 2 and 5, and the other step-size rules are unchanged.
* **The learning rates of the xNES step-size rules and of `NATURAL_GRADIENT`** (#72). `XNES`
  without `NATURAL_GRADIENT` has the rate of Krause, Glasmachers & Igel (2017), Table 3,
  cs = µ_eff/(2 ln d √d), instead of 0.01: on the sphere (d = 10) it needs 750–1070 evaluations
  instead of 3650–9950. `NATURAL_GRADIENT` gives xNES's η_σ only to `XNES`; the other step-size
  rules keep their own cs (and CSA its damps), and `cc`, the rate of A, is xNES's η_B/2 with every
  rule (it was cs/2). `XNES` and `LPXNES` weight the µ best with `Weights.positive`, which sums to
  1, as in Krause et al.; with `UTILITY` they used the positive utilities, which sum to less. The
  rate of the path p_σ is cs, at most 1 (`Weights.cs_path`).
* **The active update of `SEPARABLE`, `MATRIX` and `CHOLESKY`** is that of `COVARIANCE` (#73; Hansen
  2016, Eq. (46), (47); pycma): the negative weights are rescaled by d/‖A⁻¹y‖², and the decay of
  `MATRIX` and `CHOLESKY` includes them. For `MATRIX` (MA-ES has no active update) this is the
  active CMA-ES update written in M. `CMSA` has no active variant: `Settings` switches `active` off
  (with negative weights C became indefinite), and `get_configspace` has no `active`, `c1` and `cmu`
  for it.
* **The default weights for µ ≠ λ/2** (a given `mu0`, and the default µ = ⌊λ/4⌋ of SA and CMSA)
  are those of a population of 2µ, as pycma with `CMA_mu` (#73): ln(µ + ½) − ln i for the
  µ best, and the µ negative weights on the worst ranks (0 between them when λ > 2µ; the last λ − µ
  of them when λ < 2µ). They were ln((λ + 1)/⌊λ/µ⌋) − ln i for all ranks. With pairwise selection
  every negative weight now falls on a loser, so the active update has no effect there.
* **`SEPARABLE`'s learning rates** are pycma's separable rates (`CMA_diagonal`, pycma 4.6.0): c1 =
  1/(d + 2√d + µ_eff/d), cµ = min(1 − c1, (0.25 + µ_eff + 1/µ_eff − 2)/(d + 4√d + µ_eff/2)), the
  cap last (#73). They were the full update's c1 and its cµ × (d + 2)/3, attributed to Ros & Hansen
  (2008) but not checked against the paper. c1 is 4× to 40× larger at d = 10 to 100; `acov` no
  longer affects `SEPARABLE`.
* `get_configspace`: `cs` is inactive with `NATURAL_GRADIENT` for `ssa` `SR` and `SA` (its default
  `cc` no longer reads `cs`, #72).
* **The end of a run** (#111): every run ends with an evaluation of its center of mass, as pycma's
  `eval_final_mean`, also when the optimizer stops (it was evaluated only at a restart). The budget
  always keeps that evaluation: a generation runs only when it leaves one, `Settings` rejects a
  budget smaller than the first generation plus one, and when the generations would use the budget
  up exactly, the last one is dropped for the center. `break_conditions()` holds once the final
  center of mass is evaluated: the last `ask()` returns it, a single point, and the last `step()`
  evaluates only it; after it the 1.x steps `select()`, `recombine()` and `adapt()` do nothing.
  `stats.solutions`, like `stats.centers`, has one entry per run, the last one too. Trajectories are
  unchanged until the last generations of a run (checked on all golden configurations, 260 at the
  time).
* **The restart criteria for every matrix adaptation** (#80; Hansen 2016, App. B.3; pycma):
  * `TolX` and `NoEffectCoord` are used with every matrix adaptation, from the diagonal of C (and
    p_c where there is one); `ConditionC` and `NoEffectAxis`, which read `Adaptation.metric()`,
    with `COVARIANCE`, `SEPARABLE` and `NONE`. Before, the four were used only with `COVARIANCE`,
    so not when a step-size adaptation other than CSA turned it into `COVARIANCE_NO_EIGV`.
  * `NoEffectAxis` and `NoEffectCoord` test, as pycma, whether m + 0.1σDᵢbᵢ equals m, and
    mᵢ + 0.2σ√Cᵢᵢ equals mᵢ, in floating point; they compared the step with 10 ε. Their
    `tolerance` is now the step in standard deviations (0.1, 0.2).
  * `MaxSigma` is Hansen's TolXUp: σ max(D) grew by more than 1e4 since the start of the run. It
    compared σ with 1e4, whatever σ₀ and C. With `constants.clip_sigma = True` (off by default) σ
    is still clipped to at most 1e4 (`MaxSigma.tolerance`).
  * `MaxSigma`, `MaxDSigma` and `MinDSigma` read max(D) and min(D) from the metric for
    `COVARIANCE`, `SEPARABLE` and `NONE`, and √max Cᵢᵢ and √min Cᵢᵢ for the other matrix
    adaptations (they used D = 1 for every type but `COVARIANCE`).
* **BIPOP** (#80; Hansen 2009, pycma's `fmin`): each regime counts the evaluations of its runs, the
  first run as small (pycma), and a restart uses the regime that has used fewer. A small run takes
  about half the evaluations of the large regime so far: at most ⌈½ used_large/λ⌉ generations,
  pycma's cap, rounded up (`Strategy.max_iter`, enforced by `ExceededMaxIter`; #80, #139).
  λ of a small run is rounded down, as in the paper; up to even only with `PAIRWISE` (it was always
  even).
* **BIPOP's large population** stops growing at `BIPOP.max_lambda_factor` λ₀ = 2⁹ λ₀, the limit of
  IPOP, and restarts continue at that size; it had no limit (#140). pycma's `fmin` has none
  either, but stops after `restarts` large runs, 2⁹ λ₀ with `restarts=9` (Hansen 2009). The limit
  acts from the 10th large run on; the golden runs and the statistical grid (3000 d) reach 8 λ₀ at
  most, and are unchanged. `restart.BIPOP.lambda_large` (internal) is now the λ of the latest large
  run instead of the next one, and the new `restart.BIPOP.max_lambda_factor` sets the limit.
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
* **The lower bound of `cs`, `cc` and `damps` in `get_configspace` is 1e-6, not 0** (#169), since
  `Settings` rejects a value of 0 (#147). Names and defaults are unchanged. A configuration saved
  from the old space still loads unless a value is exactly 0 (a draw of a normal distribution,
  so practically never). `c1` and `cmu` keep [0, 1]; a sample with `c1 + cmu > 1` makes
  `settings_from_config` raise `ValueError`, which a tuner should record as a crash (documented on
  the tuning page).
* **`Settings` rejects learning rates that make a run diverge, and `max_generations` of 0** (#147),
  in the constructor and the setters, with a `ValueError` that names the field: `cs`, `cc` in
  (0, 1]; `c1`, `cmu` in [0, 1]; `damps` > 0; `acov` > 0 and finite; `max_generations` >= 1. `None`
  is still the default. `c1 + cmu > 1`, which makes the decay of C negative, is rejected at creation
  when both are given. Before, `cs=3` or `damps=-1` ran and ended with sigma at 1e26 or more.
  Hansen (2016) has c_c <= 1, c_sigma < 1, c_1 + c_mu <= 1; `cs = 1` stays allowed (it works).
  The default of one of `c1`, `cmu` against a given other is not checked. `es.p.settings.<field> =`
  on a running optimizer checks the field, as the setters do, but not the checks between fields
  (`c1 + cmu > 1` is accepted there).
* **`Settings` rejects `mu0` above `lambda0`** (#136) with a `ValueError` that names both; 1.x
  replaced it by `lambda0 // 2` (but not with SA or CMSA). A (1+1)-ES (`lambda0 = 1`) still uses
  µ = 1 whatever `mu0` is. **A NaN `target`** raises `ValueError` too: it never stopped a run.
  `None` is no target.
* **`lambda0 = 1` with `mirrored=PAIRWISE`** (#136, F17 of #75): the (1+1)-ES turns `PAIRWISE`
  into `NONE`. Before, `PAIRWISE` made λ = 2 with µ = 0 and NaN weights. `MIRRORED` stays.
* **`ssa=SA`: the learning rate of the sampled step sizes** is CMSA's τ = 1/√(2d) (Beyer &
  Sendhoff 2008; Spettel, Beyer & Hellwig 2018, Table I), for the one step size per point (#71);
  it was 1/√(2µ_eff). `LPXNES` keeps ln 2/(√d ln max(d, 2)) (Krause et al. 2017), `sample_sigma`
  with the other rules 1/√(2µ_eff).
* Internal: `BoundCorrection.correct_point` corrects a point that is not a sample (TPA's points
  when the mean is outside the box, #98, and the gradient point, #90);
  `SigmaSampler(dimension)` no longer uses the dimension (one step size per individual, #71).
* **TPA's test points are m ± ½σ⟨y⟩**, half the last step of the mean with the step size that
  sampled it, as Hansen (2008, Eqs. (8), (9)); they were m ± σ_new⟨y⟩, about one step (#167).
  Every TPA configuration gives other runs; on BBOB the length changed the evaluations by 2 % or
  less in the investigation.
* **`MATRIX` and `CHOLESKY` stall their rank-one update by h_σ** with every step-size rule but
  `CSA`, as `COVARIANCE` does (Hansen 2016, after Eq. (45); #160): `CHOLESKY` stalls p_c and
  compensates the lost variance in the decay, `MATRIX` skips its rank-one term, which reads p_σ.
  With `XNES` and `SA` the path stayed long, the rank-one term stretched the matrix and the runs
  stalled: on the ellipsoid (d = 5 / 10 / 20) `MATRIX` + `XNES` solved 0 / 0 / 0 of 20 runs,
  `CHOLESKY` + `XNES` 3 / 0 / 0; now 20 / 20 / 20 each. With `CSA` the results are unchanged
  (MA-ES with CSA is as published).
* **`COVERAGE` repelling's radius** (#158): the volume is in σ₀^d units,
  V = n_T S / (c σ₀^d N), so the radius does not change when the box and σ₀ are scaled together
  (Eq. (10) of de Nobel et al. 2024, "Avoiding Redundant Restarts", divides by σ₀), and the radius
  has no extra factor 1/√d (their Eq. (11) has none). Computed in logarithms, so it stays finite at
  large d (the volume of [−5, 5]^d overflowed from d = 309). σ₀ is the problem's in every run, also
  in BIPOP's small runs, as Eq. (10) has it (documented). The distance is `CURRENT`, the
  Mahalanobis distance in the current distribution (their Eq. (8)), as in 1.x (during the 2.0
  work the default had become `EUCLIDEAN`). In the experiment of #158 (BBOB f19, f21, f22;
  d = 5, 10, 20) the new radius with `CURRENT` solved 128 of 270 runs, 101 without repelling.
* `repelling_type=ADAPTIVE` is documented as experimental: modcma's own, without a published
  source; it solved fewer runs than no repelling in the experiment of #158.

### Fixed

Checked against the original sources (Hansen's tutorial, pycma, the papers of each module); most
change results, compared on BBOB before and after (`tests/statistical.py`):

* The covariance update had the h_sigma correction with the wrong sign (Hansen 2016, pycma).
* With a lazy eigendecomposition (large d), the covariance matrix itself was only updated every few
  generations, losing the updates in between; now every generation, as pycma.
* The negative weights of the active update were scaled with 1 + c1 / mu instead of 1 + c1 / cmu.
* `NATURAL_GRADIENT` ignored `sigma0` (A started at I / sigma) and its learning rate was
  (9 + 3 + ln d) / (5 d sqrt d) instead of xNES's (9 + 3 ln d) / (5 d sqrt d).
* BIPOP's small populations were at most the initial lambda; now they are between it and the
  latest large population, as pycma's `fmin` (Hansen 2009's text: half of it; #140). IPOP's growth
  limit never applied, and a non-integer `ipop_factor` acted as its integer part.
* BIPOP's budgets left wrapped around (unsigned) when a run used more than its regime had left:
  one regime then ran for the rest of the budget (Rastrigin, d = 2, budget 20000, seed 0) (#80).
* µ after an IPOP or BIPOP restart kept the ratio µ₀/λ₀, rounded by IPOP and truncated by BIPOP.
  It is now the default of the new λ, λ/2 (λ/4 with SA and CMSA), as pycma's `fmin` and Hansen
  (2009); with a given `mu0`, µ₀λ/λ₀ rounded, the same in both strategies and in the cap of the
  first generation at the budget. This changes runs with an odd λ₀ (d = 3: λ = 14 had µ = 6, now 7
  as pycma), with SA or CMSA when λ₀ is not a multiple of 4, and BIPOP with a given `mu0` (#150).
* `ExceededMaxIter` let a run take `max_iter` + 2 generations; now at most `max_iter`, as pycma's
  `maxiter` (#139).
* `MaxDSigma` and `MinDSigma` compared σ with `Settings.sigma0`, also in BIPOP's small runs, which
  start smaller; now with the step size of the run's start, as pycma's `tolupsigma` (#139).
* `AdaptiveRepelling`'s running average of the run lengths, which sets how much a radius grows when
  a run finds a basin again, counted the first run as one evaluation and started at 1; it now
  starts from the first run's length (`average_restart_evaluations` is 0 before the first
  restart). A run that left no finite point was not counted, so the next run counted both (#121).
* BIPOP's random draws (λ of a small run, its σ) and the random choice of a center when
  `NOVELTY_WEIGHTED`'s scores are not usable used `std::uniform_real_distribution` and
  `std::uniform_int_distribution`, whose algorithms depend on the standard library: the same seed
  gave other runs with libstdc++, libc++ and MSVC. They draw from the generator's raw output now
  (#164).
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
* `set_module` did not accept enum members; `Population.__add__` changed its left operand; an
  empty `CachedSampler` was not checked.
* h_σ's bias correction counted evaluations/λ over all runs (restarts and TPA's points included)
  instead of the generations of the current run (Hansen 2016, pycma); `Stats.t_run` (#77).
* Sequential selection compared with the best of all runs, so after a restart it hardly ever
  stopped a generation; it now compares with the best of the previous generation, as the framework
  paper (the first generation of a run is complete) (#77).
* TPA counted a tie of its two points (as after every restart, when the mean has not moved) as a
  success, so σ grew by 16 % after each restart and diverged with elitism; a tie now gives no
  signal, as in pycma (#78). (TPA with elitism still diverged for another reason: #179, below.)
* Sampled step sizes (`sample_sigma`, `SA`, `LPXNES`) were drawn per coordinate: a point m + S∘y
  was not along y (no rotation invariance), and the signal of SA was averaged over d draws. Each
  point now has one step size, as in CMSA-ES and lp-xNES; CMSA with SA solves a rotated ellipsoid
  (d = 10) in 10 of 10 runs, before in 0 of 5 (#71, F1).
* The mean shift Δm of the evolution paths (and of TPA's points) was the step of the mean divided
  by the mean of all step sizes of the population. It is now divided, per coordinate, by the step
  size recombined from the selected points (Hansen 2016, Eq. (24)): by σ when every selected point
  has the step size σ, by σ raised by the lower bound of integer variables (as Marty et al. 2024,
  Eqs. (5), (7)), and by the recombined sampled step size (as CMSA recombines its σ). On a
  mixed-integer ellipsoid (d = 10, 5 integer variables) the median evaluations fell from 6351 to
  2461 (25 runs) (#71, F14).
* TPA's two points were not bound-corrected: TPA evaluated points outside the box, and one could
  be reported as the best. With a bound correction both now move toward the mean by the same
  factor until both are in the box, so the test steps stay symmetric (Hansen 2008); corrected one
  by one by the method (as pycma's boundary handler does), TPA lost runs (#98).
* PSR ranked equal values in the order of the previous generation's ranking, so the same values
  gave a different step size depending on the history (with elitism the parents are in both
  generations). Equal values now share their mean rank (mid-ranks), so a tie counts for neither
  generation (#120).
* With `lambda0 = 1` (the (1+1)-ES, which runs SR) `COVARIANCE` was kept or converted to
  `COVARIANCE_NO_EIGV` according to the `ssa` given: the default `CSA` kept the eigendecomposition
  that SR does not use, and `ssa` changed the run. The conversion now follows the rule that runs:
  every `ssa` gives the same run, with `COVARIANCE_NO_EIGV` (#131).
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
* Uniform random numbers could be exactly 0 or 1, giving infinite normal samples.
* Shared state between optimizers in the normal sampler (Box–Muller) and the generators.
* A non-finite covariance matrix counted as a successful adaptation, so a run could evaluate NaN
  until the budget was used up; now it restarts.
* Sequential selection kept unevaluated columns in the population.
* `COVARIANCE_NO_EIGV`: the restart criteria that read the eigenvalues saw the initial ones.
* `MaxDSigma`, `MinDSigma` and `NoEffectAxis` used the fourth root of the eigenvalues instead of
  the square root (now as pycma).
* The stagnation criterion took O(t) per generation (5× slower long BIPOP runs).
* `LPXNES`: β = ln 2/(√d ln d) was infinite at d = 1 (now ln(max(d, 2))); its cs, 9µ_eff/(10√d),
  is above 2 for µ_eff > 2.22√d (d = 2, λ = 12, or IPOP), and as the rate of p_σ it made p_σ NaN.
  With `NATURAL_GRADIENT` it had xNES's rate and did not converge (#72). Its cs is now capped at 1
  (ours; above 1 the update overshoots, 1.29 at d = 2 with the default λ), and the rate of p_σ is
  at most 1 for every rule.
* `SEPARABLE`: the cap c1 + cµ ≤ 1 was applied before cµ was multiplied by (d + 2)/3, so with large
  populations (IPOP) cµ exceeded 1 and the decay of C was negative (#73; now pycma's rates, capped
  last).
* The default weights with µ > λ/2 (and some µ < λ/2, such as µ = 4, λ = 10) had positive values
  among the "negative" weights, which the active update applied as positive (#73).
* `CHOLESKY` with the active update used a factor that a failed rank-one downdate had left partly
  updated; the downdate is now skipped (#74).
* `MATRIX` with elitism updated the inverse of M incrementally, which is not the inverse of the
  updated M, and it drifted from the inverse (relative error 0.98 after 60 generations), so the
  elites' steps were wrong. It is now updated by the Woodbury identity, in O(k d²) as before, and up
  to d = 30 computed exactly from M after every 400 generations of a run, which bounds the
  rounding errors that the update accumulates with the active update at small d (#74; the
  constants `MatrixAdaptation.inverse_recompute_interval` and `inverse_recompute_max_dim` are the
  maintainer's choice). On BBOB (d = 5, 20) it solves 249 instead of 105 of 350 runs.
* `NATURAL_GRADIENT` with `active` used CMA-ES's negative weights, which made σ shrink even on a
  linear function; with the default weights it now uses xNES's utilities (`weights=UTILITY`).
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
  adaptation, #123); `MuCommaLambdaES` stays within its budget and sorts NaN last.
* `Settings.center` with an infinite bound (#134): it was 0 in every coordinate, so the first
  mean could start outside the box. Per coordinate: the midpoint for two finite bounds, 0 for two
  infinite bounds, and for one infinite bound 0 if the half-line holds it, else the finite bound
  moved inward by `sigma0`. `center` now follows `sigma0` too. Finite boxes are unchanged.
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
* **Mirrored and pairwise sampling** (#75).
  * The mirrored pairs are formed per generation: individuals 2i and 2i + 1 are z and -z, and a
    pair that `RESAMPLE` or repelling rejects is sampled again as a pair. With sequential
    selection the partner of a batch's last individual is sampled with it. Before, the mirror
    state carried over between generations and through rejections, so the pairs fell out of step
    with `RESAMPLE`, repelling, an odd λ or sequential selection (0 of 5 pairs mirrored with
    `RESAMPLE`). With an odd λ the last individual is unpaired; with λ = 1, `MIRRORED` has no
    effect (it mirrored the sample of every second generation, around the new mean).
  * `PAIRWISE`: the losers of the pairs keep their objective values (they got f = +inf). The
    selection sorts the winners, the unpaired points and the parents of elitism, then the losers
    (`Population.n_ranked`). `MSR` had shrunk σ in every generation, and `stats.current_avg` and
    the medians of `Stagnation` were infinite. `MSR` and `PSR` compare the ranked points; the
    losers still get no negative weights.
  * `PAIRWISE` with an odd default λ0: the default µ0 is half of the even λ0 (it was half of the
    odd one: 3 of 8 at d = 3). A given `mu0` is kept.
* **Orthogonal sampling** (#76). The blocks start with every generation: the first min(n, d)
  samples, then blocks of at most d (n = λ, or ⌈λ/2⌉ with mirroring); with λ < d a block spanned
  several generations. The "- 2 for TPA" in the block size is gone (TPA's points are extra). The
  orthonormal vectors are now those of Gram-Schmidt (diag(R) > 0): sample j of every block was
  biased towards -e_j (d = 5: E[z_j[j]] = -0.80 ... -0.70), which made σ grow without selection.
  `orthogonal` now converges with `lambda0 = 2` (rotated ellipsoid, d = 5: 0 → 5 of 5 runs) and
  with `EQUAL` weights and `TPA`, `MSR` or `PSR` (shifted sphere, d = 5: 0 → 10 of 10 runs each).
  `functions::random_rotation_matrix` (the C++ examples) had the same bias: its rotations are now
  uniformly distributed.
* **The expected length E‖z‖** of the `SCALED_UNIFORM`, `LAPLACE`, `LOGISTIC` and
  `DOUBLE_WEIBULL` transformations (and `NONE`) is that of their distribution, computed when the
  sampler is created (`DOUBLE_WEIBULL`: Γ(d + 1/2)/Γ(d)); it was √d, up to 23 % too long at
  d = 2, which biased CSA and threshold convergence (#76).
* **HALTON and SOBOL runs agree across platforms**, as with the default sampler (#116). The digit
  permutations of `HALTON` used `std::shuffle`, and the order of the `SOBOL` points
  `std::uniform_int_distribution`, whose results depend on the standard library: on macOS and Linux
  the same seed gave different runs. Both now draw bounded integers from the raw mt19937 output
  (`rng::below`, a Fisher-Yates shuffle). In the golden set of the release (macOS clang 21 against
  Linux gcc 13, σ and the best f at generation 10 within a relative 1e-12), 15 of 16 `HALTON` and
  12 of 14 `SOBOL` runs agree (none before), and 238 of 241 with the default sampler; the rest
  differ in the last bits (re-measured for #165). The order of the Sobol points was a constant
  stride through them (the multiplier of its generator was 1 modulo the number of points); the
  multiplier is now random, = 5 (mod 8).
* Internals: `sampling.get` no longer wraps a `Mirrored` sampler (the optimizer mirrors the
  pairs; `sampling.Mirrored` remains, standalone); `Sampler.begin_generation`; `Orthogonal` takes
  the samples per generation; `selection.Pairwise.sort` and `NoPairwise.sort`, and
  `NoPairwise.__call__` sorts (it did nothing); `Population.n_ranked`.
* **Elitism with the active update** (#161): the negative weights of every matrix update with an
  active update went to ranks µ+1..λ of the (µ+λ) pool, which are parents and good offspring, and
  the worst points got none. They now go to the λ − µ worst points of the pool, as the framework
  paper defines the active update and pycma does with `CMA_elitist`; selection is unchanged.
  `COVARIANCE` with `elitist` and `active` solved BBOB f10 and f12 (d = 10) in 0 of 25 runs. On
  BBOB (7 functions, d = 5 and 20, 25 runs) it solves 250 instead of 85 of 350 runs.
* **XNES with elitism** (#172): its step-size update (both forms, xNES SA and the σ part of the
  natural gradient) read the parents among the µ best, whose z are short; σ settled 3 to 14 times
  too small and the runs stalled on the ellipsoid (d ≥ 10: at most 3 of 20 runs). It now reads the
  offspring alone, ranked among themselves, as xNES and xNES SA; selection is unchanged.
* **TPA and TPA_INPLACE with elitism** (#179): a generation whose (µ+λ) selection keeps every
  parent (no offspring among the µ selected) is a failure. Its mean did not move, and `TPA`
  compared two test points that differed from m only by rounding: σ diverged (to 1e205 to 1e307;
  the compatibility page said it collapsed). In the next generation `TPA` now evaluates no test
  points (`ask()` returns λ points; the budget still reserves them) and takes r = −`a_tpa`;
  `TPA_INPLACE` samples ordinary offspring instead of its test pair and takes z = −1. No TPA
  source covers elitism; the rule follows the success rules of elitist ESs (Rechenberg's 1/5th
  rule, the (1+1)-CMA-ES). On a sphere and two ellipsoids (d = 5, 10, 20; 20 runs each) `TPA`
  with `elitist` solved 15 of 180 runs and `TPA_INPLACE` 130; both solve 180. On BBOB (7
  functions, d = 5 and 20, 25 runs) 56 → 243 and 174 → 251 of 350. Without `elitist` nothing
  changes.
* **The best point lies in the box** (#168): with a bound correction, only a point in [lb, ub]
  becomes `stats.global_best`, what `run()` and `fmin` return, and in the standalone ES classes
  `OnePlusOneES.x` and `MuCommaLambdaES.x_min`. `RESAMPLE` evaluates a point outside the box after
  5d rejected tries, and such a point could be returned. The trajectories are unchanged (except
  `OnePlusOneES`, whose parent is its best point).
* The reproducibility guide said that the same seed gives the same run on every platform below the
  LAPACK thresholds. It is the same on the same platform and toolchain; across platforms the last
  bits differ (FMA contraction, libm), and an ambiguous eigenbasis (C with equal eigenvalues, with
  the default settings from d = 12) can amplify them from the first generations (#165).
* **XNES with the active update** (#175): `COVARIANCE` + `XNES` + `active` solved the ellipsoid
  (d = 10 and 20, 20 runs) in 1 and 0 runs, `CHOLESKY` in 1 and 0, `MATRIX` in 20 and 11; without
  `active` all three solve 20 of 20. xNES SA keeps σ too small and h_σ at 0 in most generations
  (89–99 % at d ≥ 10). In that nearly linear surrounding the negative weights undid the growth of
  C towards the optimum. With `ssa=XNES` a generation with h_σ = 0 now has no negative weights
  (`COVARIANCE`, `COVARIANCE_NO_EIGV`, `SEPARABLE`, `CHOLESKY`, `MATRIX`): 20 of 20 at each d.
  Not with `weights=UTILITY`, whose flat negative utilities did not fail and were 15–20 % slower
  without them. This is modcma's rule; Hansen (2016) and pycma keep the negative weights for
  every h_σ, as modcma does with the other step-size rules. BBOB (7 functions, d = 5 and 20, 25
  runs): `XNES` + `active` solves 150 of 350 runs (144 before), with no cell worse.
* `repelling.archive`, `stats.solutions` and `stats.centers` returned references into the C++
  vectors, although the docstrings said copies (#176). A new entry reallocates a vector (and the
  archive is sorted before each population), so a list held across `step()` read freed memory and
  could segfault, and a write to an element changed the optimizer. Reading now returns a copy.
* The compatibility page lists `ssa=XNES` with `weights=EQUAL` as a combination to avoid (#180):
  0 of 20 runs solve the ellipsoid at d = 5, 10 and 20, with or without `active`.
* `mutation.SequentialSelection` (internal) accepted a `seq_cutoff_factor` below 1, NaN or inf. Below
  1 a population could stop after µ or fewer individuals, and the updates read columns that the
  generation had not evaluated (SA raised an error); NaN and inf have no cutoff. It now raises
  `ValueError` (#191). The optimizer always uses 1.
* Elitism appended a parent with f = +inf when the previous generation also held NaN values, or
  losers of pairwise selection with a value, and parents with f = NaN: it counted the parents
  with `Population.n_finite`, which counts every f but +inf. The parents are now the leading
  ranked points with f < +inf (#191). Only runs whose objective returns +inf or NaN change; the
  golden trajectories are bit-identical.
