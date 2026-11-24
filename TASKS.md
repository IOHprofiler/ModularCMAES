# ModularCMAES (C++ version): nanobind, ask-tell, memory & compute — task list

Draft for sign-off. Nothing here is implemented yet. Scope: the C++ implementation (`src/`, `include/`,
`modcma/c_maes/`) and its packaging. The pure-Python implementation (`modcma/*.py`, incl.
`modcma/asktellcmaes.py`) is stale and is **not touched**; its tests only have to keep running (until
WP6 moves it to a branch, D9).

Work packages (WPs) are merged in order unless §2 says otherwise; one branch + PR per WP (or per task
for the big ones). Decisions that need the maintainer are marked **D1–D8** (§2).

---

## 0. Background: measurements on master (f09c60c)

Measured on this Mac (arm64, AppleClang, `-O3 -fno-math-errno`), sphere, default modules
(`COVARIANCE` + CSA, no restarts), seed 1, 3000 generations. The harness is in the session scratchpad.
T0.2 commits it.

**Time per generation and where it goes** (sample+eval = `Strategy::mutate`, incl. the C++ objective):

| d | λ | µs/gen | sample+eval | select/sort | matrix update (eigh) | eigh every |
|---|---|---|---|---|---|---|
| 2 | 6 | 2.3 | 57 % | 20 % | 17 % | 0.24 gen |
| 10 | 10 | 7.8 | 39 % | 7 % | 51 % | 0.28 gen |
| 40 | 15 | 67 | 18 % | 2 % | 79 % | 0.58 gen |
| 100 | 17 | 292 | 13 % | 1 % | 85 % | 1.21 gen |
| 200 | 19 | 1381 | 10 % | 0.5 % | 90 % | 2.11 gen |
| 10, `MATRIX` | 10 | 4.3 | 76 % | 14 % | 7 % | – |
| 100, `MATRIX` | 17 | 53 | 74 % | 6 % | 19 % | – |
| 10, `SEPARABLE` | 10 | 3.8 | 78 % | 16 % | 2 % | – |
| 100, `SEPARABLE` | 17 | 31 | 87 % | 9 % | 2 % | – |

* **Heap allocations: 13–17 `malloc` calls per evaluation** (interposed malloc counter: d=2 17.0,
  d=10 14.6, d=40 13.4), i.e. ~100–250 per generation. A heap `Vector` alloc+free costs 15–41 ns;
  drawing one coordinate (mt19937 uniform + `ppf`) costs 10–25 ns.
* **Eigendecomposition** (`SelfAdjointEigenSolver`) dominates the default configuration from d≈10,
  because `lazy_update_interval = 1/(c1+cmu)/d/10` is < 1 generation up to d≈80 (same rule as pycma).
  Changing that is methodological (out of scope, see D5). **A LAPACK backend does not help**: Eigen vs
  Accelerate `dsyevd` per call: d=10 4.0 vs 4.8 µs, d=40 66 vs 121 µs, d=100 562 vs 623 µs, d=200
  4011 vs 2469 µs. Only beyond d≈200 would it pay, at the cost of a BLAS/LAPACK dependency in every
  wheel. **Not doing.**
* For the O(d²) variants and small d, the cost is the per-sample path: allocations, virtual calls
  returning `Vector` by value, `std::function`, `ppf`, then `Population::sort` (7 gathers with `.eval()`).
* **Python boundary** (pybind11, current wheel), µs per evaluation: d=2: C++ objective 0.45, Python
  lambda 0.89, `ioh` problem 0.69; d=10: 0.83 / 1.26 / 1.23; at d=100 the boundary is negligible
  next to eigh. So the crossing is ~0.3–0.45 µs per evaluation, i.e. up to half the run time at
  small d.

Per-sample allocation sites in `Strategy::mutate` (`src/mutation.cpp:65`): every layer of the sampler
chain returns a `Vector` (`Uniform` → `GaussianTransformer` → [`Orthogonal`] → [`Mirrored`] →
[`CachedSampler`]), `tc->scale` returns a copy even for `NoThresholdConvergence`, `compute_y` returns a
`Vector`, `is_out_of_bounds` materialises a `Mask` (bounds are on by default: ±5),
`coordinate_mapping->transform` returns a `Vector` (also with no integer variables), and
`objective(p.pop.X_transformed.col(i))` copies the column into a temporary `Vector` because
`FunctionType` takes `const Vector&`. Per generation, in addition: `old_pop = pop` (`src/parameters.cpp:101`,
a copy of five d×λ matrices), `Population::sort` (7 allocations + an index vector), `resize_cols`
(`conservativeResize`), elitism's `hstack`, `update_best` from a `Block`.

## 1. Bugs found during the review

None of these were asked for, and all are outside the three directions. They are listed because
several change results, and golden fixtures (T0.1) should be generated after the decision on them (D6).

| # | Where | What |
|---|---|---|
| B1 | `include/mapping.hpp:70` | `transformations[i] = iround` uses the loop index, not `integer_idx[i]`: `integer_variables=[3]` rounds coordinate 0 and leaves 3 continuous (verified). `tests/test_c_integer.py` only tests all-integer problems. |
| B2 | `src/matrix_adaptation.cpp:102` | Active CMA: the loop that rescales negative weights by `d / ‖C^{-1/2} y‖²` runs over `[mu, λ−mu)`, which is empty for the default µ=λ/2. Looks like it should be `[mu, λ)`. |
| B3 | `src/matrix_adaptation.cpp:546` | `NaturalGradientAdaptation::invert_y` never resets `outdated_A_inv`: an O(d³) pseudo-inverse on every call (each elitist parent, each bound correction). |
| B4 | `include/matrix_adaptation.hpp:18,211`, `src/repelling.cpp:224,365` | `Adaptation::M` is never updated (`MatrixAdaptation` shadows it with its own `M`), so taboo points copy an identity matrix: repelling distances are Euclidean, computed as an O(d²) quadratic form, and each taboo point stores a d×d identity. |
| B5 | `include/common.hpp:286` | `rng::uniform` is the closed interval [0,1] with 32-bit resolution: `ppf(0) = −∞`/`ppf(1) = +∞`, probability 2·2⁻³² per coordinate. A large benchmark campaign draws enough coordinates to hit it. |
| B6 | `src/center_placement.cpp:14,30,42` | `UNIFORM`, `MAXIMIN_TABOO` and `NOVELTY_WEIGHTED` use `Vector::Random`/`std::rand`, which `rng::set_seed` does not seed: those runs are not reproducible. |
| B7 | `include/common.hpp:320`, `include/sampling.hpp:289` | `rng::normal` and `GaussianTransformer::box_muller` keep their state in function statics, shared by all instances (two optimizers in one process interfere; not thread-safe). |
| B8 | `setup.py:34` | Linux wheels are built with `-march=native -mtune=native` on the CI runner. They can raise SIGILL on CPUs without the runner's ISA extensions. |
| B9 | `.github/workflows/python-publish.yml:67-74` | The `cleanup` job deletes the release *and its tag* when a wheel build fails (IOHexperimenter lost v0.3.23 this way). Unit-test CI uses `actions/*@v2` and Python 3.8; publishing uses cibuildwheel 2.16.5, which predates CPython 3.13/3.14. |
| B10 | `src/restart_criteria.cpp:251-256` | `Stagnation::update` copies and `nth_element`s the whole fitness history 4× every generation: O(t) per generation, O(T²) per run. Only noticeable in long runs. **Fixed in WP4** (needed for 10⁷·d budgets, where BIPOP's λ=2 runs last 10⁵–10⁶ generations): running medians (`SplitMedian`), O(log t) per generation, bit-exact. A 2M-evaluation BIPOP run (MATRIX, d=40, f24): 33.1 s → 6.3 s. |
| B14 | `src/restart_criteria.cpp` | `COVARIANCE_NO_EIGV` derives from `CovarianceAdaptation` but never computes `B` and `d`: the criteria that read them (MaxDSigma, MinDSigma, ConditionC, NoEffectAxis) see the initial identity and ones. No effect with the default criteria (only MinDSigma/MaxDSigma are created for NO_EIGV, and ones give their default 1.0); wrong when ConditionC/NoEffectAxis are added by hand. **Fixed in WP4**: `has_eigendecomposition()`; regression test. |
| B15 | `src/restart_criteria.cpp` (MaxDSigma, MinDSigma, NoEffectAxis) | `d` holds √λ (pycma's D), and these criteria took `sqrt(d)` again (λ^¼), copied from the Python version (`modcma/parameters.py:840,844`). **Fixed in WP4, as pycma** (decided 2026-11-23): tolupsigma uses max(D), NoEffectAxis D[t], MinDSigma min(D). No golden run reaches these thresholds (unchanged); regression tests. |

## 2. Decisions for sign-off

**Decided (2026-11-23):**
* D1: order WP0 → WP1 (nanobind) → WP2 (memory) → WP3 (ask-tell) → WP4.
* D2: changes that alter the summation order are allowed. They come in separate PRs after the
  bit-exact work and are validated statistically (T0.1 tier 2).
* D3: a generator per optimizer, keeping today's default stream for `set_seed(s)` + one optimizer. The
  faster generator is an opt-in setting only.
* D6: B1–B3 and B5–B9 are fixed in WP0 (T0.3), before the golden fixtures are generated.
* B4: repelling should use the real metric (the current `C`, or `M` for MA-ES, at archive time),
  not the identity. This changes repelling results, so it is fixed in T0.3 too, before the fixtures.
* D4: every module works in ask-tell, bit-exact with `run()`. `tell(f, X=None)` accepts
  modified/injected solutions already in v1 (T3.5 is part of v1). Out-of-order/partial tell and async
  evaluation are not in v1.
* D7: as IOHexperimenter 0.4: Python ≥ 3.10, numpy ≥ 2.0; wheels for Linux x86_64/aarch64, Windows
  x64/ARM64, macOS arm64/x86_64; one abi3 wheel for CPython 3.12+; free-threaded wheels after T4.1.
* D8: one release at the end. PyPI stays at 1.2.0 until all WPs are merged; test wheels come from
  manual workflow runs.
* D5 stays a flag only (no task).

The original options, for reference:

* **D1: order of the WPs.** Recommended: WP0 → WP1 (nanobind + build/CI) → WP2 (memory/in-place) →
  WP3 (ask-tell) → WP4 (remaining compute items). Reasons: WP0 guards everything after it. With
  nanobind first, the ask-tell Python API (zero-copy views) is designed once, not bound twice. The
  preallocated population buffers of WP2 are exactly what `ask()` hands out. Alternative, if ask-tell
  is the most urgent: WP0 → WP2 → WP3 → WP1 → WP4 (ask-tell is then bound in pybind11 first and
  ported later).
* **D2: bitwise reproducibility.** Some speed-ups change the floating-point summation order (one GEMM
  `Y = A·Z` instead of λ GEMVs, `rankUpdate` instead of `Y·W·Yᵀ` + symmetrisation, not forming
  `inv_root_C`). After them, trajectories differ in the last bits and then diverge chaotically.
  Recommended: allow them, but in separate PRs after the bit-exact ones, and validated statistically
  (T0.1, tier 2) instead of bitwise.
* **D3: random numbers.** (a) Generator state per optimizer instead of the global `rng::GENERATOR` and
  the statics of B7. This is needed for free-threading and for two optimizers in one process.
  Recommended. The default stream for `set_seed(s)` + one optimizer can stay identical. (b) A faster
  generator/normal sampler (e.g. xoshiro256++ with ziggurat, instead of mt19937 32-bit + `ppf`) would
  cut most of the sampling time that dominates `MATRIX`/`SEPARABLE`. It changes every random stream,
  so old results are no longer reproducible bit-for-bit with new versions. Recommended: an opt-in
  setting, not the default.
* **D4: ask-tell scope.** Recommended: *every* module works and gives exactly the same trajectory as
  `run()`, with a variable batch size (see WP3). Not in v1: out-of-order or partial `tell`, asynchronous
  evaluation. Open: should `tell` accept user-modified/injected `X` (pycma-style)? It is feasible
  because elitism and bound correction already map x back to y and z.
* **D5: lazy eigendecomposition interval.** Methodological, so out of scope as you asked, but it is
  51–90 % of the run time for d ≥ 10. Only flagged here; no task.
* **D6: bugs B1–B10.** Recommended: fix B1–B3 and B5–B9 in small PRs at the start of WP0, then
  generate the golden fixtures. B4 needs a decision: the intended repelling metric (Euclidean, or the
  current `C`/`M`).
* **D7: floors.** Python ≥ 3.10 and numpy ≥ 2.0, as IOHexperimenter 0.4? Wheel matrix as IOH (Linux
  x86_64/aarch64, Windows x64/ARM64, macOS arm64/x86_64; abi3 for 3.12+, free-threaded wheels once
  D3a is done)?
* **D8: releases.** One release at the end (as IOH), or a release after each WP?
* **D9: the pure-Python implementation (signed off).** Moved to the branch `legacy-python` (it stays
  installable as `pip install "modcma<2"`); master keeps only the C++ version. `import modcma`
  exposes its API directly (`modcma.fmin`, `modcma.ModularCMAES`, ...), `modcma.c_maes` stays as an
  alias; the release is 2.0.0. Constructing `ModularCMAES` the 1.x way raises an error that points to
  the migration guide (WP6).
* **D10: documentation (signed off; since 2026-11-24 Zensical instead of MkDocs Material, whose
  end of life is announced).** MkDocs Material + mkdocstrings (reads the `.pyi` stubs, no
  build needed) on Read the Docs, the same URL. The docstrings are written once, in the bindings, and
  reach `help()`, IDEs, the stubs and the reference from there (WP7). WP6 and WP7 are two PRs,
  stacked on WP5.

## 3. Invariants

* `run()`/`step()` results for a fixed seed stay bitwise identical through WP1, WP2 and WP3, except
  where D2/D3 explicitly allow otherwise. That PR then says so and shows the statistical check.
* Never regenerate golden fixtures to make a test pass. Regenerating needs maintainer approval plus a
  written reason in the PR.
* Each performance PR reports before/after from the committed harness (T0.2): min of ≥ 5 repetitions,
  µs/generation and malloc/evaluation, for d ∈ {2, 10, 40, 100} and the default, `MATRIX` and
  `SEPARABLE` configurations.
* Python API stays source-compatible unless a task lists the change. API changes go in a changelog
  draft.

---

## WP0: Safety net (prerequisite for everything): implemented on branch `wp0-safety-net`, not pushed

Status (2026-11-23):

* Done: T0.2 (`modcma_bench`, `bench/bench_python.py`), T0.1 (`tests/golden.py`, `tests/test_c_golden.py`,
  fixtures for darwin-arm64 and linux-x86_64 from a0b3167), T0.3: B1, B2 (+ follow-up: the rescaling
  uses `invert_y`, since `COVARIANCE_NO_EIGV` never computes `inv_root_C`), B3, B4 (as an option:
  `Modules.repelling_metric` = EUCLIDEAN (default, unchanged) / CURRENT / ARCHIVED with unit determinant),
  B5, B7, B8, B9, plus new bugs:
  * B11: a non-finite matrix counted as a successful adaptation, so runs evaluated NaN until the budget
    was used up (natural gradient). Now `UnableToAdapt` restarts.
  * B12: `adapt_evolution_paths` had `(lambda, mu)` in C++ and `(mu, lamb)` in Python. Now `(mu, lambda)`
    everywhere internally; maintainer preference: **always (mu, lambda)**.
  * B13: sequential selection kept unevaluated columns (stale or uninitialized) in the population. Now
    `Population::n` holds the active columns (separate from the capacity, no reallocation); with
    PAIRWISE, stops only after complete pairs; `Weights::for_population(n)` keeps the total negative
    weight. Python's `Population.X/f/...` are views of the active columns.
  * Three C++ tests (`test_keeponly`: a view that dangles after a reallocation; the Halton/Sobol sums
    depend on the standard library) are fixed.
* B6 was wrong in the review: `set_seed` also calls `srand`, so center placement is reproducible on one
  platform. Moving it to `rng::GENERATOR` (it changes results) is now part of T4.1.
* Open for WP1:
  * The `Settings` constructor takes `(..., lambda0, mu0, ...)`. Switching to (mu, lambda) positionally
    would silently swap two ints, so make the arguments after `dim` keyword-only in the nanobind port.
  * The `.pyi` stubs are stale (no `integer_variables`, `X_transformed`).
  * The tests of the stale Python implementation fail on Python 3.14 (31 failures, unrelated to the C++
    version). Decide whether CI skips them on ≥ 3.14.
  * The first CI run checks the Docker-generated linux-x86_64 fixture against GitHub's runners.
* Not doing (WP4): a LAPACK eigensolver. Revisit only for users at d ≥ 500 (opt-in via scipy's LAPACK).

## WP0: Safety net (prerequisite for everything)

### T0.1 Golden trajectory tests
* Tier 1 (bitwise): for a fixed seed, a grid over every module option (one-at-a-time from the default
  plus a pairwise covering set), d ∈ {2, 5, 10}, sphere/ellipse/rastrigin in C++: record per
  generation `f` (sorted), `m`, `sigma`, `evaluations`, restart events. Compare with `==` (or ≤1 ulp).
  Run from Python (`tests/test_c_golden.py`) and generate with a script, as in IOH.
* Tier 2 (statistical, for D2/D3 PRs): a small BBOB subset × seeds; compare final-target-hit
  distributions or ERTs with a pre-registered tolerance.
* Include configurations that hit every evaluation site: TPA, sequential selection, all restart
  strategies, repelling (hill-valley), elitism, bound corrections incl. `RESAMPLE`, integer variables.

### T0.2 Benchmark harness
* Commit the C++ harness (per-phase timing + malloc counter) as a CMake target, plus a Python script
  for the boundary costs. It is the reference for every number in §0.

### T0.3 Bug fixes (per D6)
* One small PR per fix with a regression test (B1: non-contiguous integer indices; B2: the active
  update with λ = 2µ; B3: count of pseudo-inverse calls; B5: no infinite samples at the extremes; B6:
  seeded center placement is reproducible).

## WP1: Bindings to nanobind, build & CI (as IOHexperimenter WP1/WP5/WP6): implemented on branch `wp1-nanobind`

Status (2026-11-23), on top of WP0:

* T1.1 done: `tests/api_dump.py` (API dump of the pybind11 bindings in `tests/data/api/pybind11.json`,
  `compare` against the build) and `tests/test_c_api_semantics.py`.
* T1.2 done: scikit-build-core + nanobind 3.1.0 (pinned), `NOMINSIZE`, `STABLE_ABI` (one cp312-abi3
  wheel); `setup.py`, `MANIFEST.in`, `src/interface.cpp` removed. Python ≥ 3.10, numpy ≥ 2.0.
* T1.3 done: `bindings/*.cpp`. API compare: 0 incompatibilities. Golden trajectories bit-identical to
  the pybind11 build. Deliberate changes, in the commit message for the changelog: `Settings` arguments
  after `dim, modules` are keyword-only (mu0 before lambda0), Eigen members are writable views,
  Python criteria are kept alive by the vector, `i8_sobol` callable, stricter scalar/array arguments
  of low-level methods (lists still accepted where users pass them: `Settings`, `CachedSampler`),
  `CriteriaVector` methods positional-only.
* T1.4: tests on ubuntu/macos/windows × 3.10–3.14 (3.14: only the C++-version tests), a C++ job
  (ctest, harness), wheels via cibuildwheel 4.3.3 on six runners (upload only on a published
  release). Stubs: `tools/gen_stubs.py` (nanobind.stubgen).
* T1.5 measured (this Mac, Python 3.14, min of 7; pybind11 → nanobind):
  * per evaluation, Python lambda objective: d=2 0.65 → 0.77 µs, d=10 1.06 → 1.19 µs, d=100 15.8 → 16.1 µs;
    ioh problem: 0.69 → 0.79, 1.20 → 1.32, 17.7 → 18.0 µs. **Slower by ~0.12 µs per evaluation**: nanobind
    exports each numpy array through a Python call (`numpy.asarray`); the same holds for any vector
    returned to Python (`Gaussian()`: 0.24 → 0.36 µs). A numpy-C-API fast path is incompatible with the
    stable ABI; WP3's batch evaluation (T3.4) removes the per-evaluation crossing instead.
  * attribute access: `cma.p.stats.global_best.y` 0.51 → 0.15 µs, `cma.p.mutation.sigma` 0.30 → 0.12,
    `cma.p.pop.X` 0.47 → 0.26, `cma.p.adaptation.C` 0.45 → 0.27.
  * editable build 38.6 s → 15.5 s wall; extension 3.7 MB → 1.3 MB; import time unchanged (268 ms, the
    package's own imports).

### T1.1 Pin the current Python API first
* API dump of `modcma.c_maes.cmaescpp` (modules, classes, members, signatures, defaults), compared
  automatically after the port.
* Pin the semantics that ports silently change: copy vs view of Eigen members (`p.pop.X`,
  `adaptation.C`, `def_readwrite` writes), enums (pybind11 `native_enum` is a Python `enum.Enum`;
  `modcma/c_maes/__init__.py` relies on that for ConfigSpace), `Settings`'s keyword arguments,
  `std::optional` arguments, `shared_ptr` identity (`cma.p` is the same object across calls).

### T1.2 Build system
* scikit-build-core + CMake, nanobind pinned (3.1.0 as IOH), `nanobind_add_module(... NOMINSIZE)`. IOH
  measured that nanobind's default `-Os` costs 1.4× on compute.
* Portable flags for wheels (no `-march=native`, fixes B8); keep `-O3`, `-fno-math-errno`. Native
  flags only as an opt-in for local builds. Drop the `CC=g++` override.
* One CMake project for the extension, the `main` executable and the benchmark harness (T0.2).
  `setup.py` goes; `pyproject.toml` declares the metadata (version 1.2.0 is in `setup.py` today).

### T1.3 Port `src/interface.cpp`
* 1354 lines in one TU today. Split it per submodule (`parameters`, `sampling`, `matrix_adaptation`, …)
  for parallel builds, unity-safe (no file-scope `using namespace`).
* Holders: `std::shared_ptr` everywhere (`nanobind/stl/shared_ptr.h`); `std::function`, `std::optional`
  and Eigen casters (`nanobind/eigen/dense.h`). There are no Python-subclassable C++ classes today
  (no trampolines), only `PySampler`, which wraps a `std::function`.
* The objective: a Python callable is called with a **read-only numpy view** of the column, not a
  copy. This needs WP2's `Eigen::Ref` signature (T2.6). Until then a copy, as now.
* `ndarray` overloads registered before the `std::vector`/list ones (IOH T6.3: otherwise nanobind
  converts on every call).

### T1.4 Stubs, CI, wheels
* `nanobind.stubgen` replaces the `.pyi` files in `modcma/c_maes/cmaescpp/`.
* Replace both workflows (B9): tests on all three OSes for 3.10–3.14, cibuildwheel ≥ 3 with the D7
  matrix, abi3 wheel for 3.12+, no release-deleting cleanup job, upload only on `release: published`.
* Free-threaded wheels only after D3a (global RNG and statics). Until then the module declares itself
  not free-threading-safe.

### T1.5 Measure
* Per-evaluation overhead (Python lambda, `ioh` problem), attribute access (`cma.p.stats.global_best.y`,
  `cma.p.pop.X`), clean build time, wheel size, `import` time; pybind11 vs nanobind, in the PR.

## WP2: Memory: preallocated, in-place population and allocation-free sampling: bit-exact part on branch `wp2-memory`

Status (2026-11-23), on top of WP1, every step bit-exact (254/254 golden runs on darwin-arm64;
ASan/UBSan clean):

* T2.1: `Population::n` = active columns, separate from the capacity (done with B13 in WP0);
  `resize_cols` only changes `n` (no reallocation).
* T2.2: `Parameters::start` swaps `pop` and `old_pop` (no copy).
* T2.3: `Population::sort` with a kept order vector (`std::sort` + index tie-break = the stable order,
  NaN last) and a kept scratch buffer, gathering column by column.
* T2.4/T2.5: `Sampler::sample(Eigen::Ref<Vector>)`, in-place transformers, `scale_inplace`,
  `compute_y_into`, `invert_x_into`/`invert_y_into`, bound check without a mask, repelling and the
  integer mapping by reference.
* T2.6: `FunctionType = std::function<Float(Eigen::Ref<const Vector>)>`; Python objectives still
  get an array of their own (a dedicated nanobind caster).
* T2.7 (partly): recombination, evolution paths, the natural gradient's G, the covariance update's
  eigensolver and buffers. Kept (T2.8): the temporaries of `C = decay C + c1 pc pcᵀ + cmu Y W Yᵀ`
  and of the `MATRIX`/`CHOLESKY`/`CMSA` updates (4–6 allocations per generation) and the natural
  gradient's matrix exponential.
* Lessons for bit-exactness (in the commit messages): Eigen's vectorized `log` differs from `std::log`
  in the last bit (an unaligned destination mixes them: Logistic evaluates into an aligned scratch);
  Eigen evaluates `dst = a + A x` as `dst = a; dst += A x`, and a product with an expression as
  left operand as a coefficient loop; the in-place code reproduces those orders.

modcma_bench, before WP2 → now (µs/gen, min of 5; malloc/evaluation 13–25 → 0.09–0.7):

| config | d=2 | d=10 | d=40 | d=100 |
|---|---|---|---|---|
| default | 2.98 → 0.91 | 8.12 → 5.00 | 64.4 → 57.8 | 265 → 256 |
| matrix | 2.21 → 0.53 | 4.65 → 1.75 | 15.4 → 10.1 | 48.9 → 38.3 |
| separable | 2.08 → 0.42 | 4.20 → 1.40 | 12.5 → 7.19 | 28.6 → 19.5 |
| active | 2.77 → 0.70 | 8.63 → 5.22 | 66.3 → 60.0 | 273 → 261 |
| elitist | 3.52 → 0.82 | 10.3 → 5.64 | 70.2 → 61.4 | 284 → 268 |

The default configuration at d ≥ 10 is bounded by the eigendecomposition (70–90 %, D5).

Goal: **0 heap allocations per generation** in steady state (λ unchanged), measured with the T0.2
counter. Everything here is bit-exact except T2.8.

### T2.1 Population with fixed capacity
* `Population` owns d × capacity buffers (capacity = λ + elitist k), with `n` active columns.
  `resize_cols`/`+=` only change `n`. Grow capacity only when λ grows (IPOP/BIPOP restart), never
  shrink. Drop `X_transformed` when there are no integer variables (alias `X`).
* Revisit `resize_population` and `perform_restart`, which construct new populations today.

### T2.2 No copy of the population per generation
* `old_pop = pop` in `Parameters::start` → double buffering with `std::swap` (O(1)). Check every reader
  of `old_pop`: elitism (top k), MSR/PSR (`old_pop.f`), TPA.

### T2.3 Selection without seven gathers
* Sort an index array into a reused buffer, then permute in place (cycle-following on the buffers, or
  a gather into the double buffer). Elitism (`Elitsm::operator()`) writes the k parents into the
  reserved columns instead of `hstack`.

### T2.4 Sampler API without return-by-value
* `Sampler::operator()()` → `void sample(Eigen::Ref<Vector> out)`, plus `sample(Eigen::Ref<Matrix>)`
  for a batch. Transformers work in place. `Orthogonal` fills its batch directly. `CachedSampler`
  copies into `out`.
* Python-facing `__call__` of the samplers keeps returning a new array (a thin wrapper).

### T2.5 The per-sample path
* `ThresholdConvergence::scale` in place (and skipped for `NoThresholdConvergence`). `compute_y` /
  `invert_y` / `invert_x` write into an `Eigen::Ref` out-argument. `is_out_of_bounds` without
  materialising a `Mask` (`.any()` on the expression first). Bound correction in place.
* `CoordinateMapping::transform`: no-op without integer variables; otherwise a loop over the integer
  indices only (fixes B1 along the way if not fixed in WP0). No `std::function` per coordinate.

### T2.6 Objective signature
* `FunctionType = std::function<Float(Eigen::Ref<const Vector>)>`: no copy of the column. This is a
  C++ API change (documented); `main.cpp` and the tests follow.
* `Stats::update_best` takes `Eigen::Ref<const Vector>`; it copies only on improvement.

### T2.7 Small per-generation items
* Restart criteria: `dynamic_pointer_cast` per criterion per generation → resolved once at
  construction/reset. `Stagnation` (B10): keep the medians incrementally or bound the history,
  bit-exact.

### T2.8 (D2) Batched linear algebra, not bit-exact

**Done on branch `wp4-compute`** (PR stacked on WP3), validated statistically (below, WP4 status).
* `Y = A·Z` as one GEMM for the whole batch, then `X = m + S∘Y`. Only columns rejected by `RESAMPLE` or
  repelling are redrawn per column.
* `CovarianceAdaptation`: don't form `inv_root_C` (an O(d³) GEMM per eigendecomposition). Compute `ps`
  as `B·(D⁻¹·(Bᵀ·dm))` and the active-weight norms via `invert_y`. Rank-µ update as a lower-triangular
  `rankUpdate`/`selfadjointView`, which drops the `0.5(C + Cᵀ)` pass. Same for `SelfAdaptation`.
* `NaturalGradientAdaptation::compute_gradients`: `G = Z·diag(w)·Zᵀ − Σw·I` in one GEMM instead of λ
  outer products with d×d temporaries.

## WP3: Ask-tell interface (C++ first, then Python): implemented on branch `wp3-ask-tell`

Status (2026-11-23), on top of WP2 (PR #61 has WP0–WP2; WP3 is a PR stacked on it):
* **T3.1/T3.2 done.** An explicit state machine in `ModularCMAES` (phases: restart center, restart
  hill-valley point, population batch, extra points). `step()`, `run()` and `mutate()` are loops
  over it; `tell()` completes a generation with select, recombine, adapt. The resumable pieces:
  `mutation::Strategy` (begin_population / sample / evaluated / end_population, TPA's two points
  as `extra_points`, SA's mean sigma as a hook), `Parameters` (begin_generation / begin_restart /
  tell_restart_center / finish_restart) and `Repelling` (begin_update / query_point / answer, with
  hooks for coverage and adaptive repelling). The objective-taking functions (`perform_restart`,
  `start`, `update_archive`, `Strategy::mutate`) remain, as loops over the same pieces.
  Batches, as designed: λ (+2 for TPA); with sequential selection up to `first_stop` (the first
  individual after which it can stop; with PAIRWISE the odd one), then one point or one pair; the
  TPA pair on its own after a stopped population; restart center and hill-valley points one at a
  time. Threshold convergence gets `evaluations + j` for the j-th sample of a batch.
* **Golden:** 1016/1016 bit-exact on darwin-arm64: the 254 runs with four drivers (`step()`,
  ask/tell from Python, ask/tell passing the points back, vectorized objective); the golden test
  runs all four (`python -m tests.golden check --drivers all`). `tests/test_c_ask_tell.py` adds
  configurations the golden runs do not reach (pairwise sequential stops, TPA after a stopped
  population, restarts with both repelling types, integer + RESAMPLE, BIPOP), batch sizes, errors,
  injection. The C++ test checks ask/tell against the classic loop; ASan/UBSan clean.
* **T3.3 done:** `ask() -> (k, d)` (a copy), `tell(f, X=None) -> bool` (any array-like),
  `has_pending`; `tell` without `ask` → RuntimeError, a wrong length/shape → ValueError (nothing
  consumed). Decided: methods on `ModularCMAES`, no separate wrapper class.
* **T3.4 done:** `run/step/__call__(objective, *, vectorized=False)`. Per evaluation (SEPARABLE,
  sphere): d=2 0.75 → 0.29 µs, d=10 0.82 → 0.27 µs, d=10 λ=100 0.79 → 0.16 µs. The C++ benchmark is
  unchanged (the state machine costs nothing measurable).
* **T3.5 done:** a row of `X` that differs from the asked point replaces it: `X` and
  `X_transformed` are set to it (no integer rounding: it is the point that was evaluated), `y` and
  `z` derived with the individual's own sigma (`S` unchanged), as elitism does. For TPA, center and
  hill-valley points the given point is what is recorded (best solution, centers). Unchanged rows
  are left alone, so passing the asked points back is bit-exact.
* **T3.6 done:** README section with a `ProcessPoolExecutor` example and the vectorized form.
* Also: the C++ `Settings` constructor takes (mu, lambda) now (the bindings were already
  keyword-only); the verbose restart message prints (mu, lambda, sigma) in that order.
* Open: delete `origin/ask-tell` and `origin/inplace-sampling` after merge (your call).


### Where the algorithm evaluates the objective today
1. `Strategy::mutate`: λ samples, one at a time. **Sequential selection** stops early depending on
   `f(i)`. **Threshold convergence** reads `stats.evaluations`, which increments inside the loop.
2. `TPA::mutate`: 2 extra points (`m ± σ·dm`) after the population. They do not depend on its f.
3. `Parameters::perform_restart`: 1 evaluation of the center `m`.
4. Repelling `update_archive` → `hill_valley_test_p`: up to 10 points per taboo point, sequentially,
   with an early exit.
5. `es.hpp` (`OnePlusOneES`, `MuCommaLambdaES`): separate classes. Out of scope (follow-up).

### Design (recommended: explicit state machine, one code path)
* `ask()` returns the **largest batch that can be evaluated without changing the algorithm**:
  * normal generation: λ (+2 for TPA) columns;
  * sequential selection: first the points up to the cut-off (`µ·seq_cutoff_factor`, minding the
    pairwise parity rule), then one point per `ask()`;
  * restart: the center (1 point), then the hill-valley points one at a time.
  Threshold convergence uses `evaluations + i` for sample i, which is bit-exact.
* `tell(f)` takes exactly the values of the pending batch, in order, and advances the state machine
  until the next batch is needed (select → recombine → adapt → restart decision → sample).
  `ask()` twice without `tell` returns the same pending batch.
* `run(objective)` and `step(objective)` become loops over `ask`/`tell`, so there is a single code
  path, and T0.1 tier 1 proves that `run()` did not change.
* The batch is a view of the population buffer (WP2): Eigen's column-major d×k is a C-contiguous k×d
  array, so Python gets a zero-copy, read-only `(k, d)` numpy array.
* Needed refactors: split "decide to restart" from "perform restart" (the old `ask-tell` branch did
  this: `adapt()` returns `should_restart`, `finalize_restart` + `perform_restart`); split
  `Strategy::mutate` into sample/consume; TPA's two points become part of the batch; hill-valley
  becomes a resumable query.
* Rejected: running the optimizer on a second thread that blocks inside the objective (GIL,
  determinism, free-threading). C++20 coroutines would keep today's control flow but need C++20 and
  turn every evaluation site into a coroutine. An option if you'd rather move to C++20.

### Tasks
* **T3.1** Restart split + state machine in C++, `run`/`step` on top of it; T0.1 unchanged.
* **T3.2** Sequential selection, TPA, threshold convergence, restart center, hill-valley in ask-tell;
  a test per module: ask-tell with an external objective == `run()`, bitwise, for every T0.1 config.
* **T3.3** Python API on the C++ class: `ask() -> ndarray (k, d)`, `tell(f: ndarray (k,))`, `stop()` /
  `break_conditions()`, errors for a wrong length and for `tell` before `ask`. A thin Python
  convenience wrapper (`modcma.c_maes.AskTellCMAES` or methods on `ModularCMAES`, to decide in review).
  The old `modcma/asktellcmaes.py` stays as it is.
* **T3.4** Vectorised objective: `run(objective, vectorized=True)` calls the objective once per batch
  with `(k, d)`. This removes the per-evaluation Python crossing (§0: up to half the time at small d).
* **T3.5** (D4, v1): `tell(f, X=None)`: when `X` is given, the columns replace the asked solutions
  (modified or injected) and Y, Z are recomputed via `invert_x`/`invert_y`, as elitism and bound
  correction already do. To define in the PR: how S (per-sample sigma) is set for injected columns,
  and whether the integer mapping applies to `X` or `X_transformed`.
* **T3.6** Docs and an example (parallel evaluation with `concurrent.futures`, an ioh problem).
* The old `origin/ask-tell` branch: do not merge. It is 163 commits behind master, the restart code was
  split into `restart_criteria`/`restart_strategy` since, and its design (Python queue, NaN
  placeholders filled by `mutate`) does not handle sequential selection, TPA or restarts. Reuse only
  the restart split and the test ideas. `origin/inplace-sampling` has no commits of its own (it points
  at master of Dec 2025). Delete both branches after WP2/WP3?

## WP4: Remaining compute items (after WP2/WP3; each measured, each optional): branch `wp4-compute`

Status (2026-11-23), on top of WP3, one PR with T2.8 (third PR of the stack):
* **T4.1 done.** `Parameters::generator` (mt19937) for every module of an optimizer: sampler, sigma
  sampler, COTN/uniform-resample bound correction, BIPOP, center placement. Default: `rng::fork()`
  copies the global generator at creation and moves the global one on, so `set_seed(s)` + one
  optimizer draws exactly the old stream (golden bit-exact except B6), and two optimizers after
  one `set_seed` differ. `Settings(seed=...)`: an optimizer's own seed. Note: `set_seed` *after*
  creating an optimizer no longer affects it. **B6 fixed** here: center placement drew from
  `std::rand`; it changes the 17 golden runs with UNIFORM/MAXIMIN_TABOO/NOVELTY_WEIGHTED placement
  after a restart. Built with nanobind `FREE_THREADED`: on CPython 3.14t the GIL stays disabled;
  cp314t wheels; CI jobs on 3.14t (Linux, macOS). 8 threads x 20 runs: identical results per
  thread, ~2.3x the throughput of one thread. Still global: `constants` (set them before starting
  threads) and the generator of objects created outside an optimizer (ES baselines, a sampler on
  its own).
* **T4.2 done.** `Settings(fast_sampling=True)`: `sampling.FastGaussian`, xoshiro256++ and the
  256-layer ziggurat (numpy's 64-bit variant) instead of mt19937 + inverse CDF, for the default
  UNIFORM + GAUSSIAN sampler. us/generation: SEPARABLE d=10 1.43 -> 0.89, d=100 19.9 -> 10.2;
  NONE similar; MATRIX d=100 38.7 -> 29.0; default (eigendecomposition-bound) ~4 %. Checked:
  moments, a KS bound, the tail; two golden runs pin its stream.
* **T4.3 measured, no change.** Repelling with 10 taboo points (RESTART, Rastrigin): at most 1-3 %
  of the generation time for every metric at d=10 and d=40 (the criticality filter skips most
  points). One d x d matrix per taboo point is 80 kB at d=100. Not worth a factorization.
* **T4.4 measured, not doing.** 5 virtual calls per sample cost ~3.8 ns (M-series), against
  ~70 ns per sample at d=2 and more at larger d: <= 5 %, < 2 % from d=10.
* **T2.8 done** (see its section): batched `Y = A Z`, triangular products for Cholesky factors,
  COVARIANCE without forming `C^-1/2` (`inv_root_C` became a computed read-only property) and with
  symmetric rank updates (also CMSA and the natural-gradient G), MA-ES in place. us/generation,
  same session: default d=40 58 -> 54, d=100 255 -> 227; active d=100 262 -> 235; CHOLESKY d=100
  45.9 -> 40.4; MATRIX d=10 1.79 -> 1.61; mallocs/evaluation default 0.44 -> 0.15, MATRIX 0.45 ->
  0.05.
* **Tier 2 (D2):** `tests/statistical.py`, 7 BBOB functions x 14 configurations x d in {5, 20} x
  25 seeds, Mann-Whitney U per cell (evaluations and final precision), pre-registered: no cell
  below the Bonferroni threshold, at most 10 % raw p < 0.05. WP3 head (2cf6646) vs T2.8: 0 cells
  after Bonferroni (392 tests), 2 raw p < 0.05 (0.5 %), median evaluations new/old geometric mean
  1.004 over 131 solved cells: **met**.
* Golden fixtures regenerated from 4892d8e for macOS, gcc 13 and the CI runner (gcc 15).
* **docs/performance.md** (`scripts/performance/`): modcma against pycma, cmaes (Optuna), evosax (JAX)
  and EvoTorch on the ellipsoid, d = 2..1000, one thread, Apple M5; every matrix adaptation;
  vectorized vs per-point objectives; threads on free-threaded 3.14t (10 threads: 4.3x). Up to
  d = 10 modcma is 20-45x faster than pycma; above d ~ 200 the default is slower than pycma
  (Eigen's eigensolver vs LAPACK dsyevd) -> WP5.
* **xNES (`NATURAL_GRADIENT`)**: exact low-rank matrix exponential for A <- A expm(eta G)
  (G = Z diag(u) Z^T - alpha I: thin QR + lambda x lambda eigendecomposition), O(d^2 lambda)
  instead of O(d^3): d = 1000 224 ms -> 2.8 ms per generation. For lambda >= d or d < 16 the dense
  exponential, bit-identical to before. `G` is a computed read-only property.


* **T4.1** (D3a) Generator per optimizer instance, passed down instead of the global `rng::GENERATOR`.
  No function statics (B7). `set_seed` semantics kept for the single-optimizer case.
* **T4.2** (D3b) Opt-in fast normal sampler (`Settings`/`constants` flag); measure the share of
  sampling for `MATRIX`/`SEPARABLE`/`NONE` at d = 10, 100.
* **T4.3** Repelling after B4 (real metric): avoid the O(d²) quadratic form per check where possible
  (e.g. store each taboo point's factor and use `invert_y`-style solves, or a Cholesky factor), and
  measure the cost of one d×d matrix per taboo point.
* **T4.4** Devirtualisation where it is free: the module objects are `shared_ptr`s with virtual calls
  per sample (`sampler`, `tc`, `bounds`, `repelling`, `coordinate_mapping`). After WP2 the per-sample
  calls become per-batch calls, so measure first; this is likely not worth more than that.
* **Not doing** (with evidence in §0): a LAPACK/Accelerate eigensolver; PCH; changing
  `lazy_update_interval` (D5).


## WP5: LAPACK/BLAS for the linear algebra in high dimension: branch `wp5-lapack` (on WP4)

Eigen's dense kernels are portable but not the fastest: its eigensolver is unblocked, and the
wheels compile it without AVX (no `-march`), while LAPACK/BLAS choose their kernels at run time.
Where LAPACK/BLAS comes from, without a new build dependency for the wheels:
* macOS: Apple Accelerate, linked (`MODCMA_LAPACK`, on by default on Apple);
* Linux/Windows: scipy's (`scipy.linalg.cython_lapack` / `cython_blas` capsules, handed to the C++
  library at import of `modcma.c_maes`; scipy is a dependency) -> OpenBLAS, MKL in some conda builds;
* C++ programs: `MODCMA_LAPACK` links `find_package(LAPACK)`; without it Eigen only.
Results for d >= `lapack_min_dim` (120) are not bit-identical to Eigen's and depend on the LAPACK;
the golden runs (d <= 10) are unchanged. OpenBLAS's own threads: `OPENBLAS_NUM_THREADS=1` for
parallel runs (documented).

Status (2026-11-24):
* **T5.1 done.** Eigendecomposition with `dsyevd` from d = 120 (`constants.eigen_solver`, -1 auto,
  0 Eigen, 1 dsyevd, 2 dsyevr). One decomposition, d = 1000: Eigen 289 ms, Accelerate 50 ms,
  OpenBLAS 57 ms (one thread). COVARIANCE per generation: d=200 855 -> 575 us, d=500 5.1 -> 2.0 ms,
  d=1000 18.5 -> 5.4 ms; modcma is then faster than pycma, cmaes, evosax and EvoTorch at every d
  from 10 to 1000. pycma does no approximation in high d (full C, no diagonal mode); 75 % of its
  time is the same `dsyevd` (via numpy), the same number of decompositions as modcma.
* **T5.2 done.** Cholesky with `dpotrf` from d = 120 (`constants.cholesky_solver`) for CMSA (every
  generation) and COVARIANCE_NO_EIGV. One factorization, d = 1000: Eigen 7.4 ms, Accelerate 1.0,
  OpenBLAS 2.2. CMSA d=1000 9.1 -> 2.4 ms per generation; NO_EIGV 1.7 -> 1.0 ms.
* **T5.3 done.** Wheels: Accelerate on macOS (no size), scipy's LAPACK on Linux/Windows (no size, no
  build change); `build_info()["lapack"]`: linked / scipy / none. CI: tests on every platform
  (`tests/test_c_lapack.py`), a C++ build with LAPACK on Linux. docs/performance.md re-measured.
* **T5.4 done: BLAS `dgemm`** (linked, or scipy's `cython_blas`) from `constants.blas_min_dim` = 100
  (`constants.blas`: -1 auto, 0 Eigen, 1 always) for the batch `Y = A Z` (also the triangular
  factors: `dtrmm` of OpenBLAS is slower than Eigen up to d ~ 200, `dgemm` is not), the MA-ES
  update of M and M^-1, `A U` in xNES. One operation, d = 1000: `Y = A Z` Eigen 1.13 ms,
  Accelerate 0.095, OpenBLAS 0.104. The rank updates of C (COVARIANCE, CMSA, xNES) stay with
  Eigen's `rankUpdate` on the lower triangle: with lambda columns they are bound by the passes over
  S, and `dgemm` on all of S plus decay and mirroring was slower at every d (d = 1000: 1.0 vs
  0.60 ms; CMSA generation 2.4 vs 1.9 ms). That rank update caused the 6-32 % slowdown at d = 50;
  without it BLAS breaks even at d ~ 50-75 (the call itself costs 20-50 ns), threshold 100.
* **T5.5 done: LU inverse** (`dgetrf` + `dgetri`) for M (MA-ES) and A (xNES) from `blas_min_dim`,
  with the pseudo-inverse (complete orthogonal decomposition) as the fallback when singular; d =
  1000: 150 ms -> 6.5 ms (Accelerate) / 14.9 ms (OpenBLAS) whenever `invert_y` is needed.
* **Timing re-run** (all implementations, one session): per generation at d = 1000, before T5.4/T5.5
  -> after: COVARIANCE 5.4 -> 4.2 ms (Eigen only 18.4; pycma 7.1), MATRIX 2.3 -> 0.95, xNES 2.8 ->
  0.87, COVARIANCE_NO_EIGV 1.0 -> 0.55, CHOLESKY 2.6 -> 2.2, CMSA 2.4 -> 1.9; d <= 50 unchanged
  (modcma rows re-measured after the rank updates went back to Eigen; the other libraries are from
  the full run).
  Tier 2 at d = 40 (blas forced on there through the threshold at the time: 8 configurations x 7
  functions x 25 seeds) against 81cb955: no difference (1 of 112 raw p < 0.05, none after
  Bonferroni; median evaluations ratio 0.998).

---

## WP6: The pure-Python implementation to a branch (D9): branch `wp6-legacy` (on WP5)

**Status: done.** `legacy-python` (c24606e: v1.2.0's master plus a notice in the README) is pushed.
On master: the Python implementation, its 7 test files, `expected.json`, `create_expected.py`, the
Sphinx sources and `.readthedocs.yml` are gone (WP7 brings the new site); `test_c_mutation` no longer
cross-checks against `modcma.modularcmaes.scale_with_threshold` (its expected value is computed in the
test). `modcma` re-exports the API of `modcma.c_maes` (`__all__`, plus `fmin` and
`settings_from_dict`, also added to `c_maes.__all__`), `__version__` from the package metadata; a
`ModularCMAES.__init__(fitness_func, *args, **kwargs)` overload in the bindings raises the TypeError of
D9, and a module `__getattr__` names the removed 1.x names (`AskTellCMAES`, `evaluate_bbob`, ...).
`tests/test_c_package.py`. CI runs every test on every Python version. README: the Python-only
sections, the Python column of the module table and the Python examples per module are removed (the
rewrite is WP7). Version 2.0.0, ioh no longer a dependency.

The pure-Python implementation (`modcma/{modularcmaes,asktellcmaes,parameters,population,sampling,
utils,__main__}.py`) has not been developed for several releases: it lacks most modules (the module
table of the README), fails its tests on Python 3.14 and is documented on Read the Docs as if it were
the package. Nothing in `modcma.c_maes` or `scripts/` imports it.

* **T6.1** Branch `legacy-python` from master (the state of v1.2.0 plus the 1.x fixes), with a README
  note that it is not maintained; tag `v1.2.0` and PyPI 1.2.0 keep it installable.
* **T6.2** Remove it from master: the modules above, their tests (`test_asktellcmaes`, `test_fmin`,
  `test_modularcmaes`, `test_parameters`, `test_population`, `test_sampling`, `test_utils`),
  `tests/expected.json`, `tests/create_expected.py`, the Sphinx pages of `docs/source/`; CI runs all
  tests on every Python version, coverage of the C++ module's Python part only.
* **T6.3** `modcma/__init__.py` exposes the API of `modcma.c_maes` (`from modcma import fmin,
  ModularCMAES, Settings, Modules, ...`); `modcma.c_maes` stays. Constructing `ModularCMAES` with a
  callable first argument (the 1.x signature) raises a `TypeError` that names the 2.0 equivalent and
  the migration guide. `__version__`.
* **T6.4** Version 2.0.0. Dependencies: numpy, scipy (LAPACK/BLAS on Linux and Windows) and
  configspace (`c_maes` tuning) stay; ioh was only used by the Python implementation (`evaluate_bbob`)
  and is dropped (`tests/statistical.py`, `scripts/` and the examples install it themselves).

## WP7: Documentation (D10): branch `wp7-docs` (on WP6)

**Status: done**, except where noted.
* T7.1: every class, method, property, enum value and submodule documented in the bindings (written
  in three parts, by file); stubs regenerated; `tests/test_c_stubs.py` fails on any undocumented
  public name.
* T7.2: Zensical (`zensical.toml`; Material for MkDocs announced its end of life for May 2027),
  mkdocstrings reading the stubs (no build of the extension needed); `tools/gen_docs.py` writes the
  option tables of the guide from the stubs (checked by `tests/test_c_stubs.py` and CI);
  `.readthedocs.yml` (build commands), `docs/requirements.txt` (pinned), a `docs` CI job
  (`zensical build --strict`). `CHANGELOG.md` and `CONTRIBUTING.md` are included as pages
  (snippets).
* Source tree: `PERFORMANCE.md` and `REPRODUCIBILITY.md` moved to `docs/` (figures to
  `docs/assets/performance`, the banner to `docs/assets`), `requirements-experiments.txt` to
  `scripts/`; `CMakeSettings.json` (Visual Studio) and `.prospector.yaml` (lint configuration of the
  Python implementation) removed.
* References checked against the paper (arXiv:2610.09699, and its LaTeX): m-xNES and lp-xNES are
  Krause, Glasmachers & Igel (2017); COTN is the complete one-tailed normal correction, and the
  bound corrections follow Caraffini, Kononova & Corne (2019); SA is Beyer & Sendhoff (2008) with
  the correction of Ahrari et al. (2017); PSR is Loshchilov (2014). The framework paper is cited
  by its arXiv entry until the journal version.
* T7.3–T7.6: getting started (4 pages), user guide (13 pages), examples (6 scripts in `examples/`
  plus the C++ program `examples/cpp`, built in CI), reference (15 pages), migration guide,
  changelog, citing. `tests/test_c_docs.py` runs every python block of the docs and README and every
  example (the BBOB example is skipped without `ioh`).
* T7.7, T7.8: README (641 -> about 150 lines), `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`,
  pyproject metadata (keywords, classifiers incl. free-threading, URLs).
* Besides documentation: `fmin` ignored `x0` (fixed, a test), `str(Modules)` printed `unknown` for
  `CENTER` (fixed), `modcma_core` declares C++17 for the programs that link it.
* Open for the maintainer: the bibliographic entry of the framework paper (only its title is cited);
  references for MXNES, LPXNES and COTN (left out of the docstrings).

No change of behaviour. Today: about 450 bound functions, classes and properties have one docstring
between them (the stubs: 31 of 732 definitions), so `help()`, IDE hovers and any API reference are
empty; the Read the Docs site documents only the pure-Python implementation and its configuration
(`python.version: 3.8`, no `build.os`) no longer builds; the README is 821 lines with a broken first
example (`fmin`: a missing comma, `func` undefined, no import); there is no changelog, migration guide,
`CITATION.cff` or contributing guide.

* **T7.1 Docstrings in the bindings**, the one source for `help()`, IDEs, the stubs and the reference.
  Every class, method, property and enum value; for each module option what it does, when to use it,
  its default and the paper. Stubs regenerated (`tools/gen_stubs.py`); a test that every public name
  in the stubs has a docstring.
* **T7.2 Site skeleton**: MkDocs Material + mkdocstrings-python on the stubs; `.readthedocs.yml`
  (version 2, `build.os`); `mkdocs build --strict` in CI.
* **T7.3 Getting started**: installation (wheels, LAPACK per platform, from source); a first
  optimization with `fmin`; the `ModularCMAES` object; ask-tell and vectorized evaluation; bounds;
  restarts; stopping criteria; reading the result.
* **T7.4 User guide**: the idea of modules (one optimizer, many variants; the default is CMA-ES with
  CSA); one page per module family (matrix adaptation, step-size adaptation, sampling, selection and
  weights, restarts and repelling, bounds, center placement) with options, defaults, when to use them
  and references; `Settings` (every parameter); integer variables; reproducibility (seeds,
  `Settings(seed=)`, threads, `fast_sampling`); performance (LAPACK/BLAS, constants); tuning
  (configspace); using the C++ library from CMake.
* **T7.5 Examples**, executed in CI so they cannot rot: a multimodal problem with BIPOP and repelling
  restarts; parallel evaluation through ask-tell (processes); BBOB with IOHexperimenter; tuning a
  configuration; a mixed-integer problem; from pycma (the equivalent calls).
* **T7.6 Reference pages**: API reference (generated), docs/performance.md and docs/reproducibility.md as pages,
  changelog, migration guide 1.x -> 2.0 (the Python implementation, nanobind, `Settings(mu, lambda)`
  order, ask-tell, constants, removed or renamed names from `tests/data/api/pybind11.json`), citing.
* **T7.7 README** of about 150 lines: what it is and why (modular, fast, ask-tell, tunable), a
  quickstart that runs (tested), the speed figure, a short module table, links to the site.
* **T7.8 Repository files**: `CITATION.cff` (both papers), `CHANGELOG.md`, `CONTRIBUTING.md` (build,
  tests, golden fixtures, statistical tier, stubs), pyproject metadata (description, keywords,
  classifiers incl. Python versions and free-threading, Documentation/Issues/Changelog URLs), badges
  (CI, wheels, PyPI, docs, Python versions).

**Found while writing the docstrings (not fixed: they change results; each needs a decision and the
statistical tier, D2):**

| # | Where | What |
|---|---|---|
| B16 | `src/matrix_adaptation.cpp` (`CovarianceAdaptation`, `SeparableAdaptation`: `decay`) | The h_σ correction has the wrong sign: `1 - c1*dhs - c1 - cmu*sum(w)`, where Hansen (2016) and pycma have `1 + c1*δ(hσ) - c1 - cmu*sum(w)`. Only matters in the generations where hσ = 0 (mostly early). |
| B17 | `src/matrix_adaptation.cpp` (`CovarianceAdaptation::adapt_matrix_inner`) | The lazy interval (`lazy_update_interval`) skips the whole update of C, not only the eigendecomposition: the rank-one and rank-µ information of the skipped generations is lost. pycma updates C every generation and decomposes lazily. Only for d large enough that the interval exceeds 1 (d ≳ 100); it is also part of the speed difference with pycma at d = 1000 (docs/performance.md compares the two as they are). Related to D5. |
| B18 | `src/weights.cpp:21` | The default `cs` of `NATURAL_GRADIENT` is `(9 + 3 + ln d) / (5 d √d)`; the xNES learning rate is `(9 + 3 ln d) / (5 d √d)` (Glasmachers et al. 2010). |
| B19 | `src/matrix_adaptation.cpp` (`NaturalGradientAdaptation`) | `A = I / σ` at the start and at restarts, so σ·A = I: the initial spread does not depend on `sigma0` (the first population has a spread of about 1 for σ₀ = 0.01 and for σ₀ = 1). `A_inv` starts at I, inconsistent with A until the first update. |
| B20 | `src/restart_strategy.cpp:46-49` | BIPOP's small population is `λ_init · (0.5 / λ_large / λ_init)^(u²)`; Hansen (2009) has `λ_init · (½ λ_large / λ_init)^(u²)`, between λ_init and λ_large / 2. As coded the small regime gets λ ≤ λ_init (down to the floor of 2). Verified in the code. |
| B21 | `src/sampling.cpp` (`Orthogonal::sample`) | Orthogonal sampling is not orthogonal: with λ = d = 10, the samples of one generation have cosines up to 0.07 (measured). Likely aliasing: the norms for the rescaling are a lazy expression read while `samples` is overwritten. |
| B22 | `src/restart_strategy.cpp` (IPOP) | The growth limit `(dim * lambda)^2` uses the current λ, so it never binds; `ipop_factor` is cast to an integer (1.5 acts as 1). |
| B23 | `include/sampling.hpp` (`Mirrored`, `Orthogonal`, `CachedSampler`) | `expected_length` is not forwarded to the wrapped sampler: with mirrored or orthogonal sampling CSA normalizes with √d instead of E‖N(0, I)‖. |

Smaller, also not fixed: the first run starts at the origin when `x0` is not given, a restart with
`X0` placement at the center of the box (they differ for asymmetric bounds); COTN can return a point
outside the box for a large σ; `AdaptiveRepelling.target_rejection_rate` and
`local_target_rejection_rate` are not used; misspelled bound names (`Elitsm`, `euclidian`,
`mahanolobis`, `seq_cuttoff_factor`, `adapt_evolution_paths_innner`); the step-size classes and
`CovarianceNoEigvAdaptation` have no Python constructor. `str(Modules)` printed `unknown` for
`center_placement=CENTER`: fixed (display only). `fmin` ignored `x0` and started at the origin:
fixed (it is what the documentation leads with).

Also found, not fixed: `Parameters(d)` / `ModularCMAES(d)` print at every restart (they build the
C++ `Settings` whose `verbose` defaults to true; Python's `Settings(d)` defaults to false);
`Population.__add__` is bound to `+=` and modifies its left operand; `OnePlusOneES::sample` checks
the bounds of the parent instead of the candidate; the scaling of the negative weights uses
`1 + c1/mu` where Hansen (2016) has `1 + c1/cmu` (commented out in `weights.cpp`); with `ssa=SA` or
CMSA an explicit `mu0 > lambda0` is not corrected; `set_module` takes only option names, not enum
members; a `get_configspace` warning names a parameter `only_modules` that does not exist;
`MuCommaLambdaES::step` keeps a sampler in a function static (shared by instances and threads); the
derived bounds values (`db`, `center`, ...) go stale when `lb`/`ub` are changed after construction;
a run stops when `evaluations + lambda >= budget`, so up to λ evaluations of the budget are never
used.
Also: `TolX` compares √Cᵢᵢ·σ/σ₀ with tol·σ₀ (pycma: σ√Cᵢᵢ < tolx; a TODO in the code);
`NoImprovement.tolerance` is not bound; `SampleTranformerType.NONE` gives U(0, 1) mutations, not
centred; `utils.random_normal` keeps a static Box–Muller state across `set_seed`;
`CachedShuffleSequence.fill`/`get_index` and `CachedSampler([])` have no bounds checks;
`utils.compute_ert` divides by zero without successes; `CriteriaVector` has two `__repr__`.

Follow-up, not in WP7 (code): named presets (`"sep-cma"`, `"ma-es"`, `"bipop"`, `"xnes"`, ...) as
starting points for people who know the algorithms by name.

---

## WP8: Bug fixes: branch `wp8-fixes` (on WP7)

The bugs found while writing the docstrings (WP7: B16-B23 and the smaller ones), the open GitHub
issues, and a report about elitism. Each was checked against its source before fixing: Hansen
(2016, the tutorial), pycma 4.6, Glasmachers et al. (2010, xNES), Hansen (2009, BIPOP), Wang et al.
(2014), Hansen (2011, injecting solutions), the framework paper (arXiv:2610.09699) and the Python
implementation (branch `legacy-python`). Most change results, so the golden fixtures are regenerated
(with this list as the reason) and the affected configurations are compared on BBOB (tier 2).

**The elitism report** (`compute_algos.py`: the number of logged evaluations differs between runs
with `elitist=True`). Reproduced: on a random objective, 4994 or 4995 of 5001, against 5000 for
every non-elitist run. Cause: a lucky parent stays in the population while sigma shrinks (to 1e-11);
its step y = (x - m) / sigma, recomputed when it is re-inserted, grows without bound and blows up C
(largest eigenvalue 4e13, then a negative one), so `UnableToAdapt` restarts the run, and each restart
evaluates its new center once. pycma behaves the same way on this function: its elitist runs stop
by `tolx` after about 1850 evaluations, as sigma collapses; modcma with `restart_strategy=NONE` has
no such stop and continues until the update of C breaks (about 1800 evaluations). So the counts
vary with the number of restarts, which is expected on a noise function; what was wrong is the
budget off-by-one (T8.20: a budget of 5000 gave 4990; hence the 5001 of the report). Clipping the
step of the re-inserted parents (Hansen 2011) or rescaling it to a random length (pycma's
`CMA_elitist`) did not prevent the breakdown on noise, and broke MATRIX, CHOLESKY with elitism on
BBOB (their mean comes from x but their step size from the clipped z: 25 -> 16 of 25 runs on f10,
d = 20); the elitist step stays as the paper defines it (Sec. 4.1, Eq. 1-3).

| # | Fix | Source |
|---|---|---|
| T8.1 | Points injected with `tell(f, X)`: the Mahalanobis length of their step is clipped at sqrt(d) + 2d / (d + 2), for the mean as for the paths and the matrix (the evaluated point stays the best solution) | Hansen (2011), pycma |
| T8.2 | B16: covariance decay 1 + c1 delta(h_sigma) - c1 - cmu sum(w) | Hansen (2016), pycma `c1a` |
| T8.3 | B17: C is updated every generation; only the eigendecomposition is lazy | Hansen (2016), pycma, the paper (Sec. 5.2), the Python implementation |
| T8.4 | B18: xNES learning rate (9 + 3 ln d) / (5 d sqrt d) | Glasmachers et al. (2010) |
| T8.5 | B19: xNES starts with A = I (sigma0 sets the spread) | Glasmachers et al. (2010): x = m + sigma B z |
| T8.6 | B20: BIPOP small population lambda_def (lambda_large / (2 lambda_def))^(U^2) | Hansen (2009), pycma |
| T8.7 | B21: orthogonal sampling: the norms are evaluated before the samples are overwritten; blocks of d samples when lambda > d | Wang et al. (2014) |
| T8.8 | B22: IPOP limit (d lambda_0)^2 from the initial lambda; non-integer `ipop_factor` | the Python implementation |
| T8.9 | B23: the mirrored, orthogonal and cached samplers pass on the expected length of their sampler | E||N(0, I)|| (Hansen 2016) |
| T8.10 | Negative weights scaled with 1 + c1 / cmu (not 1 + c1 / mu) | Hansen (2016), pycma |
| T8.11 | EXPONENTIAL weights w_i = 2^-i + 1 / (mu 2^mu) (sum 1); the negative ones mirrored | the paper (Sec. 4.2) |
| T8.12 | TolX: sigma sqrt(C_ii) < tol sigma_0 and sigma |p_c| < tol sigma_0, with sigma_0 of the current run | Hansen (2016) App. B.3, pycma |
| T8.13 | p_sigma starts at 0 for every adaptation (MATRIX, CHOLESKY, CMSA, NATURAL_GRADIENT and NONE started at ones) | Hansen (2016) |
| T8.14 | The first run starts at x0, or at the center of the bounds (as restarts with X0 placement), not at the origin | consistency with `CenterPlacement.X0` |
| T8.15 | `OnePlusOneES` checks the bounds of the candidate (it checked the parent); `MuCommaLambdaES` has its own sampler (a function static, shared by instances and threads) | |
| T8.16 | COTN places the coordinate inside the box (resampled when sigma |n| > 1) | Caraffini et al. (2019), the paper (Sec. 3.6) |
| T8.17 | Settings: `mu0 > lambda0` corrected also with SA and CMSA; C++ `verbose` defaults to false (as in Python; `ModularCMAES(d)` printed at every restart); changing `lb`/`ub` updates the derived values | |
| T8.18 | `set_module` accepts enum members; the `get_configspace` warning names the right arguments | |
| T8.20 | The budget: a run stopped when `evaluations + lambda >= budget`, so a generation that ends exactly at the budget was not run (budget 5000, lambda 10: 4990 evaluations); now `>` (`max_generations` was already right) | |
| T8.19 | Bindings: `Population.__add__` returns a new population (it modified the left one); `NoImprovement.tolerance` bound; one `CriteriaVector.__repr__`; `utils.random_normal` without a static state; bounds checks in `CachedShuffleSequence` and `CachedSampler`; correctly spelled names next to the old ones (`Elitism`, `NoElitism`, `seq_cutoff_factor`, `euclidean`, `mahalanobis`) | |

**Issues**

* #42 (sequential selection with active update breaks the eigendecomposition): reproduced with
  modcma 1.2.0 (230 failures in 216 BBOB runs with active, none without); none with the current
  code (fixed by B2 and B13 in WP0). A regression test.
* #35 (stubs not in the pip package): 1.2.0 installs no `.pyi`, the current wheels install 13 (WP1).
  Adds `py.typed` (PEP 561), without which type checkers ignore them; the stub test requires them.
* #59 (an allocation per sample): allocation-free since WP2 (`modcma_bench`: 0.03-0.22 mallocs per
  evaluation).
* #24 (the favorite candidate, as pycma's `xfavorite`): `ModularCMAES.xfavorite`, the mean of the
  search distribution as a point of the objective (integer variables rounded, inside the bounds
  when a bound correction is used): with noise, a better estimate of the optimum than the best
  evaluated point.

**Not changed** (for the maintainer): a run never exceeds the budget, so up to lambda - 1
evaluations can stay unused (pycma may exceed `maxfevals` instead); `AdaptiveRepelling.target_rejection_rate`/`local_target_rejection_rate` are not used;
the step-size classes have no Python constructor.

---

## WP23: Second round of fixes (found while writing the deep dives; not started)

Three read-only reviews (2026-11-24) compared every module with its source (the papers, pycma 4.6,
`cmaes`, the framework paper's LaTeX, the Python implementation), and tested every module value,
every pair and the triples of passing pairs, plus 600 random configurations (sphere and Rastrigin
with a shifted optimum: with the default box the origin is the start, which hides problems).
Scripts: the session scratchpad, research_a/b/c. Not fixed yet; most change results (statistical
tier, golden fixtures).

| # | Where | Problem | Evidence / source |
|---|---|---|---|
| F1 | `mutation.cpp:56-64`, `mutation.hpp:95-108` | Sampled step sizes (`sample_sigma`, SA, LPXNES) are one per coordinate, not one per individual: x = m + S∘y is not along y, rotation invariance is lost, and SA's signal shrinks by sqrt(d) | the paper (one sigma per individual), the Python implementation; CMSA+SA and CSA+`sample_sigma` solve 0/5 rotated ellipsoids (d = 10) |
| F2 | `matrix_adaptation.cpp:693-742` | NATURAL_GRADIENT + active uses CMA's active weights (sum -0.4 to -1.2, heavy on the worst) as xNES utilities: sigma shrinks on a linear function, runs converge prematurely (0/10 on the sphere). The xNES utilities are now available as `RecombinationWeights.UTILITY` (2026-11-24: NATURAL_GRADIENT + XNES + UTILITY matches `cmaes`' XNES within 1 %); **fixed**: NATURAL_GRADIENT with active and the default weights uses them (an adjustment of the Settings) | xNES utilities sum to 0 (Glasmachers et al. 2010; `cmaes` XNES) |
| F3 | `weights.cpp:20-22` | NATURAL_GRADIENT replaces `cs` (and `damps`, `cc`) of every step-size rule by the xNES rate; meant for ssa=XNES (LPXNES+NG does not converge) | |
| F4 | `weights.cpp:88-97` | SEPARABLE: the cap cmu <= 1 - c1 is applied before the (d+2)/3 scaling; with large populations (IPOP) the decay of C becomes negative | |
| F5 | `matrix_adaptation.cpp:508-519` | CHOLESKY + active: the failure return of `llt_rank_update_lower` is ignored, a partially downdated factor is used (4 of 88 generations, d = 5) | Eigen LLT.h |
| F6 | `matrix_adaptation.cpp:570-597` | CMSA + active: negative weights without rescaling, decay ignoring them: C indefinite, thousands of UnableToAdapt restarts | |
| F7 | `matrix_adaptation.cpp:412, 508; 330-334` | MATRIX, CHOLESKY, SEPARABLE + active: the decay assumes sum(w) = 1 and the negative weights are not rescaled by d/||A^-1 y||^2 (COVARIANCE does both) | Hansen 2016 (active rescaling) |
| F8 | `weights.cpp:175-185` | DEFAULT weights with mu != lambda/2: "negative" weights come out positive (lambda 10, mu 6: all four), applied with a minus sign by some adaptations and as positive by others | |
| F9 | `selection.cpp:26-34`, `mutation.cpp:239-258` | PAIRWISE sets the losers' f to +inf: MSR's median is inf (sigma shrinks 5 % per generation whatever f does), `stats.current_avg` is inf, Stagnation's medians are inf, active assigns the negative weights by index | |
| F10 | `sampling.cpp:12-23` | Mirrored pairs fall out of step: a RESAMPLE or repelling rejection, odd lambda, sequential selection, lambda = 1 carry the mirror state over; PAIRWISE then compares unrelated points | 0 of 5 pairs mirrored with RESAMPLE |
| F11 | `sampling.cpp:51-66` | Orthogonal blocks are not aligned with generations when lambda < d (samples of one generation not orthogonal); the "- 2 for TPA" is a leftover (TPA's points are extra here) | Wang et al. 2014 (per generation) |
| F12 | `matrix_adaptation.cpp:139-142, 306` | h_sigma's bias correction counts generations as evaluations / lambda over all runs (restarts, TPA, restart centers included) | Hansen 2016, pycma: per run |
| F13 | `mutation.cpp:169-171` | Sequential selection compares with the best over all runs; after a restart it never triggers | the paper: improvement over the previous generation |
| F14 | `matrix_adaptation.cpp:85-86` | dm is divided by mean(S) instead of sigma (differs with sampled step sizes, integer variables, elitism) | |
| F15 | `mutation.cpp:217` | TPA counts a tie (f+ == f-) as success: sigma grows 16 % after every restart (dm = 0), and diverges with elitism | pycma: a tie is 0 |
| F16 | `weights.cpp:170` | LPXNES at d = 1: beta = ln 2 / (sqrt(d) ln d) is inf | the Python implementation used max(..., 1) |
| F17 | `settings.hpp:78, 97-98` | lambda0 = 1 with PAIRWISE: mu0 = 0, NaN weights | |
| F18 | `matrix_adaptation.cpp:424-441` | MATRIX + elitist: the incremental update of M^-1 drifts from the inverse (0.98 relative after 60 generations); the elites' z are wrong (3/5 failures on an ellipsoid) | |
| F19 | `center_placement.cpp:26-31, 42-49, 87-94` | UNIFORM, NOVELTY_WEIGHTED, MAXIMIN_TABOO with an infinite or half-infinite box: NaN restart centers | |
| F20 | `settings.hpp:142-156`, `bounds.cpp:34` | A half-infinite box disables bound correction on the finite side too | |
| F21 | `mutation.cpp:141`, `bounds.cpp:47` | Integer variables: rounding after the correction can leave the box (lb 0.6, ub 0.9 evaluates 1.0) | |
| F22 | `sampling.hpp:33, 341-349`; configspace | `sample_transformation=NONE` (U(0, 1), mean 0.5) is in the configuration space and fails everywhere | |
| F23 | `modcma/c_maes/__init__.py:245-276` | Configspace: the numeric defaults are those of CSA + COVARIANCE and are passed explicitly, so module-specific defaults (XNES/MXNES cs, SR, NG, SEPARABLE, mu = lambda/4) are lost; no conditions for inactive options | matters for tuning |
| F24 | `mutation.cpp:9-20, 123-124` | Threshold convergence: a threshold in x units applied to z; `budget - evaluations` underflows (size_t) past the budget in ask-tell | the paper: ||x - m||, per generation |
| F25 | `restart_criteria.cpp:384-390, 226-279` | TolX, ConditionC, NoEffectAxis, NoEffectCoord only for COVARIANCE with CSA (lost when ssa != CSA converts to NO_EIGV); NoEffect criteria are absolute, not relative to m; MaxSigma absolute | pycma (also WP17) |
| F26 | `restart_strategy.cpp` | BIPOP: small runs have no iteration cap; lambda_small rounded up to even | Hansen 2009, pycma |
| F27 | `weights.cpp:24-28` | ssa=XNES without NATURAL_GRADIENT: cs = 0.01 (the formula of the Python implementation is commented out): slow | |
| F28 | docstrings | "With XNES the full xNES" overstated; "CSA does not suit COVARIANCE_NO_EIGV" not supported (same results as COVARIANCE on the tested functions); NG's override of cs undocumented | |
| F29 | sampling | The expected length of UNIFORM, LAPLACE, LOGISTIC, DOUBLE_WEIBULL transformations is sqrt(d), not theirs (CSA bias); Sobol points repeat with period N (d = 1000: after about 85 generations) | |
| F30 | `c_maes.cpp` `break_conditions` | The budget check assumed the next generation has the current lambda; when the restart criteria fire, the next generation also evaluates the restart center (and repelling queries) and IPOP/BIPOP change lambda, so a run could exceed the budget by up to lambda + 1 (rotated Rosenbrock, SEPARABLE + IPOP, d = 10: 200500 of 200000). Also the hill-valley tests of repelling evaluated points at every restart without a budget check (CMSA + active + ADAPTIVE repelling, budget 1000: up to 1043). **Fixed**: the first generation after a restart takes at most the rest of the budget (lambda and mu capped, at least 2 points; `Parameters::fit_lambda_to_budget`); the check asks for the center + 2 points when a restart is due; the hill-valley tests stop when the next point would leave less (`Parameters::restart_fits`). Changes one golden run (pair/015: a restart with lambda = 2048 and 1648 evaluations left now runs instead of stopping at 4352 of 6000; on gcc 13 also pair/017, repelling: 6025 of 6000 before) | pycma checks `maxfevals` after a generation (overshoots by up to lambda - 1) |

Design questions (not bugs, for the maintainer): elitism keeps all lambda survivors ((lambda+lambda),
the paper: (mu+lambda)) and each parent's own step sizes; `restart_strategy=NONE` still restarts on
UnableToAdapt (the paper: ignore and continue); MSR compares with the median (pycma: the 0.3-quantile);
PSR's cs 0.9 (the Python implementation: 0.4); the IPOP limit (d lambda_0)^2 (the paper: 2^9 lambda_0);
pairwise + active (mirrored losers get negative weights; pycma never gives mirrors negative weights).

**The paper and the code disagree** (for the review response): one sigma per individual (paper) vs
per coordinate (F1); the xNES step-size description matches MXNES; SR's success is "better than the
best of generation g" in the paper, the best so far of the run in the code; sequential selection
(F13); threshold convergence (F24); the IPOP limit; elitism ((mu+lambda) with sigma^(g)); the
variant count 14,859,936 counts combinations that are identical (COVARIANCE vs COVARIANCE_NO_EIGV
with ssa != CSA, center placement and repelling metric without restarts or repelling, sample_sigma
with SA/LPXNES, sequential selection with PAIRWISE at mu = lambda/2, the lambda = 1 overrides).

## Planned work (WP9-WP22, later; not started)

From the comparison with pycma and `cmaes` (2026-11-24). Divided into **modules** and **features**:

* A **module** is an interchangeable part of the evolution strategy: an option of `Modules` (an enum
  or a boolean), so it combines with every other module, is part of the configuration space
  (`get_configspace`) and can be tuned, benchmarked and compared like the existing ones. It changes
  how the strategy samples, selects or adapts.
* A **feature** changes what the optimizer can be used for, not the strategy itself: the problem it
  accepts (constraints, variable types, a gradient), how it is set up or run (a normalized space, a
  warm start, stopping, saving a run). It is set through `Settings` or the API, works with every
  configuration of modules, and is not part of the configuration space.

All work with ask-tell, are off by default (no change of results), and are validated on BBOB (or the
problem class they are for) against pycma or `cmaes` where these have them. Not planned: surrogates
(lq-CMA-ES), the options that shape the search distribution (per-coordinate sigma0, fixed variables,
min/max std, condition alleviation, mean learning rate, arbitrary weights, selective mirroring, CSA
variants), multi-objective variants (COMO-CMA-ES, MO-CMA-ES), Safe CMA-ES, confidence intervals
(libcmaes), DX-NES-IC and MAP-CMA, GPU or distributed evaluation.

### Modules

**Common requirement: extra points from several modules.** Today only the step-size rule can add
evaluations to a generation (`mutation::Strategy::n_extra_points`, TPA's two points); the budget
check, the batches of ask-tell and `consume` ask only it. HE-ES/QN-ES (f at the mean, needed before
the adaptation), the noise handling (re-evaluations, needed before selection) and the quasi-Newton
step length need the same, together with TPA's: a list of extra-point providers, each with the
phase in which its values are needed. A prerequisite of WP9, WP11 (HE-ES) and WP22.

The interactions below come from the compatibility review (2026-11-24); each task settles them,
the Settings enforce the incompatible ones (adjusting or rejecting, as for the existing
combinations), and the compatibility page of the docs lists them.

#### WP9: Noise handling (`Modules.noise_handling`)

* **T9.1 Uncertainty handling (UH-CMA-ES; Hansen et al. 2009, "A method for handling uncertainty in
  evolutionary optimization with an application to feedback control of combustion"; pycma's
  `NoiseHandler`).** A module `noise_handling` (NONE, UH): each generation a few solutions are
  re-evaluated, the change of their ranks measures the noise, and when it exceeds a threshold sigma
  is increased (or the number of evaluations per point). Ask-tell: the re-evaluations are extra
  points of the generation, as TPA's (the state machine of WP3 already has them).
* **T9.2 Adaptive re-evaluation** (Dinu et al. 2025, cited in the framework paper) as a second option,
  after T9.1.
* **T9.3** The guide page on noise: `xfavorite` (#24), larger populations, learning-rate adaptation
  (WP10), the restart criteria that misfire under noise (flat fitness, no improvement), with a
  BBOB-noisy example (ioh has the noisy functions).

* **Interactions.** Re-evaluations happen before selection; the budget counts them (variable with
  adaptive re-evaluation). Conflicts: sequential selection (stops on one noisy value), elitism
  (elites keep lucky values: re-evaluate them or disable), PAIRWISE and TPA (decide on single
  values), SR's success ratio, the hill-valley tests of repelling; the restart criteria FlatFitness,
  NoImprovement, Stagnation and TolX misfire under noise, MaxSigma fights the sigma increase.
  Restarts reset its state. With SA/LPXNES sigma is also encoded in the sampled step sizes: decide
  how the increase applies. Bound correction and integer variables: re-evaluate the evaluated point.

#### WP10: Learning-rate adaptation (`Modules.lr_adapt`)

* **T10.1 Learning-rate adaptation** (LRA-CMA-ES; Nomura, Akimoto & Ono 2023, "CMA-ES with learning
  rate adaptation: can CMA-ES with default population size solve multimodal and noisy problems?";
  `cmaes`' `CMA(lr_adapt=True)`): the learning rates of the mean and of C are adapted to keep a
  target signal-to-noise ratio of the update, so the default population size copes with noise and
  multimodality. A boolean module `lr_adapt`; for every matrix adaptation that has the two rates,
  first COVARIANCE. Compared with `cmaes` on the noisy and multimodal BBOB functions.

* **Interactions.** A mean learning rate below 1 shrinks dm, which CSA's path, TPA's line and h_sigma
  see; user-given c1 and cmu are scaled by it. With COVARIANCE_NO_EIGV the signal-to-noise ratios
  can use the Cholesky factor (invariant to the factor). MATRIX (multiplicative update) needs its
  own definition of the change of the matrix; NATURAL_GRADIENT overlaps with xNES's rates.
  Elitism and lambda = 1 are outside the method (Settings switch it off); meaningless with the
  quasi-Newton recombination (WP22). Restarts reset it. The lazy decomposition gives a stale metric.

#### WP11: Matrix adaptations (`Modules.matrix_adaptation`)

* **T11.1 dd-CMA-ES** (diagonal decoding; Akimoto & Hansen 2020, "Diagonal acceleration for covariance
  matrix adaptation evolution strategies", Evolutionary Computation; pycma's `CMA_diagonal_decoding`):
  a diagonal scaling adapted alongside C, fast on separable and ill-scaled problems without losing
  rotation invariance. `MatrixAdaptationType.DIAGONAL_DECODING`.
* **T11.2 VkD-CMA** (Akimoto & Hansen 2016, "Projection-based restricted covariance matrix
  adaptation for high dimension", cited in the paper; pycma's `restricted_gaussian_sampler`):
  C = D (I + V V^T) D with k vectors, O(d k) memory and time per sample; k adapted. VD-CMA is k = 1.
  `MatrixAdaptationType.VKD`.
* **T11.3 HE-ES** (Glasmachers & Krause 2020, "The Hessian Estimation Evolution Strategy", PPSN XVI;
  arXiv:2003.13256): a covariance adaptation that estimates curvature directly. Each generation it
  samples mirrored pairs of orthogonal directions, evaluates f at the mean as well, and from
  f(m + sigma b), f(m - sigma b) and f(m) estimates the curvature along each direction; the factor A
  (an estimate of the inverse square root of the Hessian) is updated from these estimates.
  `MatrixAdaptationType.HESSIAN_ESTIMATION`; it needs orthogonal mirrored sampling (set by the
  Settings, as other combinations are adjusted) and the evaluation at the mean (an extra point of
  the generation, as TPA's: the state machine of WP3 supports it). Details (the update and its
  learning rate, the step-size adaptation, the treatment of negative curvature) from the paper;
  the authors' reference code where available.
* **T11.4** The elitist (1+4)-HE-ES (Glasmachers & Krause 2022, "Convergence analysis of the Hessian
  estimation evolution strategy", Evolutionary Computation 30(1)), for which linear convergence on
  convex quadratic functions is proven, as the lambda = 4 elitist configuration.
* **T11.5** Validated against the paper's results (convex quadratic and BBOB functions) and against
  COVARIANCE.
* **T11.6 QN-ES** (Glasmachers 2025, "A superlinearly convergent evolution strategy",
  arXiv:2505.10987) is HE-ES (T11.3) with a quasi-Newton step instead of the weighted recombination
  of the mean: in the modular design, `matrix_adaptation=HESSIAN_ESTIMATION` with
  `recombination=QUASI_NEWTON` (WP22), validated against the paper's results.
* **T11.7** The benchmarks of the docs extended to d = 10 000 for SEPARABLE, VKD, MATRIX; the guide
  page on matrix adaptation with these two.

* **Interactions.**
  * dd-CMA-ES: CSA works in its own coordinates, so the "ssa != CSA -> COVARIANCE_NO_EIGV" rule does
    not apply; the restart criteria that read B and D (today only for COVARIANCE) must read
    D C~ D; the integer floor and the margin (WP14) use its diagonal; no (1+1) variant.
  * VkD-CMA: the inverse in O(dk) (Woodbury) for elitism, bound repair, injection and the CURRENT
    repelling metric; the ARCHIVED metric (a d x d matrix per taboo point) defeats its O(dk)
    memory; its reference step-size rule may not be CSA (to check); ConditionC from D and V.
  * HE-ES: needs MIRRORED (not PAIRWISE, which sets the losers' f to inf before the adaptation),
    orthogonal sampling and an even lambda (the Settings set them) and f at the mean (an extra
    point; after a restart it is the evaluated restart center). Incompatible: sequential selection,
    sampled step sizes (so SA and LPXNES), RESAMPLE and repelling rejections (they break the pairs,
    F10), injected points, integer variables; a bound repair of one member drops the pair (or use the
    transformation, WP12). Active is meaningless (switched off, as for NONE). The (1+4) variant is
    lambda = 4, mu = 1, elitist, mirrored, orthogonal, with a success rule, not the lambda = 1 path.
  * QN-ES / quasi-Newton recombination: see WP22; CSA's path length is meaningless for Newton steps
    (it needs its own step-size rule or TPA as a line search).

#### WP12: Bound transformation (`Modules.bound_correction`)

* **T12.1 Bound transformation** (pycma's default `BoundTransform`: a piecewise linear-quadratic
  map, the identity in the interior of the box, onto [lb, ub]). The search distribution lives in an
  unbounded space and the objective sees x = T(x'): no repair, so the update needs no recomputed
  y and z. As a `bound_correction` option (TRANSFORM) or in the coordinate mapping (which already
  rounds the integer variables), so it composes with integer variables.
* **T12.2** The guide page on bounds compares repair and transformation (BBOB with bounds, SBOX-COST).

* **Interactions.** Best placed in the coordinate mapping (so round(T(x)) composes with integer
  variables); no recomputation of y and z. Must work per coordinate for half-infinite boxes (F20);
  `xfavorite` returns T(m); injection needs a preimage; the repelling archive must hold internal
  coordinates (today the hill-valley query maps archived points again). MaxSigma no longer pulls
  sigma back; `n_out_of_bounds` loses its meaning; it excludes the other corrections (same enum).

#### WP13: Restart from the best point (`Modules.center_placement`)

* **T13.1 Restart from the best point**: `CenterPlacement.BEST` (pycma's `restart_from_best`), the
  best solution so far as the center of a new run.

* **Interactions.** Conflicts with repelling restarts (the best point is in a taboo region:
  rejections until TooMuchRepelling) and with NOVELTY_WEIGHTED/MAXIMIN_TABOO (opposite purpose):
  warn or exclude. The best point is the evaluated one (rounded, or transformed with WP12): it
  needs internal coordinates. Its value is known, so the restart center needs no evaluation.

#### WP14: Integer handling with a margin (`Modules.integer_handling`, new)

The integer variables are a feature (`Settings.integer_variables`); how they are handled becomes a
module: LOWER_BOUND (the current lower bound on their step size, Marty et al. 2024, the default)
or MARGIN.

* **T14.1 CMA-ES with Margin** (Hamano, Saito, Nomura & Shirakawa 2022, "CMA-ES with margin:
  lower-bounding marginal probability for mixed-integer black-box optimization"; `cmaes`' `CMAwM`):
  keeps the probability that an integer variable changes its value above a margin, by correcting the
  mean and the diagonal of C. Replaces the lower bound on the step size of the integer variables
  (Marty et al. 2024, the current mechanism) as an option; compared on mixed-integer benchmarks.

* **Interactions.** Replaces the floor on the step size of integer variables; uses the marginal
  variances (available for every adaptation). Its mean correction comes after recombination, so it
  changes dm (CSA, TPA, h_sigma) and conflicts with elitism and the quasi-Newton step; rounding
  targets within [lb, ub] (fixes F21). With dd-CMA two diagonal corrections: check they do not
  fight. Incompatible with HE-ES and QN-ES (rounding). Restarts reset it.

#### WP22: Recombination (`Modules.recombination`, new)

The mean update becomes a module. The framework's purpose is to combine mechanisms that were
proposed separately and see whether they work together; QN-ES's quasi-Newton step is one of them.

* **T22.1** `RecombinationType.WEIGHTED`: the current weighted recombination of the mu best points
  (the default, unchanged results).
* **T22.2** `RecombinationType.QUASI_NEWTON` (Glasmachers 2025): the mean moves by a quasi-Newton step
  m <- m - H^-1 g, with g estimated from the function values of mirrored pairs (central
  differences along the sampled directions, (f(m + sigma b) - f(m - sigma b)) / (2 sigma)) and H^-1
  from the matrix adaptation. First with HE-ES (T11.6, QN-ES as published: HE-ES estimates the
  curvature, so the step has its scale).
* **T22.3 Research: with the other matrix adaptations.** On a quadratic function C, M M^T, A A^T all
  approach H^-1 up to a scale that sigma absorbs; the direction -sigma^2 C g is available, the step
  length is not. Candidates: the curvature along the step direction from one or two extra
  evaluations (TPA already evaluates two points along the last mean shift), or a trust region.
  Compared on smooth BBOB functions (sphere, ellipsoids, Rosenbrock, f1-f14) and on rugged and noisy
  ones (f15-f24, the noisy suite), against WEIGHTED; a hybrid with restarts (weighted recombination
  for the global phase, the quasi-Newton step once a run has converged into a smooth basin) as an
  option to evaluate. Not aware of a published version: possibly a paper.
* **Interactions** (to settle in the task):
  * the step uses f-values, not ranks: the invariance of CMA-ES to monotone transformations of f is
    lost (documented; the guide must say when not to use it: noise, plateaus, rugged landscapes);
  * requires mirrored sampling (the Settings set it) and, for the curvature, an evaluation at the
    mean or along the step (extra points, as TPA's);
  * the recombination weights do not apply to the mean (they still weight the matrix update);
    elitism (the mean is no longer a combination of the parents) and sequential selection (the
    gradient needs every direction evaluated) conflict: disabled or rejected by the Settings;
  * with bounds the step may leave the box (bound correction of the mean, or a step length limited
    to the box); integer variables break the smoothness the step relies on;
  * ask-tell: the extra points are part of the generation, as TPA's.

### Features

#### WP15: Constraints

* **T15.1 Augmented Lagrangian** (Atamna, Auger & Hansen 2016; Dufossé & Hansen 2021, cited in the
  paper; pycma's `ConstrainedFitnessAL`/`fmin_con2`). The objective returns f and the constraint
  values g (g <= 0 feasible); the Lagrange multipliers and penalty factors adapt every generation;
  `tell(f, g)` for ask-tell. The best *feasible* solution is tracked separately.
* **T15.2 Infeasible points** (pycma's `is_feasible`): an objective value of NaN (or None) means
  infeasible; the point is resampled (bounded number of times), as RESAMPLE does for the box.
* **T15.3** Examples: a linearly constrained sphere, a design problem with nonlinear constraints.

#### WP16: A normalized search space

* **T16.1 Normalized space**: `Settings(normalize=True)` optimizes in [-1, 1]^d (centered at 0, the
  natural space of CMA-ES: sigma0 relative to a box of width 2) and maps to [lb, ub] for the
  objective; for problems whose variables have different scales known a priori. With T12.1 the map
  is smooth at the bounds. The results (best solution, `xfavorite`, the logged x) are in the space
  of the objective.

#### WP17: Stopping criteria

* **T17.1 The missing stopping criteria of pycma**, as `Criterion` classes (they restart, or stop with
  `restart_strategy=STOP`): `TolFun` (range of the recent and current values below tolfun),
  `TolFunRel` (relative to the first values), `TolFacUpX` (sigma grew by a factor 1e3 against
  sigma0: sigma0 too small), `TolXStagnation` (the mean stops moving), `Timeout` (wall-clock time).
  Each against pycma's definition; the guide page on restarts lists them with pycma's names.

#### WP18: Categorical variables

* **T18.1 CatCMA** (Hamano et al. 2024, "CatCMA: stochastic optimization for mixed-category
  problems"; `cmaes`' `CatCMA`, `CatCMAwM`): categorical variables with a categorical distribution
  next to the Gaussian, updated jointly. `Settings(categorical_variables=...)` (the number of
  categories per variable); the objective receives the category indices. Useful for algorithm
  configuration (the modules of modcma itself are categorical).

#### WP19: Warm starting

* **T19.1 WS-CMA-ES** (Nomura, Watanabe, Akimoto, Ozaki & Onishi 2021, "Warm starting CMA-ES for
  hyperparameter optimization"; `cmaes`' `get_warm_start_mgd`): the initial mean, sigma and C
  estimated from good solutions of a related task (a list of (x, f) pairs).
  `modcma.warm_start(solutions, ...)` returns the `Settings` (x0, sigma0) and the initial C; the
  optimizer accepts an initial C (or A).

#### WP20: Gradient injection

For users who have a gradient (analytic, automatic differentiation, adjoint simulations) and come
from gradient-based optimization. Built on `tell(f, X)` and the clipping of injected points (WP8,
Hansen 2011).

* **T20.1** Each generation one candidate along the negative gradient from the mean, x = m - sigma
  c C grad / sqrt(grad^T C grad) with the step length of a typical sample, replaces a sampled point
  (Hansen 2011, "Injecting external solutions into CMA-ES"; libcmaes' GI-CMA-ES). The gradient comes
  from a callable `gradient(x)` (`run(f, gradient=...)`, `fmin(..., gradient=...)`) or, with ask-tell,
  from `es.inject_gradient(g)` before `tell`. A bad gradient costs nothing but the replaced sample:
  the injected point is only selected when it is good.
* **T20.2** Compared with CMA-ES without injection on smooth, ill-conditioned functions (BBOB f2,
  f10, f12; d up to 100) with exact gradients, and with libcmaes' results; a guide page ("Coming
  from gradient-based optimization").

#### WP21: Saving and resuming a run

* **T21.1** Pickling of the optimizer: `ModularCMAES`, `Parameters`, `Settings` and every module
  (`__getstate__`/`__setstate__` in the bindings, from a serialization of the C++ state, including
  the random generators and the samplers' positions, e.g. Sobol and cached sequences).
* **T21.2** A run that is pickled and resumed continues bit for bit (a test over the golden
  configurations); the ask-tell state included (a pending batch).
* **T21.3** A how-to: checkpointing a long run every n generations.
