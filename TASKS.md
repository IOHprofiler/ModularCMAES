# ModularCMAES (C++ version): nanobind, ask-tell, memory & compute — task list

Draft for sign-off. Nothing here is implemented yet. Scope: the C++ implementation (`src/`, `include/`,
`modcma/c_maes/`) and its packaging. The pure-Python implementation (`modcma/*.py`, incl.
`modcma/asktellcmaes.py`) is stale and is **not touched**; its tests only have to keep running.

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
| B10 | `src/restart_criteria.cpp:251-256` | `Stagnation::update` copies and `nth_element`s the whole fitness history 4× every generation: O(t) per generation, O(T²) per run. Only noticeable in long runs. |

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

## WP2: Memory: preallocated, in-place population and allocation-free sampling

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
* `Y = A·Z` as one GEMM for the whole batch, then `X = m + S∘Y`. Only columns rejected by `RESAMPLE` or
  repelling are redrawn per column.
* `CovarianceAdaptation`: don't form `inv_root_C` (an O(d³) GEMM per eigendecomposition). Compute `ps`
  as `B·(D⁻¹·(Bᵀ·dm))` and the active-weight norms via `invert_y`. Rank-µ update as a lower-triangular
  `rankUpdate`/`selfadjointView`, which drops the `0.5(C + Cᵀ)` pass. Same for `SelfAdaptation`.
* `NaturalGradientAdaptation::compute_gradients`: `G = Z·diag(w)·Zᵀ − Σw·I` in one GEMM instead of λ
  outer products with d×d temporaries.

## WP3: Ask-tell interface (C++ first, then Python)

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

## WP4: Remaining compute items (after WP2/WP3; each measured, each optional)

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
