# Compatibility of modules

Every combination of modules can be constructed and run, but not every combination is a different
algorithm, and a few are known to work badly. This page collects how modules interact: the
combinations the `Settings` adjust, the options that have no effect in some combinations, the ones
whose meaning changes, and the combinations to avoid. The [deep dives](../deep-dives/matrix-adaptations.md)
explain how the modules compare with the published algorithms they come from.

The lists come from the code and from a test of every module value, every pair of values and
600 random configurations (shifted sphere and Rastrigin, 2026-11-24; WP23 of
[`dev/overhaul-2.0.md`](https://github.com/IOHprofiler/ModularCMAES/blob/master/dev/overhaul-2.0.md)),
updated with every fix since. A measured number gives its setup, and the issue or pull request
that measured it.

## Combinations that are adjusted

The `Settings` change some choices when they are created; `es.p.settings.modules` shows what is
used. Only the constructor adjusts: setting `lambda0` or `mu0` later, or changing
`settings.modules` in place, does not, and the optimizer rejects an odd λ with PAIRWISE and a
change of λ to or from 1 when it is created ([settings](settings.md#changing-the-settings)). A
`mu0` above `lambda0` is not adjusted: it raises `ValueError` (before 2.0 it became λ/2).

| when | then |
|---|---|
| `ssa` other than `CSA` with `matrix_adaptation=COVARIANCE`, also with `lambda0=1` (which runs `SR`) | `COVARIANCE_NO_EIGV` (unless `always_compute_eigv=True`): the same covariance update, sampled through a Cholesky factor |
| `ssa=SA` or `matrix_adaptation=CMSA` | µ defaults to λ/4 (an explicit `mu0` wins), also for the new λ of an IPOP or BIPOP restart |
| `matrix_adaptation=NONE` | `active` off |
| `matrix_adaptation=CMSA` | `active` off: CMSA has no active variant, and its update with negative weights made C indefinite (#73); with `weights=UTILITY` CMSA uses the positive utilities |
| `mirrored=PAIRWISE` and an odd λ above 1 | λ + 1; the default µ is half the new λ (a given `mu0` is kept) |
| `lambda0=1` | the (1+1)-ES: µ = 1, elitist, equal weights, `ssa=SR`, no active update or sequential selection; IPOP and BIPOP become `RESTART`; `mirrored=PAIRWISE` becomes `NONE` (it has no pairs to select from; `MIRRORED` stays, without effect) |
| `ssa=SA` or `LPXNES` | sampled step sizes (`sample_sigma`) are always on |
| `weights=UTILITY` | `active` on (the utilities of the worse half are negative; every matrix update uses all of them) |
| `matrix_adaptation=NATURAL_GRADIENT` with `active` and the default weights | `weights=UTILITY`: xNES's active form (CMA-ES's negative weights broke it) |
| `mirrored=PAIRWISE` with `sequential_selection` | the generation stops only after complete pairs, after 2µ + 2 points at the earliest (µ + 1 without `PAIRWISE`) |
| `matrix_adaptation=NATURAL_GRADIENT` | `cc`, the learning rate of A, is xNES's η_B/2; with `ssa=XNES` also `cs` is xNES's η_σ. The other step-size rules keep their own `cs` (#72) |
| `ssa=LPXNES` | `cs` = min(1, 9µ_eff/(10√d)): Krause et al.'s rate, capped at 1 by modcma (above 1 the update overshoots; 1.29 at d = 2) (#72) |
| a `cs` above 1 (`XNES` with a large λ: 13.8 at d = 2, λ = 100) | the path p_σ (h_σ, MATRIX's rank-one update) has the rate 1, the current step (`Weights.cs_path`); with `cs` above 2 it became NaN (#72) |
| `matrix_adaptation=SEPARABLE` | `c1` and `cmu` are pycma's separable rates (`CMA_diagonal`), `cmu` at most 1 − `c1`; `acov` is not used (#73) |
| a coordinate with one infinite bound | `MIRROR`, `TOROIDAL`, `COTN` and `UNIFORM_RESAMPLE` set a value that crosses its finite bound to that bound, as `SATURATE` (they need the width ub − lb); the coordinates with two finite bounds use the method ([infinite bounds](bound-correction.md#infinite-bounds)) |
| `UNIFORM`, `NOVELTY_WEIGHTED`, `MAXIMIN_TABOO` with an infinite bound | in that coordinate, the centers are uniform in x0 ± 2σ0, shifted inside a finite bound ([restarts](restarts.md#where-a-restart-begins)) |
| `C0` with `matrix_adaptation=SEPARABLE` | only the diagonal of `C0` is used (for the `C0` of `warm_start`, the diagonal covariance of the paper's WS-sep-CMA-ES) |
| `C0` with `matrix_adaptation=NONE` | rejected (`ValueError`): C stays the identity |

## Options without effect

| option | no effect when |
|---|---|
| `active` | `matrix_adaptation=NONE` or `CMSA` (switched off); λ = 1; µ = λ (no negative weights); `mirrored=PAIRWISE` with the default weights (the negative weights fall on losers, see below; with `elitist` too, where they go to the worst points of the pool, mostly losers) |
| `repelling_type`, `repelling_metric` | there are no restarts: `restart_strategy=STOP`, and `NONE` unless `UnableToAdapt` fires; `repelling_metric` also without `repelling_type`, and `CURRENT` (the default) and `ARCHIVED` equal `EUCLIDEAN` with `matrix_adaptation=NONE` |
| `center_placement` | without restarts (as for repelling): it only places restarts, the first run starts at `x0` (or the center of the bounds). `X0` and `CENTER` coincide without `x0`, `ZERO` and `CENTER` for a symmetric box |
| `NOVELTY_WEIGHTED`, `MAXIMIN_TABOO` | without repelling restarts (empty archive): they are `UNIFORM` |
| `sample_sigma=False` | with `ssa=SA` or `LPXNES` (forced on) |
| `sequential_selection` | with `PAIRWISE` and the default µ = λ/2 (its earliest stop, after 2µ + 2 points, is past the whole generation); with λ = 1 |
| `weights`, `elitist`, `ssa`, IPOP/BIPOP, `mirrored=PAIRWISE` | with λ = 1 (overridden) |
| `mirrored=MIRRORED` | with λ = 1: the pairs are formed within a generation, and its one individual is unpaired |
| `fast_sampling` | with a sampler other than `UNIFORM` + `GAUSSIAN` |
| `COVARIANCE` vs `COVARIANCE_NO_EIGV` | identical when `ssa` is not `CSA`, and with λ = 1 (converted) |
| all restart criteria but `UnableToAdapt` | with `restart_strategy=NONE` |
| `bound_correction` | in the coordinates with two infinite bounds |
| `threshold_convergence` | late in a run: its threshold decays to 0 with the budget |
| learning rates | `cs` with `ssa` `SR` or `SA` and a matrix adaptation that does not read the path p_σ (`NONE`, `CMSA`, `NATURAL_GRADIENT`; `CHOLESKY` reads it for h_σ since #160, except in a (1+1)-ES); `damps` except with `CSA`, `MSR`, `PSR`, `SR` and `TPA_INPLACE`; `c1`, `cmu` with `NONE` and `CMSA`, and with `NATURAL_GRADIENT` (which uses `cc`) without `active`; `cc` with `MATRIX`, `CMSA`, `NONE` |

These equivalences mean that the number of combinations overstates the number of distinct
algorithms.

## Combinations that change the meaning of an option

* **The path of CSA.** With `COVARIANCE` the path of the step size uses C^-1/2 Δm, with Δm the
  step of the mean divided, per coordinate, by the step size recombined from the selected points;
  every other matrix adaptation uses Σ wᵢ zᵢ ([deep dive](../deep-dives/step-size-adaptation.md#csa)).
  When the selected points share their step sizes (σ, raised in the coordinates of integer
  variables; with elitism and repaired points too) the two are the same up to a rotation; with
  sampled step sizes they differ. (Δm was divided by the mean of all step sizes of the
  population, which differed also with integer variables; #71.)
* **Pairwise selection.** Only the better point of each mirrored pair is ranked; the losers follow
  the ranked points, with their real values (`Population.n_ranked`). The losers get no negative
  weights in the active update (as pycma: mirrored points never get negative weights). The default
  weights give negative weights only to the worst µ ranks (or the worst λ − µ, for µ > λ/2), all
  beyond the λ/2 winners, so the active update has no effect with pairwise selection, whatever µ;
  with `EQUAL` or `EXPONENTIAL` weights and µ < λ/2 the winners from rank µ + 1 on get negative
  weights. `MSR` and `PSR` compare the ranked points only; the statistics and the restart criteria
  read every value.
* **µ other than ⌊λ/2⌋ with the default weights** (a given `mu0`, which IPOP and BIPOP scale with
  λ, and the default µ = λ/4 of `SA` and `CMSA`): the weights of a population of 2µ, as pycma with
  `CMA_mu` (#73): w′ᵢ = ln(µ + ½) − ln i, positive for the µ
  best; the µ negative ones go to the worst ranks, with weight 0 for the ranks in between when
  λ > 2µ, and only the last λ − µ of them when λ < 2µ. A generation that sequential selection stops
  before the first negative rank has no negative weights.
* **`weights=UTILITY`** recombines the mean from all λ points: the worse points move the mean too
  (away from themselves), so outliers matter, unlike with the other weights, where only the µ best
  do. The evolution paths are normalized with the utilities' selection mass (`mueff_mean`).
* **Elitism** is (µ+λ) selection: the µ parents compete with the offspring. Their steps are
  recomputed in the current distribution with the generation's σ. `MSR` and `PSR` then compare
  against the elite. With `PSR` the parents are in both generations it ranks; equal values share
  their mean rank, so a parent counts for neither generation (#120). `TPA` compares its two test
  points with each other, around the mean that the (µ+λ) selection gives; a generation that keeps
  every parent is below.
  * **With `active`** the negative weights of the matrix update go to the λ − µ worst points of
    the (µ+λ) pool, the points that the selection dropped (`Population.n_pool`), as the framework
    paper defines the active update ("the λ − µ worst individuals") and as pycma does with
    `CMA_elitist`; the mean, the paths and the step-size rules read the λ selected points as
    before. Ranks µ+1..λ of the pool, which had them before #161, are parents and good
    offspring: the negative weights removed variance along the flat directions, and `COVARIANCE`
    with `elitist` and `active` solved BBOB f10 and f12 (d = 10) in 0 of 25 runs.
  * **`XNES`** reads the offspring alone, ranked among themselves, in both forms (xNES SA and the
    natural gradient's σ part): xNES's gradient is an estimate from the samples of the current
    distribution. The parents' z, recomputed against the mean they recombined, are short
    (‖z‖²/d of 0.3 to 0.6), and as part of the µ best they kept σ 3 to 14 times too small (#172).
  * **`TPA_INPLACE`** ranks its two test points among the offspring, without the parents.
  * **`TPA` and `TPA_INPLACE`** count a generation whose selection kept every parent (no offspring
    among the µ selected) as a failure (#179). The mean did not move in it, so the next
    generation has nothing to test along the mean shift: `TPA` evaluates no test points there
    and takes r = −½ (`-a_tpa`), and `ask()` returns λ points; `TPA_INPLACE` samples ordinary
    offspring instead of its pair and takes z = −1. No TPA source covers elitism: Hansen (2008)
    and Hansen, Atamna & Auger (2014) define TPA for comma selection, and pycma's `CMA_elitist`
    adds only the best solution so far, so its mean always moves. The rule follows the success
    rules of elitist ESs (Rechenberg's 1/5th rule; the (1+1)-CMA-ES of Igel, Suttorp & Hansen
    2006), in which a generation without a successful offspring is a failure. Before #179 TPA
    compared two points that differed from m only by rounding, and σ diverged (to 1e205 to
    1e307; this page said it collapsed). On a sphere and two ellipsoids (d = 5, 10, 20; 20 runs
    each) `TPA` with `elitist` solved 15 of 180 runs and `TPA_INPLACE` 130, with `active` as well
    22 and 140; all four now solve 180. On BBOB (7 functions, d = 5 and 20, 25 runs) `TPA` with
    `elitist` solves 243 of 350 runs (56 before; `TPA` without elitism 249), `TPA_INPLACE` 251
    (174 before; 252 without).
  * **`weights=UTILITY`** recombines the mean from the λ best of the pool, parents included, with
    utilities that sum to zero. It was listed as a combination to avoid (the mean drifted away);
    most of that failure was the active update's negative weights on good points, which `UTILITY`
    switches on: on BBOB (7 functions, d = 5 and 20, 25 runs) `weights=UTILITY` with `elitist`
    solves 242 of 350 runs since #161, 179 before (the default configuration: 251), and
    `NATURAL_GRADIENT` with `elitist` and `active` (which uses `UTILITY`) 217, 0 before.
* **TPA's test points** are m ± ½σ⟨y⟩, half the last step of the mean with the step size that
  sampled it (Hansen 2008, Eqs. (8), (9)). Before #167 they were m ± σ_new⟨y⟩, about one step.
* **TPA with a bound correction.** Its two points m ± ½σ⟨y⟩ are not corrected by the method:
  whatever the method, both move toward m by the same factor until both are in the box, so TPA
  still compares two points at the same distance from m (#98). The method corrects them only
  when m itself is outside the box. Corrected one by one, as the samples, they lost runs: on BBOB
  in a box of ±4 (7 functions, d = 5 and 20), `TOROIDAL` and `UNIFORM_RESAMPLE` reached 1e-8 in
  16 and 13 fewer of 350 runs than with the points left outside the box.
* **`TPA_INPLACE`** (#173) samples offspring 1 and 2 as a mirrored pair along the last mean
  shift, from the second generation of a run on.
  * With `MIRRORED` and `PAIRWISE` they are the first pair. With `PAIRWISE` the worse of them is
    not ranked for the update, but the rank difference reads the values of all offspring, as pycma.
  * Sequential selection cannot stop a generation before both are evaluated (its earliest stop is
    after µ + 1 ≥ 2 points); the ranks are then among the evaluated offspring.
  * With orthogonal sampling the first draw of the block gives the length of the pair, and the
    pair is not orthogonal to the other samples.
  * With sampled step sizes (`sample_sigma`) the second point takes the step size of the first.
  * A bound correction corrects each point on its own, as pycma's boundary handler does, so a
    corrected pair is no longer symmetric; RESAMPLE and repelling sample the pair again as a pair.
  * With λ = 1 (a run whose λ a restart capped at the end of the budget) there is no pair and
    no update; `lambda0=1` runs `SR`.
  * With `elitist`, a generation after one whose selection kept every parent has no pair and
    counts as a failure (#179, under elitism above).
* **MATRIX and CHOLESKY with a step-size rule other than `CSA`** stall their rank-one update by
  h_σ, as `COVARIANCE` does (Hansen 2016, after Eq. (45); #160): `CHOLESKY` stalls p_c and
  compensates the variance in the decay; `MATRIX`, whose rank-one term reads p_σ, skips the term
  in that generation. With `CSA` neither has h_σ: MA-ES with CSA is as published (CSA updates σ
  from the same path, so a long path raises σ and shortens itself). Without it, `XNES` and `SA`,
  which keep σ small, left the path long, and the rank-one term stretched the matrix until the
  run stalled.
* **`ssa=XNES` with `active`** (#175): in a generation with h_σ = 0 the matrix update has no
  negative weights; it is the update without `active`. This holds for `COVARIANCE`,
  `COVARIANCE_NO_EIGV`, `SEPARABLE`, `CHOLESKY` and `MATRIX`, with the `DEFAULT`, `EQUAL` and
  `EXPONENTIAL` weights, but not with `UTILITY`.
  * **Why.** xNES SA keeps σ too small (Krause, Glasmachers & Igel 2017, Sec. 6), and h_σ was 0 in
    89–99 % of the generations at d ≥ 10. The samples then see a nearly linear function, and the
    ranking says nothing about the curvature. The negative weights (Σ|w⁻| = 1 + c1/cµ, 1.76 at
    d = 10, against 1 for the positive ones) took back most of the growth of C towards the optimum.
  * **Before #175**, on the ellipsoid (x0 uniform in [−3, 3]^d, σ₀ = 2, 10⁴·d evaluations, no
    restarts, 20 runs at d = 5, 10 and 20):
    * `COVARIANCE` + `XNES` + `active` solved 20, 1 and 0 runs, now 20, 20 and 20;
    * `CHOLESKY` solved 20, 1 and 0, now 20, 20 and 20;
    * `MATRIX` solved 20, 20 and 11, now 20, 20 and 20;
    * without `active`, all three solve 20 of 20 at each d.
  * **`UTILITY`** keeps its negative weights. They are flat (−1/λ) and only as heavy as the
    positive ones; it solved the ellipsoid before #175 (20, 20 and 20 runs), and without them it
    needed 15–20 % more evaluations.
  * **The source.** This is modcma's own rule, not a published one: the tutorial (Hansen 2016)
    and pycma keep the negative weights for every h_σ. The other step-size rules do the same here;
    they keep σ near its optimum and h_σ at 1 most of the time.
* **Sequential selection** stops a generation at the first point better than the best of the
  previous generation. With elitism that is the best of the run so far.
* **Orthogonal sampling** orthogonalizes the n independent samples of each generation (n = λ,
  ⌈λ/2⌉ with mirroring) in blocks of at most d that start with the generation. Samples that
  replace rejected ones (RESAMPLE, repelling) count towards the generation, so a generation with
  rejections is not completely orthogonal.
* **Mirrored sampling** pairs individuals 2i and 2i + 1 of each generation; with an odd λ the last
  one is unpaired. A pair that RESAMPLE or repelling rejects is sampled again as a pair. With
  sequential selection the partner of a batch's last individual is sampled with it, and discarded
  if the generation stops before it.
* **Threshold convergence** applies its threshold to z (E‖z‖ in the first generation, decaying with
  the budget); it lengthens the samples that length-based step-size rules (`XNES`, `MXNES`)
  measure.
* **Integer variables** are rounded after the bound correction, to the nearest integer in
  [⌈lb⌉, ⌊ub⌋], so a value in the box stays in it; `Settings` reject an integer variable without
  an integer in `[lb, ub]` ([integer variables](integer-variables.md#integer-variables-and-bounds)).
  Their step-size floor biases the step-size rules that read sampled step sizes (`SA`, `LPXNES`):
  the step size of a point is the geometric mean of its column of S, the floor included. The
  evolution paths do not see the floor: Δm is divided by the raised step size of each coordinate
  (as Marty et al. 2024).
* **`SR` with λ > 1** counts one success per generation, when its best beats the best of the run
  so far. SR is designed for elitist selection; without elitism the best of the run takes the
  parent's role. `MSR` is the success rule meant for non-elitist selection.
* **`restart_strategy=NONE`** still restarts when the matrix cannot be adapted (`UnableToAdapt`).
* **`mu0` with IPOP and BIPOP.** A given `mu0` keeps its ratio to λ at every restart (µ₀λ/λ₀,
  rounded); without it µ is the default of each new λ, λ/2 or λ/4, as in pycma
  ([restarts](restarts.md#restart-strategies)). pycma keeps a given `CMA_mu` as the same number.
* **The restart criteria per matrix adaptation** (#80). `TolX`, `NoEffectCoord` and the σ criteria
  are used with every matrix adaptation; the criteria that need the eigenvalues of C read the
  metric (`es.p.adaptation.metric()`), which only `COVARIANCE`, `SEPARABLE` and `NONE` keep up to
  date (`es.p.adaptation.keeps_metric()`):

    | criterion | `COVARIANCE`, `SEPARABLE`, `NONE` | `COVARIANCE_NO_EIGV`, `MATRIX`, `CHOLESKY`, `CMSA`, `NATURAL_GRADIENT` |
    |---|---|---|
    | `TolX`, `NoEffectCoord` | the diagonal of C (and p_c where there is one) | the same |
    | `MaxSigma`, `MaxDSigma`, `MinDSigma` | max(D), min(D): the eigenvalues of C | √max Cᵢᵢ, √min Cᵢᵢ: the diagonal of C. It bounds the eigenvalues, so `MaxDSigma` and `MinDSigma` fire earlier and `MaxSigma` later than with the eigenvalues. For `MaxDSigma` and `MinDSigma` no source does this; `MaxSigma` becomes pycma's `tolfacupx` (with 1e4 instead of 1e3) |
    | `ConditionC`, `NoEffectAxis` | used | not created: the metric would cost an O(d³) decomposition per update of the matrix. Added from Python (`es.p.criteria.items.append(modcma.restart.ConditionC())`), they work and pay that cost |

    With `ssa` other than `CSA`, and with λ = 1, `COVARIANCE` becomes `COVARIANCE_NO_EIGV`, and so
    loses `ConditionC` and `NoEffectAxis` (unless `always_compute_eigv=True`).
* **`RESAMPLE`** redraws at most 5d times, then evaluates the point as it is. Such a point can be
  selected and move the mean, but it never becomes `es.p.stats.global_best`, the best point that
  `run()` and `fmin` return: with a bound correction only a point in [lb, ub] does (#168). With a
  large σ₀ against the box (σ₀ = 2, box [−1, 0.5]^d) RESAMPLE gives up for most points from
  d = 5 on; the best point is then the best one in the box, at worst the final center of mass,
  which is clipped into it. `stats.current_best` and `stats.solutions` (the best of each run, which
  the success rules and the restarts read) include every point.
* **`COVERAGE` repelling's radius** is in units of σ: the distance to a taboo point is divided by
  the current σ (the paper's Eq. (8)), and the radius comes from the volume of the box in
  σ₀ units, (S n_T / (c σ₀^d N))^(1/d) Γ(d/2 + 1)^(1/d) / √π (Eqs. (10), (11), with σ₀^d; #158).
  σ₀ is the problem's, also in BIPOP's small runs, which start at σ₀ 10^(−2U). The radius grows with
  the width of the box over σ₀: a large box with a small σ₀ gives a radius of many σ, which rejects
  every sample near a taboo point (`TooMuchRepelling` then ends the run). For large d it tends to
  √(d / (2πe)) (width / σ₀), about 0.24 √d times the ratio. With σ₀ = 2 in [−5, 5]^d (the
  defaults) it is 2.7 at d = 5, 4.0 at d = 10 and 5.6 at d = 20 for a point found in one of two
  runs, about 1.25 √d; a sample lies about √d σ from the mean (in the Mahalanobis distance of
  `CURRENT`, the default), so while the mean is near a taboo point most samples are rejected.
  Computed in logarithms, so it stays finite at large d.
* **`C0`** is the matrix of the first generation of every run, the first and each restart. Every
  matrix adaptation but `NONE` starts from a factor of it
  ([deep dive](../deep-dives/matrix-adaptations.md#the-initial-matrix-c0)); `TolX`, `ConditionC`
  and the other criteria that read C see it from the first generation on.

## Gradient injection

A gradient (`run(f, gradient=...)`, `inject_gradient`; [the guide](gradients.md)) is not a module:
it works with every configuration. The population is sampled as without it (the same random
numbers), then the gradient point replaces individual 0:

| with | the gradient point |
|---|---|
| `mirrored` `MIRRORED` or `PAIRWISE` | replaces the pair 0 and 1: the gradient point and its mirror (the opposite step), so no pair is broken. With `PAIRWISE` the better of the two is ranked |
| `orthogonal` | replaces an orthogonal sample: its block is no longer orthogonal |
| a bound correction | moved into the box as TPA's points (`RESAMPLE` clips it), then treated as a point given to `tell(f, X)`: its y and z from the corrected point, clipped at the Mahalanobis length √d + 2d/(d + 2) |
| `integer_variables` | rounded, as the samples |
| sampled step sizes (`sample_sigma`, `SA`, `LPXNES`) | the step size of individual 0 (σ otherwise; the integer variables' lower bound applies) |
| `sequential_selection` | in the first batch: always evaluated |
| `ssa=TPA` | TPA's two points are extra points, not samples: they stay |
| `elitist` | an offspring: it competes with the parents as the samples do |
| `threshold_convergence` | not applied: the point has the length E‖z‖ |
| restarts | every population of every run gets one, at its run's mean; the centers of mass and the hill–valley points do not |
| `repelling_type` | not injected when it lies in a taboo region (the sample stays), as a sample there is rejected |
| λ = 1, or λ = 2 with mirroring | the whole population: the run depends on the gradient alone |
| `sample_transformation=CAUCHY` | its typical length (≈ 1.18 d at large d) exceeds the clipping length from d = 3 on: the point is evaluated at that length and enters the update clipped |

The point's length c is the expected length of a sample z of the sampler (`Weights.expected_length_z`):
E‖N(0, I)‖ for the Gaussian samplers.

## Combinations to avoid

These combinations run, but work badly. Unless a row says otherwise, measured on a shifted sphere
(d = 5, 2000 evaluations) when the page was written (2026-11-24), and confirmed on more functions
where noted. They are not bugs to fix; `ADAPTIVE` repelling is studied in #177.

| combination | what happens |
|---|---|
| `sample_transformation=NONE` | the raw base sampler, for testing: uniform samples with mean 0.5, which fail everywhere |
| `sample_transformation=CAUCHY` with `XNES`, `MXNES`, `LPXNES`, `SR`, with `EQUAL` weights, with `MATRIX`, `NATURAL_GRADIENT` or `PAIRWISE` | σ diverges or collapses: the step-size rules assume finite variance. 0 of 5 runs reach 1e-8 with each, on the shifted sphere (d = 5, 2000 evaluations) and a rotated ellipsoid (condition 10⁶, d = 5, 5000 evaluations), where `CAUCHY` alone solves 5 of 5 (re-measured for #187, 2026-11-28). `MSR` solves 5 of 5 on both since #83 |
| `ssa=SA` with the default λ on ill-conditioned functions, d = 20 | σ shrinks before the matrix adapts: BBOB f2 and f8 (d = 20) reach 1e-8 in 0 of 25 runs each within 3000·d evaluations (`CSA`: 25 and 22); λ = 40 or 80 did not help within 60,000 evaluations (4 runs each). The step sizes per coordinate before #71 hid this on separable functions (25 of 25 on f2), at the cost of rotation invariance (0 of 25 on f10 and f12) |
| `PAIRWISE` with `EQUAL` weights | C degenerates and the search stalls: on the shifted sphere (d = 5) the condition of C reaches 5 × 10² to 10⁵ within 2000 evaluations (3 with `PAIRWISE` alone), and 0 of 5 runs reach 1e-8 with 2000 or 20,000 evaluations, against 5 of 5 with the `DEFAULT` or `EXPONENTIAL` weights (re-measured for #187, 2026-11-28) |
| `ssa=XNES` with `weights=EQUAL` | the runs fail, with or without `active` (#180). Ellipsoid with `COVARIANCE` (x0 uniform in [−3, 3]^d, σ₀ = 2, 10⁴·d evaluations, 20 runs): 0 runs reach 1e-8 at d = 5, 10 and 20, with 68–207 restarts per 20 runs. Sphere at d = 10: 1 of 5 runs. `MXNES` and `CSA` with `EQUAL` solve all of these. Measured on master 091c577 by the investigation of #175. The cause has not been measured; the investigation's explanation: with equal weights over the µ = λ/2 best, xNES SA's signal Σ wᵢ(‖zᵢ‖² − d) has no drift on a linear function, since the better and the worse half have the same distribution of ‖z‖². Nothing raises σ then, and the curvature lowers it |
| `repelling_type=ADAPTIVE` | experimental: modcma's own work in progress, without a published source. In the experiment of #158 (BBOB f21 and f22, d = 5 to 40, `RESTART` and `IPOP`) it solved 99 of 240 runs, against 107 without repelling and 144 with `COVERAGE` |
| `matrix_adaptation=NATURAL_GRADIENT` with `ssa=LPXNES` at small d | LPXNES keeps its own rate (cs = 1 at d = 5; before #72 NATURAL_GRADIENT gave it xNES's 0.25): BBOB at d = 5 (7 functions, 25 runs each) solves 66 of 175 runs, 118 on master 20bdf27; at d = 20 79 of 175, 0 on master |
| `weights=UTILITY` with `bound_correction=TOROIDAL` or `sample_transformation=CAUCHY` | far points (wrapped around, heavy tails) pull the mean through their negative utilities |
| `lambda0=2` | allowed for experiments, but many modules fail. On a rotated ellipsoid (d = 5, 5 seeds), 0 of 5 runs converge with `elitist`, `MATRIX`, `CMSA`, `MSR`, `PSR`, `SR`, `MXNES`, and with `restart_strategy` `RESTART` or `STOP`, whose criteria end the runs early. The default configuration needs about 6× the evaluations of λ = 4 |
| `lambda0=3` | close to λ = 4 for most modules (same measurement), except `sample_transformation=CAUCHY` (0 of 5) and `MXNES` (0 of 5 on the ellipsoid); `SR`, `CMSA` and `MSR` need about 2× the evaluations |
| a gradient (`run(f, gradient=...)`) with `lambda0=1`, or with λ = 2 and mirroring | the gradient point is the whole population: no sample is left, and the matrix learns only the gradient steps. On BBOB f10 with λ = 1 (d = 10 and 40, 15 runs each) no run reaches 1e-8, also with the exact gradient (median best f 1.5e5 and 3.9e5); without a gradient every run does ([gradient injection](gradients.md#what-a-bad-gradient-costs)) |

For tuning, `get_configspace` leaves `sample_transformation=NONE` out and has conditions for the
parameters without an effect; see [tuning](tuning.md). `Settings` still accepts `NONE`.
