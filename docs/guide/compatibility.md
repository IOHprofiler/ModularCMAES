# Compatibility of modules

Every combination of modules can be constructed and run, but not every combination is a different
algorithm, and a few are known to work badly. This page collects how modules interact: the
combinations the `Settings` adjust, the options that have no effect in some combinations, the ones
whose meaning changes, and the combinations to avoid. The [deep dives](../deep-dives/matrix-adaptations.md)
explain how the modules compare with the published algorithms they come from.

The lists come from the code and from a test of every module value, every pair of values and
600 random configurations (shifted sphere and Rastrigin; 2026-11). Items marked *known issue* are
recorded for fixing (TASKS.md, WP23; the identifier in brackets).

## Combinations that are adjusted

The `Settings` change some choices when they are created; `es.p.settings.modules` shows what is
used.

| when | then |
|---|---|
| `ssa` other than `CSA` with `matrix_adaptation=COVARIANCE` | `COVARIANCE_NO_EIGV` (unless `always_compute_eigv=True`): the same covariance update, sampled through a Cholesky factor |
| `ssa=SA` or `matrix_adaptation=CMSA` | µ defaults to λ/4 (an explicit `mu0` wins) |
| `mu0 > lambda0` | µ = λ/2 |
| `matrix_adaptation=NONE` | `active` off |
| `mirrored=PAIRWISE` and an odd λ | λ + 1 |
| `lambda0=1` | the (1+1)-ES: µ = 1, elitist, equal weights, `ssa=SR`, no active update or sequential selection; IPOP and BIPOP become `RESTART` (*known issue* with PAIRWISE: NaN weights [F17]) |
| `ssa=SA` or `LPXNES` | sampled step sizes (`sample_sigma`) are always on |
| `weights=UTILITY` | `active` on (the utilities of the worse half are negative; every matrix update uses all of them) |
| `matrix_adaptation=NATURAL_GRADIENT` with `active` and the default weights | `weights=UTILITY`: xNES's active form (CMA-ES's negative weights broke it) |
| `mirrored=PAIRWISE` with `sequential_selection` | the generation stops only after complete pairs, at 2µ points at the earliest |
| `matrix_adaptation=NATURAL_GRADIENT` | the learning rate `cs` of the step-size rule is xNES's (*known issue*: for every step-size rule, not only `XNES` [F3]) |
| `matrix_adaptation=SEPARABLE` | `cmu` × (d + 2)/3 (Ros & Hansen 2008) |
| infinite or half-infinite bounds | no bound correction at all (*known issue*: also on the finite side of a half-infinite box [F20]) |

## Options without effect

| option | no effect when |
|---|---|
| `active` | `matrix_adaptation=NONE`; λ = 1; µ = λ (no negative weights) |
| `repelling_type`, `repelling_metric` | there are no restarts (`restart_strategy` `NONE` or `STOP`); `repelling_metric` also without `repelling_type`, and `CURRENT`/`ARCHIVED` equal `EUCLIDEAN` with `matrix_adaptation=NONE` |
| `center_placement` | without restarts: it only places restarts, the first run starts at `x0` (or the center of the bounds). `X0` and `CENTER` coincide without `x0`, `ZERO` and `CENTER` for a symmetric box |
| `NOVELTY_WEIGHTED`, `MAXIMIN_TABOO` | without repelling restarts (empty archive): they are `UNIFORM` |
| `sample_sigma=False` | with `ssa=SA` or `LPXNES` (forced on) |
| `sequential_selection` | with `PAIRWISE` and the default µ = λ/2 (the earliest stop, 2µ = λ, is the whole generation); with λ = 1 |
| `weights`, `elitist`, `ssa`, IPOP/BIPOP | with λ = 1 (overridden) |
| `fast_sampling` | with a sampler other than `UNIFORM` + `GAUSSIAN` |
| `COVARIANCE` vs `COVARIANCE_NO_EIGV` | identical when `ssa` is not `CSA` (converted) |
| restart criteria `TolX`, `ConditionC`, `NoEffectAxis`, `NoEffectCoord` | with any matrix adaptation but `COVARIANCE` (so also when `ssa` is not `CSA`) [F25] |
| all restart criteria but `UnableToAdapt` | with `restart_strategy=NONE` |
| `bound_correction` | with infinite bounds |
| `threshold_convergence` | late in a run: its threshold decays to 0 with the budget |
| learning rates | `cs` with `ssa=SA`; `damps` except with `CSA` and `SR`; `c1`, `cmu` with `CMSA` and `NATURAL_GRADIENT` (which uses `cc`); `cc` with `MATRIX`, `CMSA`, `NONE` |

These equivalences mean that the number of combinations overstates the number of distinct
algorithms.

## Combinations that change the meaning of an option

* **The path of CSA.** With `COVARIANCE` the path of the step size uses C^-1/2 Δm; with every other
  matrix adaptation the weighted sum of the selected z ([deep dive](../deep-dives/step-size-adaptation.md#csa)).
  They differ wherever the evaluated points are not m + σAz: sampled step sizes, elitism, injected
  or repaired points, the step-size floor of integer variables.
* **Pairwise selection.** The worse point of each mirrored pair gets f = +inf before the update.
  The losers get no negative weights in the active update (as pycma: mirrored points never get
  negative weights). With the default µ = λ/2 the active update therefore has no effect with
  pairwise selection; `MSR` compares with an infinite median (*known issue*: σ only shrinks [F9]); `PSR` uses
  only the finite values.
* **`weights=UTILITY`** recombines the mean from all λ points: the worse points move the mean too
  (away from themselves), so outliers matter, unlike with the other weights, where only the µ best
  do. The evolution paths are normalized with the utilities' selection mass (`mueff_mean`).
* **Elitism** is (µ+λ) selection: the µ parents compete with the offspring. Their steps are
  recomputed in the current distribution with the generation's σ. `MSR`, `PSR` and `TPA` then
  compare against the elite.
* **Sequential selection** stops a generation at the first point better than the best so far *over
  all runs*: after a restart it hardly ever triggers (*known issue* [F13]).
* **Orthogonal sampling** orthogonalizes blocks of max(d, n) samples, with n = λ, λ/2 with
  mirroring, minus 2 with TPA; when n < d a block spans several generations (*known issue* [F11]).
  Rejected samples (RESAMPLE, repelling) consume block samples.
* **Mirrored sampling** keeps its state between generations and through rejected samples: with
  RESAMPLE, repelling, an odd λ or sequential selection the pairs fall out of step (*known issue*
  [F10]).
* **Threshold convergence** applies its threshold (in units of x: a tenth of the box diameter) to
  z, before A and σ; it lengthens the samples that length-based step-size rules (`XNES`, `MXNES`)
  measure.
* **Integer variables** are rounded after the bound correction (*known issue*: a rounded value can
  leave the box [F21]); their step-size floor biases the step-size rules that read sampled step
  sizes (`SA`, `LPXNES`).
* **`SR` with λ > 1** counts one success per generation, when its best beats the best of the run
  so far. SR is designed for elitist selection; without elitism the best of the run takes the
  parent's role. `MSR` is the success rule meant for non-elitist selection.
* **`restart_strategy=NONE`** still restarts when the matrix cannot be adapted (`UnableToAdapt`).
* **`RESAMPLE`** redraws at most 5d times, then evaluates the point as it is.

## Combinations to avoid

Measured on a shifted sphere (d = 5, 2000 evaluations) and confirmed on more functions where
noted; with the current version. Most are recorded for fixing.

| combination | what happens |
|---|---|
| `sample_transformation=NONE` | uniform samples with mean 0.5: fails everywhere [F22] |
| `sample_transformation=CAUCHY` with `XNES`, `MXNES`, `LPXNES`, `MSR`, `SR`, with `EQUAL` weights, with `MATRIX`, `NATURAL_GRADIENT` or `PAIRWISE` | σ diverges or collapses: the step-size rules assume finite variance |
| `active` with `CMSA` | C becomes indefinite: restarts every few generations [F6] |
| `sample_sigma`, `SA` or `LPXNES` on rotated, ill-conditioned problems | one sampled step size per coordinate breaks rotation invariance: CMSA with `SA` fails on a rotated ellipsoid [F1] |
| `MSR` with `PAIRWISE` | σ shrinks every generation [F9] |
| `TPA` with `elitist` | σ diverges [F15] |
| `MATRIX` with `elitist` | the inverse of M drifts, the elites' steps are wrong [F18] |
| `mu0` above λ/2 with the active update | some "negative" weights are positive [F8] |
| `PAIRWISE` with `EQUAL` weights | C degenerates in some runs |
| `orthogonal` with `EQUAL` weights and a step-size rule other than `CSA` | C shrinks while σ grows |
| `active` + `elitist` with a step-size rule other than `CSA` | poor convergence; with `CHOLESKY` or `SEPARABLE` and `XNES`/`MXNES`, σ diverges |
| `UNIFORM`, `NOVELTY_WEIGHTED`, `MAXIMIN_TABOO` with infinite bounds | NaN restart centers [F19] |
| `ssa=XNES` without `NATURAL_GRADIENT`, or with `matrix_adaptation=NONE` | slow (a small default learning rate) [F27] |
| `weights=UTILITY` with `elitist` | the mean drifts away: the utilities contrast the better and worse halves around the population's center, which with elitism is not the distribution's |
| `weights=UTILITY` with `bound_correction=TOROIDAL` or `sample_transformation=CAUCHY` | far points (wrapped around, heavy tails) pull the mean through their negative utilities |

For tuning, `get_configspace` currently has no conditions for inactive options and passes the
numeric defaults of CSA with COVARIANCE explicitly, which overrides the module-specific defaults
(*known issue* [F23]); restrict the space accordingly.
