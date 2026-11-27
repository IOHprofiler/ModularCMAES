# Selection, sampling and restarts compared with the published methods

## Recombination weights

* `DEFAULT`: wᵢ ∝ ln((λ + 1)/⌊λ/µ⌋) − ln i, which for µ = λ/2 are the weights of Hansen (2016) and
  pycma. *Known issue* [F8]: for µ > λ/2 some of the "negative" weights come out positive.
* `EQUAL`: 1/µ (intermediate recombination); the negative ones −1/µ.
* `EXPONENTIAL`: wᵢ = 2⁻ⁱ + 1/(µ 2^µ), summing to 1 (the framework paper).
* `UTILITY`: the utilities of NES/xNES (Wierstra et al. 2014; Glasmachers et al. 2010),
  uₖ = max(0, ln(λ/2 + 1) − ln k)/Σⱼ max(0, ln(λ/2 + 1) − ln j) − 1/λ, summing to zero; identical to
  `cmaes`' XNES. The mean moves by Σ uₖ(xₖ − m) over all λ points, the paths are normalized with
  1/Σuₖ², every matrix update uses all of them, unscaled (`active` is set), and the modules that need
  convex weights (SA, LPXNES) use their non-negative part. With sequential selection the utilities of
  the n evaluated points are used.

The negative weights (active update) are scaled so that their absolute values sum to
min(1 + c1/cµ, 1 + 2µ_eff⁻/(µ_eff + 2), (1 − c1 − cµ)/(d cµ)) (Hansen 2016). The mean is always
recombined from the µ positive weights.

## The active update, per matrix adaptation

| matrix adaptation | negative weights | rescaled by d/‖A⁻¹y‖² | decay accounts for them | in tests |
|---|---|---|---|---|
| `COVARIANCE`, `COVARIANCE_NO_EIGV` | yes | yes (Hansen 2016) | yes | faster (about 30 % on an ellipsoid) |
| `SEPARABLE` | yes | no | yes | faster |
| `MATRIX` | yes | no | no [F7] | faster |
| `CHOLESKY` | as rank-one downdates | no | no [F7] | faster; failed downdates undetected [F5] |
| `CMSA` | yes | no | no | breaks [F6] |
| `NATURAL_GRADIENT` | all λ, the utilities (`active` selects `UTILITY`) | no (unscaled, as xNES) | (trace-free) | xNES |
| `NONE` | switched off | | | |

With pairwise selection the losers (f = +inf) receive the negative weights in index order, and a
loser's step is the winner's mirrored, so its negative update cancels the winner's positive one on
the same direction; pycma never gives mirrored points negative weights.

## Elitism

(µ+λ) selection, as in the framework paper (Sec. 4.1): the µ parents, the µ best of the previous
selection, compete with the λ offspring.

* The parents' steps are recomputed in the current distribution, as for a repaired point
  (the paper's Eq. 1–3): y = (x − m)/σ, with the σ of the generation, and z = A⁻¹y.
* Integer variables keep their lower bound on the step size.

Before 2.0, modcma kept all λ survivors, a (λ+λ) selection, each with the step size it was sampled
with.

pycma's `CMA_elitist` is different: it re-injects only the best solution so far, when it beats the
generation's best, rescaled to a random length.

## Sequential selection

A generation stops at the first point (from the µ-th on, after complete pairs with pairwise
selection) that is better than the best of the previous generation (Brockhoff et al. 2010; the
framework paper). The first generation of a run has no previous generation and is evaluated
completely.

Before 2.0 the reference was the best of all runs, so after a restart the rule hardly
triggered.

## Mirrored and pairwise selection

Mirrored sampling is unconditional: z, −z, z', −z', … (Brockhoff et al. 2010); pycma's default is
selective mirroring (only the worst points are mirrored, in the next generation). With `PAIRWISE`
only the better point of each pair is used (Auger et al. 2011): the worse one gets f = +inf.
*Known issue* [F10]: the pairs fall out of step after a rejected sample or with an odd λ.

## Orthogonal sampling

The samples are orthonormalized in blocks of at most d by a Householder QR decomposition and keep
the lengths of the samples they came from (Wang et al. 2014 use Gram–Schmidt; the result is the
same). *Known issue* [F11]: when λ (λ/2 with mirroring) is below d, a block spans several
generations, so the samples of one generation are not all orthogonal.

## Threshold convergence

A sample z shorter than the threshold is reflected to length 2t − ‖z‖; y = Az is scaled with it.

* The threshold is t = α·E‖z‖·((B − e)/B)^γ, with α = 1 and γ = 0.995 by default, B the budget and
  e the evaluations at the start of the generation.
* It is computed once per generation, so it decays with the generations (for a constant λ, (B − e)/B
  is the share of the generations left). It is 0 once the budget is used up.

**Compared with Piad-Morffis et al. (2015) and the framework paper.** They put the threshold on
‖x − m‖, as a fraction of the box diameter D (0.1·D). With those defaults a run cannot converge
before the budget is nearly used up: on BBOB (f1, f2, f8, f10, f12, f15, f21; d = 5, 20) 8 of 350
runs reached 1e-8, against 255 with the threshold on z.

On z the threshold is relative to the search distribution (σ and C), so it does not depend on the
box. Before 2.0, modcma applied 0.1·D to z, which equals E‖z‖ on the default box [−5, 5]^d (the
same behaviour as now) but stretched every sample on larger boxes.

## Samplers and transformations

* `SOBOL`: a fixed set of N Sobol points (N·d ≈ 2·10⁶, at least 128) drawn in a random order; the
  set repeats after N samples (d = 1000: after about 85 generations) [F29]. `HALTON`: a scrambled
  Halton sequence, which does not repeat.
* The transformations other than Gaussian have unit variance (Cauchy: unit scale) and are mapped
  from the uniform samples through their inverse CDF (de Nobel et al. 2025). Step-size rules that
  assume a finite variance fail with Cauchy samples ([compatibility](../guide/compatibility.md#combinations-to-avoid)).
* `fast_sampling` replaces mt19937 and the inverse normal CDF by xoshiro256++ and a ziggurat, only
  for `UNIFORM` + `GAUSSIAN`.

## IPOP and BIPOP

* **IPOP** (Auger & Hansen 2005): λ doubles at every restart (µ in the same ratio), up to 2⁹ λ₀,
  as in the framework paper (`IPOP.max_lambda_factor`; before 2.0, (d λ₀)²). σ and the mean are
  reset. No budget per run: the restart criteria
  end a run. Every run ends with an evaluation of its center of mass, the last one too, as pycma's
  `eval_final_mean`; the budget always keeps that evaluation (pycma can exceed `maxfevals` by one).
  The first generation after a restart takes at most the rest of the budget, less the new run's
  center of mass (λ, and µ in proportion, are capped; at least 2 points); the hill-valley tests of
  repelling restarts stop when the budget runs short.
* **BIPOP** (Hansen 2009): two regimes, one of large populations (λ doubling) and one of small
  ones, λ_s = ⌊λ₀ (λ_large/(2λ₀))^(U²)⌋ with σ = σ₀·10^(−2U′), where λ_large is the λ of the next
  large run (twice the last one). Each regime counts the evaluations of its runs; a restart uses
  the regime that has used fewer, the large one on a tie, so the first restart is large. As in
  pycma's `fmin`, the first run counts as small (Hansen 2009 disregards it), and a small run takes
  at most half the evaluations of the large regime so far: `restart.BIPOP` sets
  `max_iter` = ½ (large evaluations)/λ_s generations, which `ExceededMaxIter` enforces (pycma's
  `maxiter`). λ_s is rounded down, as in the paper and pycma; up to even only with `PAIRWISE`.
  The population sizes are pycma's (4.6.0, `fmin`, `cma/evolution_strategy.py`:5362–5381): λ_s
  lies between λ₀ and the last large population, because pycma counts the small run
  (`runs_with_small += 1`) before it computes the multiplier, which is then the last large run's.
  Hansen (2009) writes the same formula with λ_ℓ "the latest population size from the first
  regime", and the range [λ_def, λ_ℓ/2], half of pycma's; modcma follows pycma (#140). The large
  population has no upper limit. pycma's `fmin` has none either, but stops after `restarts` large
  runs (9 with `restarts=True`), and Hansen (2009) after nine (λ up to 2⁹ λ_def); modcma restarts
  until the budget is used, and only IPOP stops growing, at 2⁹ λ₀. With its default popsize,
  pycma multiplies the unrounded 4 + 3 ln d (17, 35, 70, … at d = 5), modcma and the paper the
  integer λ₀. A C++ test (`tests/cpp/test_main.cpp`) compares λ, µ, σ and the cap of every run
  with a transcription of pycma's rule, for a fixed random stream.
  Before 2.0, the budget left after the first run was split into halves that each run decreased
  (unsigned: a run longer than its regime's rest wrapped it around, and BIPOP became IPOP), λ_s
  was always even, and small runs had no cap (#80).

## Restart criteria and pycma's termination criteria

With `restart_strategy` `RESTART`, `STOP`, `IPOP` or `BIPOP` these criteria end a run. `NONE` uses
only `UnableToAdapt`, and still restarts when it fires: a matrix that could not be updated cannot
be used further. (The framework paper describes `NONE` as continuing; the code restarts.) Defaults
in brackets.

| modcma | pycma | difference |
|---|---|---|
| `ExceededMaxIter` (100 + 50(d + 3)²/√λ generations) | `maxiter` 100 + 150(d + 3)²/√λ | a third of pycma's |
| `NoImprovement` (range of the generations' best over 10 + ⌈30d/λ⌉ generations ≤ 1e-20) | `tolfunhist` 1e-12 | tolerance; modcma has no `tolfun` (current and recent values, 1e-11) [WP17] |
| `MaxSigma` (σ max(D) grew by more than 1e4 in the run) | `tolfacupx` (σ√Cᵢᵢ > 1e3 σ₀) | Hansen (2016)'s TolXUp: the largest axis instead of each coordinate; before 2.0, σ > 1e4 |
| `MinSigma` (σ < 1e-20) | none | |
| `MaxDSigma` / `MinDSigma` | `tolupsigma` | `MinDSigma` is modcma's |
| `TolX` (σ√Cᵢᵢ and σ\|p_c,ᵢ\| < 1e-11 σ₀) | `tolx` 1e-11 | relative to the run's σ₀, as Hansen (2016) |
| `ConditionC` (cond(C) > 1e14) | `tolconditioncov` 1e14 | the same |
| `NoEffectAxis`, `NoEffectCoord` | `noeffectaxis`, `noeffectcoord` | the same: m + 0.1σDᵢbᵢ == m, mᵢ + 0.2σ√Cᵢᵢ == mᵢ in floating point (before 2.0: the step below 10 ε) |
| `FlatFitness` | `tolflatfitness` | different quantile |
| `Stagnation` | `tolstagnation` | different windows of the history |
| `UnableToAdapt` | (none) | the matrix could not be updated |
| `TooMuchRepelling` | (none) | repelling restarts |

Every matrix adaptation keeps the diagonal of C (`coordinate_wise_variances`), which `TolX` and
`NoEffectCoord` read. D and B come from `Adaptation.metric()`: free for `COVARIANCE`,
`SEPARABLE` and `NONE`, which get `ConditionC` and `NoEffectAxis`; the others would decompose their
factor every generation (O(d³)), so they do not get them by default, and `MaxSigma`, `MaxDSigma`
and `MinDSigma` use √max Cᵢᵢ and √min Cᵢᵢ instead of max(D) and min(D)
([compatibility](../guide/compatibility.md#combinations-that-change-the-meaning-of-an-option)).
Before 2.0 only `COVARIANCE` had the four criteria (and so only with CSA). pycma's `tolfun`,
`tolfunrel`, `tolxstagnation` and `timeout` are planned (WP17).

## Repelling restarts and center placement

Repelling restarts follow de Nobel et al. (2024) (`COVERAGE`); `ADAPTIVE`, which adapts the
radius to the rate of rejected samples, is modcma's own. Center placement only places restarts:
the first run starts at `x0`, or at the center of the bounds.
