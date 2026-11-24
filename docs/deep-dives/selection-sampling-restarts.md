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

modcma keeps all λ survivors of the previous selection and selects the best λ of the 2λ: (λ+λ)
selection, which for the positive weights equals (µ+λ). The survivors' steps y = (x − m)/s are
recomputed in the current distribution, each with the step size it was sampled with (the framework
paper's Eq. 1–3, with the generation's σ). pycma's `CMA_elitist` is different: it re-injects only
the best solution so far, when it beats the generation's best, rescaled to a random length.

## Sequential selection

A generation stops at the first point (from the µ-th on, after complete pairs with pairwise
selection) that is better than the best so far (Brockhoff et al. 2010). The best so far is the best
of all runs, so after a restart the rule hardly triggers (*known issue* [F13]; the framework paper:
better than the previous generation).

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

A sample z shorter than the threshold t = 0.1·D·((B − e)/B)^0.995 (D the diameter of the box, B the
budget, e the evaluations so far) is reflected to length 2t − ‖z‖. Piad-Morffis et al. (2015) and
the framework paper define the threshold on ‖x − m‖ and decay it with the generations; modcma
applies it to z, before A and σ, and decays it with the evaluations [F24].

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

* **IPOP** (Auger & Hansen 2005): λ doubles at every restart (µ in the same ratio), up to (d λ₀)²
  (the framework paper: 2⁹ λ₀); σ and the mean are reset. No budget per run: the restart criteria
  end a run.
* **BIPOP** (Hansen 2009): after the first run, the remaining budget is split into two halves, one
  for the regime of large populations (λ doubling) and one for small ones,
  λ_s = ⌊λ₀ (λ_large/(2λ₀))^(U²)⌋ with σ = σ₀·10^(−2U); the regime with more budget left runs next.
  pycma charges the first run to the small regime and caps the generations of small runs; modcma does
  neither, and rounds λ_s up to an even number [F26].

## Restart criteria and pycma's termination criteria

With `restart_strategy` `RESTART`, `STOP`, `IPOP` or `BIPOP` these criteria end a run (`NONE` uses
only `UnableToAdapt`). Defaults in brackets.

| modcma | pycma | difference |
|---|---|---|
| `ExceededMaxIter` (100 + 50(d + 3)²/√λ generations) | `maxiter` 100 + 150(d + 3)²/√λ | a third of pycma's |
| `NoImprovement` (range of the generations' best over 10 + ⌈30d/λ⌉ generations ≤ 1e-20) | `tolfunhist` 1e-12 | tolerance; modcma has no `tolfun` (current and recent values, 1e-11) [WP17] |
| `MaxSigma` (σ > 1e4) | `tolfacupx` (σ√Cᵢᵢ > 1e3 σ₀) | absolute instead of relative [F25] |
| `MinSigma` (σ < 1e-20) | none | |
| `MaxDSigma` / `MinDSigma` | `tolupsigma` | `MinDSigma` is modcma's |
| `TolX` (σ√Cᵢᵢ and σ\|p_c,ᵢ\| < 1e-11 σ₀) | `tolx` 1e-11 | relative to the run's σ₀, as Hansen (2016) |
| `ConditionC` (cond(C) > 1e14) | `tolconditioncov` 1e14 | the same |
| `NoEffectAxis`, `NoEffectCoord` | `noeffectaxis`, `noeffectcoord` | absolute instead of relative to m [F25] |
| `FlatFitness` | `tolflatfitness` | different quantile |
| `Stagnation` | `tolstagnation` | different windows of the history |
| `UnableToAdapt` | (none) | the matrix could not be updated |
| `TooMuchRepelling` | (none) | repelling restarts |

`TolX`, `ConditionC`, `NoEffectAxis` and `NoEffectCoord` read the eigendecomposition and are only
used with `COVARIANCE` (so only with CSA) [F25]. pycma's `tolfun`, `tolfunrel`, `tolxstagnation`
and `timeout` are planned (WP17).

## Repelling restarts and center placement

Repelling restarts follow de Nobel et al. (2024) (`COVERAGE`); `ADAPTIVE`, which adapts the
radius to the rate of rejected samples, is modcma's own. Center placement only places restarts:
the first run starts at `x0`, or at the center of the bounds.
