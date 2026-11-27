# Selection, sampling and restarts compared with the published methods

## Recombination weights

* `DEFAULT`: wᵢ ∝ ln((λ + 1)/2) − ln i for the default µ = ⌊λ/2⌋, the weights of Hansen (2016,
  Eq. (49)) and pycma. For another µ (`mu0`, or IPOP's ratio with an odd λ₀) the weights of a
  population of 2µ, ln(µ + ½) − ln i, as pycma with `CMA_mu` (#73): the µ negative ones go to the
  worst ranks, with 0 for the ranks in between when λ > 2µ, and only the last λ − µ of them when
  λ < 2µ (the scaling below is that of all µ). They agree with pycma 4.6.0's `RecombinationWeights`
  to 2e-16 for (µ, λ) = (3, 10), (6, 10), (5, 10), (2, 20). Before, the weights came from
  ln((λ + 1)/⌊λ/µ⌋) − ln i, and for µ > λ/2 some of the "negative" weights were positive.
* `EQUAL`: 1/µ (intermediate recombination); the negative ones −1/µ.
* `EXPONENTIAL`: wᵢ = 2⁻ⁱ + 1/(µ 2^µ), summing to 1 (the framework paper).
* `UTILITY`: the utilities of NES/xNES (Wierstra et al. 2014; Glasmachers et al. 2010),
  uₖ = max(0, ln(λ/2 + 1) − ln k)/Σⱼ max(0, ln(λ/2 + 1) − ln j) − 1/λ, summing to zero; identical to
  `cmaes`' XNES. The mean moves by Σ uₖ(xₖ − m) over all λ points, the paths are normalized with
  1/Σuₖ², every matrix update uses all of them (`active` is set), and the modules that need
  convex weights (SA, CMSA, and XNES without `NATURAL_GRADIENT` and LPXNES) use their non-negative
  part, normalized to sum 1. With sequential selection the utilities of
  the n evaluated points are used.

The negative weights (active update) are scaled so that their absolute values sum to
min(1 + c1/cµ, 1 + 2µ_eff⁻/(µ_eff + 2), (1 − c1 − cµ)/(d cµ)) (Hansen 2016, Eq. (51)–(53)). The
mean is recombined from the µ positive weights (from all λ utilities with `UTILITY`).

## The active update, per matrix adaptation

| matrix adaptation | negative weights | rescaled by d/‖A⁻¹y‖² | decay accounts for them | in tests |
|---|---|---|---|---|
| `COVARIANCE`, `COVARIANCE_NO_EIGV` | yes | yes (Hansen 2016) | yes | faster (about 30 % on an ellipsoid) |
| `SEPARABLE` | yes | yes, with the diagonal C (as pycma) | yes | faster (about 27 %) |
| `MATRIX` | yes, in M (MA-ES has none: the CMA-ES update to first order) | yes, d/‖z‖² | yes | faster (about 23 %) |
| `CHOLESKY` | as rank-one downdates; one that fails is skipped | yes | yes | faster (about 27 %) |
| `CMSA` | switched off (no active variant; C became indefinite) | | | |
| `NATURAL_GRADIENT` | all λ, the utilities (`active` selects `UTILITY`) | no (unscaled, as xNES) | (trace-free) | xNES |
| `NONE` | switched off | | | |

The rescaling and the decay of `SEPARABLE`, `MATRIX` and `CHOLESKY` are those of `COVARIANCE` since
#73; "in tests" is the median number of evaluations to 1e-8 on the ellipsoid (d = 10, 10 runs),
with active against without (COVARIANCE: 27 %). Before, the three ignored the rescaling (and
`MATRIX` and `CHOLESKY` the negative weights in the decay), which was a few percent faster on this
ellipsoid but does not keep C positive definite.

With pairwise selection the losers of the mirrored pairs get no negative weights, as in pycma
(which never gives injected or mirrored points negative weights): a loser's step is the winner's
mirrored, so a negative update would cancel the winner's positive one on the same direction. With
the default weights every negative weight falls on a loser, whatever µ (#73), so the active update
then has no effect.

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

Mirrored sampling is unconditional: individuals 2i and 2i + 1 of each generation are z and −z
(Brockhoff et al. 2010), so a generation has ⌈λ/2⌉ independent samples, as the framework paper
describes. pycma's default is selective mirroring (only the worst points are mirrored, in the next
generation).

* **Pairs per generation** (#75). Every generation starts with a new pair. With an odd λ the last
  individual is unpaired, and with λ = 1 mirroring has no effect. Brockhoff et al. (2010, Sec. 2)
  and Wang et al. (2019, Alg. 1) instead carry the mirror of an odd generation's last sample over to
  the next generation, which makes their (1+1ₘ)-ES mirror every second generation; pycma mirrors
  within a generation.
* **Rejected samples.** A pair that bound correction (`RESAMPLE`) or repelling rejects is sampled
  again as a pair.
* **Sequential selection** evaluates the pair members one by one (Brockhoff et al. 2010 evaluate
  x + N, x − N, x + N′, …), but samples the partner of a batch's last individual with it, so that a
  rejection still replaces the whole pair. A stop after the first of a pair discards the partner.
* Before 2.0 the mirror state carried over between generations and through rejected samples, so the
  pairs fell out of step with `RESAMPLE`, repelling, an odd λ or sequential selection.

With `PAIRWISE` only the better point of each pair is ranked (Auger et al. 2011, Alg. 1). The
winners, the unpaired points and the parents of elitism are sorted by f; the losers follow, sorted
by f, with their real values (`Population.n_ranked`).

* The losers get no negative weights (above).
* `MSR` and `PSR` compare the ranked points only: Auger et al. adapt σ from the ranked offspring.
  pycma's `CMAAdaptSigmaMedianImprovement` reads all values, but pycma has no pairwise selection
  of its own (its selective mirrors are injected).
* The statistics (`stats.current_avg`) and the restart criteria (`Stagnation`) read every value.

Before 2.0 the losers got f = +inf: `MSR` counted against the winners only and σ shrank every
generation, and the mean value and `Stagnation`'s medians were infinite.

## Orthogonal sampling

The independent samples of a generation, λ (⌈λ/2⌉ with mirroring), are orthonormalized in blocks
that start with the generation: the first min(λ, d), then blocks of at most d. Each keeps the length
of the sample it came from (Wang, Emmerich and Bäck 2019, Sec. 3.2).

* The orthonormalization is a Householder QR decomposition with the signs of Q chosen so that
  diag(R) > 0: the result of the Gram–Schmidt process that Wang et al. use (the first sample keeps
  its direction).
* Wang et al. (Alg. 4) draw the samples beyond the first d at random; here they form further
  orthogonal blocks.
* Samples drawn to replace rejected ones (`RESAMPLE`, repelling) count towards the generation, so a
  generation with rejections is not completely orthogonal.

Before 2.0 a block held max(λ, d) samples (λ/2 with mirroring, minus 2 with TPA), so with λ < d a
block spanned several generations; and the signs were not corrected, so sample j of every block
was biased towards −eⱼ (d = 5, 5000 blocks: E[zⱼ[j]] = −0.80 … −0.70, with σ growing without
selection).

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

* `SOBOL`: a fixed set of N Sobol points (N = max(128, the largest power of 2 ≤ 2·10⁶/d)) drawn in
  a random order. The set, and the order, repeat after N samples: at d = 1000, N = 1024, after 42
  generations of the default λ = 24 (85 with mirroring). The order is a linear congruential
  generator with a random multiplier ≡ 5 (mod 8); before 2.0 the multiplier was 1 modulo N, so the
  order was a constant stride through the points. `HALTON`: a scrambled Halton sequence, which does
  not repeat.
* The transformations other than Gaussian have unit variance (Cauchy: unit scale) and are mapped
  from the uniform samples through their inverse CDF (de Nobel et al. 2025). Step-size rules that
  assume a finite variance fail with Cauchy samples ([compatibility](../guide/compatibility.md#combinations-to-avoid)).
* **Expected lengths.** CSA (with the h_σ test of the path p_c) and threshold convergence compare
  lengths with E‖z‖ of the transformation (`es.p.sampler.expected_length()`): for `GAUSSIAN` Hansen's
  approximation √d (1 − 1/(4d) + 1/(21d²)); for `SCALED_UNIFORM`, `LAPLACE` and `LOGISTIC` E‖z‖ of
  their distribution, computed numerically when the sampler is created; for `DOUBLE_WEIBULL`
  Γ(d + ½)/Γ(d) exactly; for `CAUCHY`, which has no E‖z‖, an approximation of the median length,
  d (1 + 0.18 tanh(d/4.07)). Before 2.0 the four unit-variance transformations used √d, which is
  (E‖z‖²)^½: too long by 23 % (`LAPLACE`), 17 % (`LOGISTIC`), 7 % (`SCALED_UNIFORM`) and 6 %
  (`DOUBLE_WEIBULL`) at d = 2, by 0.1–0.6 % at d = 100. de Nobel et al. (2025, Sec. 5) use √d for
  all but Cauchy.
* `fast_sampling` replaces mt19937 and the inverse normal CDF by xoshiro256++ and a ziggurat, only
  for `UNIFORM` + `GAUSSIAN`.

## IPOP and BIPOP

* **IPOP** (Auger & Hansen 2005): λ doubles at every restart, up to 2⁹ λ₀, as in the framework
  paper (`IPOP.max_lambda_factor`; before 2.0, (d λ₀)²). σ and the mean are
  reset. No budget per run: the restart criteria
  end a run. Every run ends with an evaluation of its center of mass, the last one too, as pycma's
  `eval_final_mean`; the budget always keeps that evaluation (pycma can exceed `maxfevals` by one).
  The first generation after a restart takes at most the rest of the budget, less the new run's
  center of mass (λ is capped, at least 2 points, and µ follows it); the hill-valley tests of
  repelling restarts stop when the budget runs short.
* **µ after a restart** (IPOP, BIPOP and the cap at the budget): the default of the new λ, ⌊λ/2⌋
  (⌊λ/4⌋ with `SA` and `CMSA`), as pycma's `fmin`, which creates every run with the default
  `CMA_mu` (`popsize // 2`: the positive weights of `RecombinationWeights(popsize)`), as Hansen
  (2009) (µ = ⌊λ/2⌋) and the framework paper (Algorithm 1 sets µ ← ⌊λ₀/2⌋ at every restart). With
  a given `mu0`, µ₀λ/λ₀ rounded half up (at least 1), the maintainer's choice (#150): pycma keeps a
  given `CMA_mu` as the same number in every run, which leaves µ = 3 of λ = 2⁹ λ₀. Before 2.0 µ
  kept the ratio µ₀/λ₀ in every case, rounded by IPOP (of the previous run's µ/λ) and truncated by
  BIPOP: with an odd λ₀ the default µ differed from pycma's (d = 3: λ = 14 had µ = 6, not 7), and
  the two strategies differed for the same λ (#150).
* **BIPOP** (Hansen 2009): two regimes, one of large populations (λ doubling, up to 2⁹ λ₀ as
  IPOP) and one of small ones, λ_s = ⌊λ₀ (λ_large/λ₀)^(U²)⌋ with σ = σ₀·10^(−2U′), where λ_large
  is the λ of the latest large run. Each regime counts the evaluations of its runs; a restart uses
  the regime that has used fewer, the large one on a tie, so the first restart is large. As in
  pycma's `fmin`, the first run counts as small (Hansen 2009 disregards it), and a small run takes
  about half the evaluations of the large regime so far: `restart.BIPOP` sets
  `max_iter` = ⌈½ (large evaluations)/λ_s⌉ generations, which `ExceededMaxIter` enforces. This is
  pycma's cap, `maxiter = min(maxiter0, 0.5 * sum(large_i) / popsize)`, which it does not round
  (`cma/evolution_strategy.py`:5372), with its stop `countiter >= maxiter` (4527–4529): a run of
  ⌈·⌉ generations, one more than Hansen (2009)'s "maximum number of function evaluations of half
  of the recent large budget" when the ratio is not an integer. modcma follows pycma, as for the
  populations (maintainer, #139); before 2.0 the cap was rounded down. U and U′ are drawn from
  the generator's raw output (`rng::uniform`), the same on every platform; they were drawn with
  `std::uniform_real_distribution`, whose algorithm depends on the standard library (#164).
  The small runs start with σ₀·10^(−2U′), and `TolX`, `MaxSigma`, `MaxDSigma` and `MinDSigma`
  measure σ against it (pycma's `es.sigma0` of each run; before 2.0 `MaxDSigma` and `MinDSigma`
  used `Settings.sigma0`, #139). λ_s is rounded down, as in the paper and pycma; up to even only
  with `PAIRWISE`.
  Up to the limit, the population sizes are pycma's (4.6.0, `fmin`,
  `cma/evolution_strategy.py`:5362–5381): λ_s lies between λ₀ and the latest large population,
  because pycma counts the small run (`runs_with_small += 1`) before it computes the multiplier,
  which is then the latest large run's. Hansen (2009) writes λ_ℓ/(2λ_def) in place of λ_ℓ/λ_def,
  with λ_ℓ "the latest population size from the first regime", and the range [λ_def, λ_ℓ/2], half
  of pycma's; modcma follows pycma (#140). The large population stops growing at
  `BIPOP.max_lambda_factor` λ₀ = 2⁹ λ₀, the limit of IPOP (`default_max_lambda_factor` in
  `restart_strategy.hpp`), and λ_s then stays within it (#140). pycma has no such limit, but its
  `fmin` stops after `restarts` large runs; with 9 (`restarts=True`), as Hansen (2009), its
  largest population is 2⁹ λ_def, the same. modcma keeps restarting at the limit until the budget
  is used. The limit acts only from the 10th large run on: in the golden runs and the BIPOP
  configurations of `tests/statistical.py` (budget 3000 d) λ_large reaches at most 8 λ₀. With its
  default popsize, pycma multiplies the unrounded 4 + 3 ln d (17, 35, 70, … at d = 5), modcma and
  the paper the integer λ₀. A C++ test (`tests/cpp/test_main.cpp`) compares λ, µ, σ and the cap
  of every run with a transcription of pycma's rule, for a fixed random stream, over fourteen
  large runs.
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
| `ExceededMaxIter` (100 + 50(d + 3)²/√λ generations) | `maxiter` 100 + 150(d + 3)²/√λ | a third of pycma's, as Hansen (2009); with the λ of the run, as Hansen (2009), where pycma's `fmin` keeps the value of the first run (it evaluates its options once). A run takes at most `max_iter` generations, as pycma (`countiter >= maxiter`); before 2.0, `max_iter` + 2 (#139) |
| `NoImprovement` (range of the generations' best over 10 + ⌈30d/λ⌉ generations ≤ 1e-20) | `tolfunhist` 1e-12 | tolerance; modcma has no `tolfun` (current and recent values, 1e-11) [WP17] |
| `MaxSigma` (σ max(D) grew by more than 1e4 in the run) | `tolfacupx` (σ√Cᵢᵢ > 1e3 σ₀) | Hansen (2016)'s TolXUp: the largest axis instead of each coordinate; before 2.0, σ > 1e4 |
| `MinSigma` (σ < 1e-20) | none | |
| `MaxDSigma` / `MinDSigma` (σ/σ₀ > 1e20 max(D), < 1e-20 min(D)) | `tolupsigma` 1e20 | `MinDSigma` is modcma's. σ₀ is the run's, as pycma's `es.sigma0` and Hansen (2009)'s TolUpSigma; before 2.0, `Settings.sigma0` in every run (#139) |
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

`ADAPTIVE` grows the radius of a taboo point whose basin a run found again by
exp(0.15 min(w, 3)), with w the evaluations of that run over their running average: the first
run's length, then an exponential average with α = 0.2, updated at every restart before the
hill-valley tests. Every run counts from the previous restart, the first from the start, also a
run that left no finite point (it updates no archive). For runs of equal length w = 1, a growth by
e^0.15. Before 2.0 (#121) the first run counted as one evaluation, the average started at 1, and a
run without a finite point was not counted (the next counted both): with runs of equal length
n ≫ 1, w was 5 (capped at 3) at the second restart (the first that can find a shared basin) and
2.8 at the third.
