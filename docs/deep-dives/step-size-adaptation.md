# Step-size adaptation compared with the published rules

In modcma a point is x = m + σ·y with y = A z (or m + S∘y with sampled step sizes S). The step-size
rule runs after the evolution paths and before the matrix update, for every matrix adaptation.
When the rule is not CSA and the matrix adaptation is `COVARIANCE`, the Settings switch to
`COVARIANCE_NO_EIGV` ([compatibility](../guide/compatibility.md)).

## CSA

**What modcma does.** σ ← σ·exp(clip((cs/damps)(‖p_σ‖/E‖z‖ − 1), −1, 1)), with
p_σ ← (1 − cs) p_σ + √(cs(2 − cs)µ_eff)·v, where v = C^-1/2 Δm/σ for `COVARIANCE` and the weighted
sum of the selected z for every other matrix adaptation (the same vector when the points are m + σAz).
E‖z‖ is the expected length of the sampler: the χ_d approximation for Gaussian samples, an empirical
median for Cauchy.

**Compared with Hansen (2016) and pycma.** cs = (µ_eff + 2)/(d + µ_eff + 3) is pycma's (the
tutorial: + 5); pycma also caps cs at 0.5, modcma does not (with very large populations cs tends
to 1). damps is the tutorial's; pycma 4.6 adds terms for mirrored sampling and large µ_eff. The clip
at ±1 is pycma's. The expected length of the uniform, Laplace, logistic and double-Weibull
transformations is taken as √d instead of their own (*known issue* [F29]).

## TPA: two-point adaptation

**What modcma does.** Two extra points per generation, x± = m ± σ·Δm (Δm the last mean shift in
units of σ), evaluated after the population; s ← (1 − c)s + c·r with r = −0.5 when f(x−) < f(x+)
and 0.5 otherwise, c = 0.3, then σ ← σ·eˢ. With ask–tell, `ask()` returns λ + 2 points.

**Compared with Hansen (2008) and pycma.** pycma takes the two points from the λ offspring (no extra
evaluations), rescales them to a random length, uses the rank difference among all λ, compresses it
and damps the update by 0.7 + 2 ln d + …; modcma follows the Python implementation of the framework
(and ModEA). A tie (f(x−) = f(x+), as after a restart, when Δm = 0) gives r = 0, as the sign of
the rank difference in pycma. Before 2.0 a tie counted as a success: σ grew by 16 % after every
restart, and diverged with elitism.

## MSR: median success rule

**What modcma does.** K is the number of points better than the j-th best of the previous
generation, j = 0.3λ, as in Ait ElHara, Auger & Hansen (2013). As pycma's
`CMAAdaptSigmaMedianImprovement`, j is a fractional 0-based index into the sorted values, and K is
interpolated between its two neighbours. Then:

* z = (2/λ)(K − λ/2), so z ≥ 0 when the median point beats the j-th best;
* s ← (1 − c)s + c·z, with c = 0.3;
* σ ← σ·exp(s/d_σ), with d_σ = 2 − 2/d (1 at d = 1, where it would be 0).

`MSR.quantile` sets 0.3. K is centered at λ/2, as in pycma. Loshchilov (2014), restating the paper,
centers it at (λ + 1)/2. On BBOB at d = 20 that version solved Rosenbrock (f8) and the bent cigar
(f12) in 15 of 25 runs each. With pycma's centering, modcma solves them in 22 and 20 of 25 (pycma
itself: 22 and 25), and its median evaluations on the sphere equal pycma's (3120).

*Known issue* [F9]: with pairwise selection the losers' values are infinite, so K counts against
the winners only.

## PSR: population success rule

**What modcma does.** Ranks the current and the previous generation together (the best n of each,
n the number of finite values) and compares their ranks:

* z = Σ(r_old − r_new)/n² − z*, with the target z* = 0.25;
* s ← (1 − c)s + c·z, with c = 0.3;
* σ ← σ·exp(s/d_σ), with d_σ = 1.

These constants are those of Loshchilov (2014), Alg. 6. Before 2.0, modcma used c = 0.9 and
d_σ = 2 − 2/d; the Python implementation used c = 0.4.

## XNES, MXNES, LPXNES

* **XNES with `NATURAL_GRADIENT`** is xNES's rule: σ ← σ·exp((η_σ/2)·tr(G)/d), from the natural
  gradient G of the matrix update ([matrix adaptations](matrix-adaptations.md#natural_gradient-xnes)).
* **XNES with another matrix adaptation** is a length rule:
  σ ← σ·exp((cs/√d)·Σ wᵢ(‖zᵢ‖² − d)) with cs = 0.01, which is slow (*known issue* [F27]). The
  framework paper's description of the xNES rule (the length of the weighted mean of z) is MXNES.
* **MXNES** (m-xNES, Krause, Glasmachers & Igel 2017): σ ← σ·exp((cs/d)(µ_eff‖Σwᵢzᵢ‖² − d)), cs = 1:
  the length of the mean step, unbiased under random selection.
* **LPXNES** (lp-xNES, same paper): each point samples its own step sizes (log-normal around σ,
  β = ln 2/(√d ln d)), and σ ← σ·exp(cs·Σ wᵢ·mean log(Sᵢ/σ)) with cs = 9µ_eff/(10√d). *Known
  issues*: the step sizes are per coordinate [F1]; β is infinite at d = 1 [F16].

## SR: success rule

**What modcma does.**

* p ← (1 − 1/12)p + (1/12)·[the best of the generation beats the best of the run so far];
* σ ← σ·exp((p − 2/11)/((1 − 2/11)·damps)), damps = 1 + d/2.

For λ = 1 the best of the run is the parent, so this is the success rule of the (1+1)-CMA-ES
(Igel et al. 2006), which the Settings select automatically.

**SR is designed for elitist selection.** In the (1+1)-ES a success means beating the parent, and
the parent is always the best point so far. modcma also allows SR without elitism, and then
measures success against the best of the run, which takes the parent's role. It is a 1/5th-type
rule under comma selection; the median success rule (MSR, Ait ElHara et al. 2013) was proposed
for exactly that setting.

The success ratio is counted once per generation and is not reset at a restart. The points
evaluated outside the generation (TPA's two points, the restart centers) do not count; before 2.0
they did.

**Compared with the framework paper.** The paper's success is "better than the best of the
previous generation". With elitism this is the same as the best of the run.

Without elitism it is not: the previous generation's best is beaten in about every other
generation, even at a good σ, far above the target 2/11. On BBOB (f1, f2, f8, f10, f12; d = 5, 20)
σ then diverged and no run reached 1e-8; with the best of the run, 250 of 350 did. The paper's text
is to be corrected: "the best so far", or SR described for elitist selection (#83).

## SA: self-adaptation

**What modcma does.** Each point samples its step size (log-normal around σ, β = 1/√(2µ_eff)); σ
moves to the weighted geometric mean of the selected points' step sizes divided by the geometric
mean over the population: the weighted CMSA operator with the correction of Ahrari et al. (2017),
as the framework paper describes. µ = λ/4 by default. *Known issue* [F1]: the step sizes are drawn
per coordinate, which breaks rotation invariance; the paper and the Python implementation draw one
per point.

## Which rule with which matrix adaptation

| step-size rule | reads | works with | notes |
|---|---|---|---|
| `CSA` (default) | the path p_σ | every matrix adaptation | `COVARIANCE` uses C^-1/2 Δm, the others the weighted z |
| `TPA` | two extra evaluations | every | |
| `MSR`, `PSR` | the f-values of two generations | every | not `MSR` with pairwise selection [F9] |
| `XNES` | the natural gradient, or the lengths of z | `NATURAL_GRADIENT` (xNES's rule; with `weights=UTILITY` all of xNES) | slow elsewhere [F27] |
| `MXNES` | the length of the mean step | every | sensitive to threshold convergence |
| `LPXNES`, `SA` | sampled step sizes | every; `SA` is CMSA's | per coordinate [F1] |
| `SR` | successes against the best of the run | λ = 1 (automatic) | one success per generation for λ > 1 |
