# Step-size adaptation compared with the published rules

In modcma a point is x = m + σ·y with y = A z (or m + σᵢ·y with a sampled step size σᵢ per point;
the lower bound of integer variables raises single coordinates of the step size). The step-size
rule runs after the evolution paths and before the matrix update, for every matrix adaptation.
When the rule is not CSA (or λ = 1, which runs SR) and the matrix adaptation is `COVARIANCE`, the
Settings switch to `COVARIANCE_NO_EIGV` ([compatibility](../guide/compatibility.md)).

## CSA

**What modcma does.** σ ← σ·exp(clip((cs/damps)(‖p_σ‖/E‖z‖ − 1), −1, 1)), with
p_σ ← (1 − cs) p_σ + √(cs(2 − cs)µ_eff)·v, where v = C^-1/2 Δm for `COVARIANCE` and v = Σ wᵢzᵢ,
the weighted sum of the selected z, for every other matrix adaptation. Δm is the step of the mean
in units of the step size, (m' − m)/σ (Hansen 2016, Eq. (24)), with σ per coordinate the step size
recombined from the selected points, Σ wᵢsᵢ:

* σ when every selected point has the step size σ; then C^-1/2 Δm = B Σ wᵢzᵢ;
* σ raised by the lower bound of an integer variable in its coordinate: Δm = Σ wᵢyᵢ, the
  increment of the paths of Marty et al. (2024), Eqs. (5) and (7) (pycma divides by σ and
  `sigma_vec`);
* with sampled step sizes (`sample_sigma`, `SA`, `LPXNES`), the recombined step size, as CMSA
  recombines its ⟨σ⟩ (Spettel, Beyer & Hellwig 2018, Alg. 3, line 33).

It was the step of the mean divided by the mean of all step sizes of the population (#71). On a
mixed-integer ellipsoid (d = 10, 5 integer variables, 25 runs) the median evaluations went from
6351 to 2461; dividing by σ alone took 9881.

E‖z‖ is the expected length of the sampler: the χ_d approximation for Gaussian samples, E‖z‖ of
the distribution for the other unit-variance transformations, an empirical median for Cauchy.

**Compared with Hansen (2016) and pycma.** cs = (µ_eff + 2)/(d + µ_eff + 5) and
damps = 1 + 2 max(0, √((µ_eff − 1)/(d + 1)) − 1) + cs are the tutorial's (Table 1, Eq. (55); the
same in its 2023 revision, arXiv:1604.00772v2) and those of `cmaes`. With very large populations cs
tends to 1. pycma 4.6 has d + µ_eff + 3 in the denominator of cs and caps it at 0.5 (modcma 1.1 and
1.2 had + 3 without the cap), and adds terms to damps for mirrored sampling and large µ_eff. The
clip at ±1 is pycma's. E‖z‖ is that of the sample transformation: for the uniform, Laplace,
logistic and double-Weibull transformations their own, not √d as before 2.0
([samplers](selection-sampling-restarts.md#samplers-and-transformations)).

## TPA: two-point adaptation

**What modcma does.** Two extra points per generation, x± = m ± σ·Δm (Δm the last mean shift in
units of the step size, as for [CSA](#csa)), evaluated with the population (in its first batch:
they do not depend on its values);
s ← (1 − c)s + c·r with r = −0.5 when f(x−) < f(x+) and 0.5 otherwise, c = 0.3, then σ ← σ·eˢ.
With ask–tell, `ask()` returns λ + 2 points.

**Compared with Hansen (2008) and pycma.** pycma takes the two points from the λ offspring (no extra
evaluations), rescales them to a random length, uses the rank difference among all λ, compresses it
and damps the update by 0.7 + 2 ln d + …; modcma follows the Python implementation of the framework
(and ModEA). A tie (f(x−) = f(x+), as after a restart, when Δm = 0) gives r = 0, as the sign of
the rank difference in pycma. Before 2.0 a tie counted as a success: σ grew by 16 % after every
restart, and diverged with elitism.

**Bounds.** With a bound correction other than `NONE`, both points move toward m by the same
factor until both are in the box, then their integer coordinates are rounded. The test steps stay
symmetric about m, as Hansen (2008, Sec. 2.2) chooses them; Hansen (2008) has no bounds. pycma
passes them through its boundary handler like every other solution, which moves each point on
its own. Corrected one by one with the method of the samples, TPA compared two points at
different distances from m and lost runs on BBOB (box ±4): `TOROIDAL` and `UNIFORM_RESAMPLE`
reached 1e-8 in 16 and 13 fewer of 350 runs than with the points left outside the box, and
clipped points (for `RESAMPLE`) solved f2 (d = 20) in 10 to 15 of 25 runs instead of 25. The
method moves the points only when m itself is outside the box. The points were only rounded
before, so TPA evaluated points outside the box, and one could be reported as the best (#98).

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

With pairwise selection K counts the ranked points of the generation (the winners of the mirrored
pairs), against the j-th best ranked point of the previous one, and is centered at half their
number: Auger et al. (2011, Alg. 1) adapt σ from the ranked offspring. Before 2.0 the losers had
f = +inf, so K counted the winners against λ/2 and σ shrank every generation.

## PSR: population success rule

**What modcma does.** Ranks the current and the previous generation together (the best n of each,
n the number of finite values; with pairwise selection of the ranked points, as `MSR`) and
compares their ranks:

* z = Σ(r_old − r_new)/n² − z*, with the target z* = 0.25;
* s ← (1 − c)s + c·z, with c = 0.3;
* σ ← σ·exp(s/d_σ), with d_σ = 1.

These constants are those of Loshchilov (2014), Alg. 6. Before 2.0, modcma used c = 0.9 and
d_σ = 2 − 2/d; the Python implementation used c = 0.4.

The paper does not say how equal values are ranked. modcma gives them the mean of their ranks
(mid-ranks, as in rank tests), so a tie counts for neither generation. With elitism the parents
are in both generations: about 3 of the 10 values tie in every generation (d = 10). The order of
the previous generation's ranking decided before (#120). Index order, the current generation
first, would count each tied parent as a success of 1/n² in z; on BBOB with elitism it took 7 to
24 % more evaluations than mid-ranks on f1, f2 and f10.

## XNES, MXNES, LPXNES

The learning rates of the three are those of Krause, Glasmachers & Igel (2017), Table 3, except
XNES with `NATURAL_GRADIENT`, which has xNES's. Their cs is a learning rate of log σ, and also the
rate of the path p_σ, which h_σ and MATRIX's rank-one update read, at most 1
(`Weights.cs_path`): LPXNES's cs exceeds 2 for µ_eff > 2.22√d, and p_σ became NaN (#72).

* **XNES with `NATURAL_GRADIENT`** is xNES's rule: σ ← σ·exp((η_σ/2)·tr(G)/d), from the natural
  gradient G of the matrix update ([matrix adaptations](matrix-adaptations.md#natural_gradient-xnes)),
  with η_σ = (9 + 3 ln d)/(5 d √d) (Glasmachers et al. 2010).
* **XNES with another matrix adaptation** is the xNES SA of Krause et al. (2017), a length rule:
  σ ← σ·exp((cs/√d)·Σ wᵢ(‖zᵢ‖² − d)) over the µ best, with their convex weights (summing to 1, also
  for `UTILITY`) and cs = µ_eff/(2 ln d √d) (ln(max(2, d)) at d = 1, as 1.x). It was 0.01 until
  #72: a median of 3650–9950 evaluations on the sphere at d = 10 instead of 750–1070 (5 runs). The
  framework paper's description of the xNES rule (the length of the weighted mean of z) is MXNES.
* **MXNES** (m-xNES, Krause, Glasmachers & Igel 2017): σ ← σ·exp((cs/d)(µ_eff‖Σwᵢzᵢ‖² − d)), cs = 1
  (also with `NATURAL_GRADIENT` since #72): the length of the mean step, unbiased under random
  selection.
* **LPXNES** (lp-xNES, same paper, Sec. 4.4): each point samples its own step size σᵢ (log-normal
  around σ, β = ln 2/(√d ln(max(d, 2))), finite at d = 1 since #72), and σ ← σ·exp(cs·Σ wᵢ
  log(σᵢ/σ)) over the µ best, with the convex weights and cs = min(1, 9µ_eff/(10√d)): Krause et
  al.'s rate (their Table 3), capped at 1 by modcma, since above 1 the update overshoots the
  weighted geometric mean of the step sizes (1.29 at d = 2 and 1.05 at d = 5 with the default λ).
  One step size per point, as in the paper; it was drawn per coordinate (#71).

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
evaluated outside the generation (TPA's two points, the centers of mass that end the runs) do not
count; before 2.0 they did.

**Compared with the framework paper.** The paper's success is "better than the best of the
previous generation". With elitism this is the same as the best of the run.

Without elitism it is not: the previous generation's best is beaten in about every other
generation, even at a good σ, far above the target 2/11. On BBOB (f1, f2, f8, f10, f12; d = 5, 20)
σ then diverged and no run reached 1e-8; with the best of the run, 250 of 350 did. The paper's text
is to be corrected: "the best so far", or SR described for elitist selection (#83).

## SA: self-adaptation

**What modcma does.** Each point samples its step size σᵢ = σ·exp(β nᵢ), one per point, with
β = 1/√(2d), CMSA's τ (Beyer & Sendhoff 2008, as restated by Spettel, Beyer & Hellwig 2018,
Alg. 3 and Table I); σ moves to the weighted geometric mean of the selected points' step sizes
divided by the geometric mean over the population: the weighted CMSA operator with the correction
of Ahrari et al. (2017), as the framework paper describes. µ = λ/4 by default.

The step sizes were drawn per coordinate, with β = 1/√(2µ_eff), until #71: the points were not
along y (no rotation invariance), and the signal of SA was averaged over d draws. CMSA with SA
solved 0 of 5 rotated ellipsoids (d = 10); with one step size per point it solves 10 of 10. The
paper and the Python implementation draw one per point.

## Which rule with which matrix adaptation

| step-size rule | reads | works with | notes |
|---|---|---|---|
| `CSA` (default) | the path p_σ | every matrix adaptation | `COVARIANCE` uses C^-1/2 Δm, the others the weighted z |
| `TPA` | two extra evaluations | every | |
| `MSR`, `PSR` | the f-values of two generations | every | with pairwise selection the ranked points only |
| `XNES` | the natural gradient, or the lengths of z | every (with `NATURAL_GRADIENT` xNES's rule; with `weights=UTILITY` too, all of xNES) | not with `elitist` or `MATRIX` (σ collapses in some runs) |
| `MXNES` | the length of the mean step | every | sensitive to threshold convergence |
| `LPXNES`, `SA` | sampled step sizes | every; `SA` is CMSA's | |
| `SR` | successes against the best of the run | λ = 1 (automatic) | one success per generation for λ > 1 |
