# Matrix adaptations compared with the published algorithms

All learning rates below are modcma's defaults (`es.p.weights`): c1 = 2/((d+1.3)² + µ_eff),
cµ = min(1 − c1, 2(µ_eff − 2 + 1/µ_eff)/((d+2)² + µ_eff)), cc = (4 + µ_eff/d)/(d + 4 + 2µ_eff/d),
cs = (µ_eff + 2)/(d + µ_eff + 3) (pycma's value; the tutorial has + 5 in the denominator), and
damps = 1 + 2 max(0, √((µ_eff − 1)/(d + 1)) − 1) + cs.

## COVARIANCE: CMA-ES

**What modcma does.** Samples y = B D z from the eigendecomposition C = B D² Bᵀ; the step-size path
uses C^-1/2 Δm; the path p_c is stalled by h_σ; the update is

C ← (1 + c1 δ(h_σ) − c1 − cµ Σwᵢ) C + c1 p_c p_cᵀ + cµ Σ wᵢ° yᵢ yᵢᵀ,

over the µ best points, or over all λ with the active update (the negative weights rescaled by
d/‖C^-1/2 y‖², as in Hansen 2016). C is updated every generation; the eigendecomposition only every
1/(10 d (c1 + cµ)) generations.

**Compared with Hansen (2016) and pycma.**

* The active update is off by default (`active=False`); the tutorial's default weights and pycma
  (`CMA_active=True`) include it.
* cs uses + 3 (pycma) instead of the tutorial's + 5.
* pycma adds an offset to cµ (0.0236 against modcma's 0.0202 at d = 10), scales c1 for small λ, and
  tests h_σ with the squared path length.
* h_σ's bias correction, 1 − (1 − cs)^(2(g+1)), counts the generations g of the current run, as
  Hansen (2016) and pycma. Before 2.0 it counted evaluations/λ over all runs, so after a restart
  the correction was already about 1.

**Closest configuration:** `active=True` with the defaults.

## COVARIANCE_NO_EIGV

The update of `COVARIANCE`, with C factorized by a Cholesky decomposition C = L Lᵀ (at the same lazy
interval) instead of an eigendecomposition: cheaper. The step-size path uses L⁻¹Δm, which equals
C^-1/2 Δm up to a rotation that changes with every decomposition, so CSA's path is not exactly
CMA-ES's (the same approximation Krause et al. 2016 argue for). There is no published counterpart.
It is chosen automatically when the step-size rule is not CSA; with CSA it gave the same results
as `COVARIANCE` on the sphere and a rotated ellipsoid (d = 5 to 20), despite the docstring's
caution [F28].

## SEPARABLE: sep-CMA-ES

**What modcma does.** Adapts only the diagonal of C (O(dλ) per generation) with the same update
restricted to the diagonal; the step-size path uses D⁻¹Δm, exact here.

**Compared with Ros & Hansen (2008) and pycma.** Only cµ is scaled by (d + 2)/3; c1 and cc keep
the values of the full update. pycma's separable mode uses its own, larger rates (at d = 100:
c1 = 8.3·10⁻³ against modcma's 1.95·10⁻⁴). `cmaes`' `SepCMA` scales the same way as modcma.
With the active update the negative weights are not rescaled. *Known issue* [F4]: the cap
cµ ≤ 1 − c1 is applied before the scaling, so with large populations (IPOP) the decay of C turns
negative.

**Closest configuration:** `matrix_adaptation=SEPARABLE`, `active=False`.

## MATRIX: MA-ES

**What modcma does.** Adapts M with C = M Mᵀ, without p_c and without a decomposition:

M ← M [I + (c1/2)(s sᵀ − I) + (cµ/2)(Σ wᵢ zᵢ zᵢᵀ − I)],

computed in O(d²λ), with s the step-size path from the weighted z, updated before M as in Beyer &
Sendhoff (2017).

**Differences.** The learning rates are CMA-ES's defaults. MA-ES has no active variant: with the
active update the weights no longer sum to 1, which the update assumes, and the negative weights are
not rescaled [F7]; in tests MATRIX with the active update still converged, faster than without.

**Closest configuration:** `matrix_adaptation=MATRIX`, `active=False`.

## CHOLESKY: Cholesky-CMA-ES

**What modcma does.** Updates the triangular Cholesky factor A directly (Krause, Arbonès & Igel
2016, the variant the framework paper implements): A ← √(1 − c1 − cµ) A, then a rank-one update
with p_c and µ rank-one updates with the selected y; the step-size path uses A⁻¹Δm. Exact: A Aᵀ
equals the covariance update to rounding. p_c is not stalled by h_σ.

**Differences.** With the active update the negative weights are applied as rank-one downdates
without rescaling, and a downdate that fails is not detected (*known issue* [F5]).

**Closest configuration:** `matrix_adaptation=CHOLESKY`, `active=False`.

## CMSA: CMSA-ES

**What modcma does.** C ← (1 − 1/τ_c) C + (1/τ_c) Σ wᵢ yᵢ yᵢᵀ with τ_c = 1 + d(d+1)/(2µ_eff),
factorized by a Cholesky decomposition every generation; µ = λ/4 by default.

**Differences from Beyer & Sendhoff (2008).**

* The step size is not self-adapted unless `ssa=SA` is set: the default is CSA.
* With `ssa=SA`, each point gets its own step size, but *known issue* [F1]: one per coordinate
  instead of one per point, which breaks rotation invariance (CMSA with SA fails on a rotated
  ellipsoid, CMSA with CSA solves it).
* σ is recombined as a weighted geometric mean normalized by the population's (the weighted SA
  operator with Ahrari et al.'s correction, as the framework paper describes).
* The weights are the CMA-ES log weights with µ_eff; `weights=EQUAL` gives the original
  intermediate recombination.
* *Known issue* [F6]: the active update breaks it.

**Closest configuration:** `matrix_adaptation=CMSA`, `ssa=SA`, `weights=EQUAL` (until F1 is fixed,
`ssa=CSA` works better).

## NATURAL_GRADIENT: xNES

**Canonical xNES** (Glasmachers et al. 2010; as in `cmaes`' `XNES`) uses rank-based utilities over
all λ points that sum to zero, uᵢ = max(0, ln(λ/2 + 1) − ln i)/Σ − 1/λ, for the mean, for σ and for
the shape B, with the learning rates η_σ = η_B = (9 + 3 ln d)/(5 d √d).

**What modcma does.** It depends on the weights:

* **With `weights=UTILITY`** (also chosen when `active=True` with the default weights) it uses xNES's
  utilities: the mean moves by Σ uᵢ(xᵢ − m) over all λ points, the natural gradient is
  G = Σ uᵢ zᵢ zᵢᵀ, its trace sets σ (with `ssa=XNES`: σ ← σ·exp((η_σ/2)·tr G/d)) and its trace-free
  part updates A ← A·exp((η_B/2)·G) (exactly, by a low-rank formula when λ < d). This is xNES.
* **With the default weights** G = Σ wᵢ(zᵢzᵢᵀ − I) uses the µ positive CMA-ES weights and the mean
  is the usual recombination of the µ best: no −1/λ baseline. It needs about as many evaluations as
  xNES (2–6 % fewer at d = 5, 7–8 % more at d = 20; 10 runs, not tested for significance).
* With `ssa` other than `XNES` (CSA by default) the step size comes from that rule instead of the
  trace of G.
* *Known issue* [F3]: `NATURAL_GRADIENT` sets the learning rate of every step-size rule to η_σ.

Before 2.0, `active=True` used CMA-ES's negative weights, which do not sum to zero and are scaled
for CMA-ES: σ shrank even on a linear function and runs converged prematurely (F2, fixed by using
the utilities).

**Closest configuration:** `matrix_adaptation=NATURAL_GRADIENT`, `ssa=XNES`, `weights=UTILITY` (or
`active=True`). Median evaluations to 1e-8 over 10 runs, against `cmaes`' `XNES`:

| | modcma, UTILITY | `cmaes` XNES | modcma, DEFAULT weights |
|---|---|---|---|
| sphere, d = 5 | 1528 | 1532 | 1492 |
| sphere, d = 10 | 6180 | 6084 | 6220 |
| sphere, d = 20 | 26094 | 26114 | 28182 |
| ellipsoid, d = 5 | 2276 | 2304 | 2160 |
| ellipsoid, d = 10 | 8935 | 8904 | 8965 |
| ellipsoid, d = 20 | 37122 | 36824 | 39618 |

The step-size path of CSA is still computed with `ssa=XNES` (unused).

## NONE: CSA-ES

No matrix adaptation (C = I): only the step size is adapted, with the CMA-ES defaults of cs and
damps; the active update is off. The same as pycma with `CMA_on=False`.

## Summary

| matrix adaptation | named after | closest configuration | main remaining differences |
|---|---|---|---|
| `COVARIANCE` | CMA-ES (Hansen 2016, pycma) | `active=True` | cs + 3; pycma's cµ offset and h_σ form |
| `COVARIANCE_NO_EIGV` | (none) | with a step-size rule other than CSA | rotated step-size path |
| `SEPARABLE` | sep-CMA-ES (Ros & Hansen 2008) | defaults | c1, cc not scaled |
| `MATRIX` | MA-ES (Beyer & Sendhoff 2017) | defaults | CMA-ES learning rates |
| `CHOLESKY` | Cholesky-CMA-ES (Krause et al. 2016) | defaults | no h_σ |
| `CMSA` | CMSA-ES (Beyer & Sendhoff 2008) | `ssa=SA`, `weights=EQUAL` | sampled step sizes per coordinate [F1] |
| `NATURAL_GRADIENT` | xNES (Glasmachers et al. 2010) | `ssa=XNES`, `weights=UTILITY` | none measurable |
| `NONE` | CSA-ES | defaults | |

## The initial matrix: `C0`

Every adaptation starts from C = I, or from `Settings.C0` (warm starting, see
[settings](../guide/settings.md#the-initial-covariance-matrix-and-warm-starting)). It then starts
its sampling factor F from a factor of `C0`, F Fᵀ = `C0`, the one its own update keeps:

| matrix adaptation | starts from |
|---|---|
| `COVARIANCE` | C = `C0`, with its eigendecomposition: F = B D |
| `COVARIANCE_NO_EIGV`, `CMSA` | C = `C0`, with its lower Cholesky factor |
| `CHOLESKY` | the lower Cholesky factor of `C0` (it adapts a triangular factor) |
| `MATRIX`, `NATURAL_GRADIENT` | the symmetric square root `C0`^1/2 = B D Bᵀ (and its inverse) |
| `SEPARABLE` | the diagonal of `C0` |
| `NONE` | rejects `C0` |

Two factors of `C0` differ by a rotation Q of z (F' = F Q), so the first generation is
N(m, σ² `C0`) with either. The updates of MA-ES and xNES act on z from the right (M ← M (I + …),
A ← A expm(η G)), so with F' = F Q every later factor is the one of F times Q: with Gaussian z
(the default sampler; not Sobol or Halton points) the runs are the same in distribution, and only
which z gives which point differs. The symmetric root is the one
factor that does not depend on the signs and the order of the eigenvectors that the eigensolver
returns. For `metric()` at generation 0 every adaptation reports `C0` (`SEPARABLE` its diagonal),
and `coordinate_wise_variances` is the diagonal of `C0`.

## The metric of each adaptation

`es.p.adaptation.metric()` gives the shape of the distribution that is sampled, without σ: the
axis lengths d (the square roots of the eigenvalues of C) and the eigenvectors B. Where each
adaptation takes it from:

| matrix adaptation | metric | cost |
|---|---|---|
| `COVARIANCE` | its own d and B, which the eigendecomposition updates | none |
| `SEPARABLE` | its own d (B = I) | none |
| `NONE` | ones (C = I) | none |
| `COVARIANCE_NO_EIGV` | an eigendecomposition of L Lᵀ, from the last Cholesky factor L | O(d³), when asked |
| `CMSA` | an eigendecomposition of C | O(d³), when asked |
| `MATRIX`, `CHOLESKY`, `NATURAL_GRADIENT` | an eigendecomposition of F Fᵀ, F = M or A | O(d³), when asked |

A computed metric is reused until the factor changes (`factor_version`, increased by an update of
the matrix, a decomposition or a restart), so the requests of one state share one decomposition.

**The sampled distribution, not the latest C.** `COVARIANCE` and `COVARIANCE_NO_EIGV` update C every
generation but decompose it only every 1/(10 d (c1 + cµ)) generations, and sample with the last
decomposition. With the default settings they decompose every generation up to d = 82; at d = 100
the decomposition can be one generation behind C, at d = 1000 eight. The metric follows the
decomposition: it describes the points that were sampled. `coordinate_wise_variances`, the
diagonal of the latest C, can be ahead of it by as much.

**Cholesky-CMA-ES.** The diagonal of the Cholesky factor A gives det C = ∏ Aᵢᵢ², not the eigenvalues
of C, which are the squared singular values of A (for A = [[1, 0], [1, 1]]: 2.618 and 0.382, while
Aᵢᵢ = 1). So `CHOLESKY`'s metric is computed as well.

**The Hessian.** On a convex-quadratic function with Hessian H, C approaches H⁻¹ up to a constant
factor: Hansen (2016), Section 0.4, "we assume that the optimal covariance matrix equals to the
inverse Hessian matrix, up to a constant factor", with "a rigorous proof is missing". pycma uses
the relation (`inverse_hessian_scalar_correction` fits the factor). So `metric().inv_C()` estimates
the shape of the Hessian, not its scale, for the adaptations that learn a full matrix; `SEPARABLE`
can only match a diagonal Hessian.
