# Matrix adaptation

The mutations of CMA-ES are drawn from N(0, σ² C): the covariance matrix C learns the shape of the
landscape (its scaling and correlations), which is what makes CMA-ES invariant to rotations and
efficient on ill-conditioned problems. `matrix_adaptation` chooses how C, or a factor A of it
(C = A Aᵀ), is learned.

## Which one to choose

| option | learns | cost per generation | time at d = 1000 | choose it for |
|---|---|---|---|---|
| `COVARIANCE` | full C | O(d²λ) + an O(d³) eigendecomposition every few generations | 6.0 ms | the default: the most robust choice up to a few hundred dimensions |
| `MATRIX` (MA-ES) | full M, C = M Mᵀ | O(d²λ) | 0.95 ms | high dimension, when the full covariance matters but CMA-ES is too slow |
| `NATURAL_GRADIENT` (xNES) | full A | O(d²λ) (exact low-rank update) | 0.87 ms | as MA-ES; xNES itself with `ssa=XNES` and `weights=UTILITY` |
| `CHOLESKY` | Cholesky factor | O(d²λ) | 2.2 ms | as MA-ES |
| `COVARIANCE_NO_EIGV` | full C, Cholesky factor | O(d²λ) + O(d³) every few generations | 1.1 ms | step-size adaptations other than CSA (chosen automatically) |
| `CMSA` | full C, self-adapted | O(d²λ) + an O(d³) Cholesky factorization every generation | 2.0 ms | with `ssa=SA` (CMSA-ES) |
| `SEPARABLE` | the diagonal of C | O(dλ) | 0.29 ms | very high dimension, (nearly) separable problems |
| `NONE` | nothing (C = I) | O(dλ) | 0.28 ms | isotropic problems, or as a baseline |

The times are the cost of the algorithm per generation on the Apple M5 laptop of the
[benchmarks](../performance.md#modularcmaes-matrix-adaptations). The cheaper adaptations are
different algorithms, not faster versions of the default: they can need more evaluations, or fail,
on problems where the full covariance update works. When evaluations are expensive, the cost of
the algorithm hardly matters and `COVARIANCE` is the safe choice.

```python
import numpy as np
import modcma

def ellipsoid(X):
    return (X * X) @ (10.0 ** (6 * np.arange(X.shape[1]) / (X.shape[1] - 1)))

for option in ["COVARIANCE", "MATRIX", "SEPARABLE", "NONE"]:
    es = modcma.ModularCMAES(
        modcma.settings_from_dict(
            20, matrix_adaptation=option, x0=np.full(20, 3.0), budget=40_000, target=1e-8, seed=1
        )
    )
    es.run(ellipsoid, vectorized=True)
    print(f"{option:12s} f = {es.p.stats.global_best.y:9.2e}  evaluations = {es.p.stats.evaluations}")
```

On the separable ellipsoid `SEPARABLE` needs the fewest evaluations; `NONE` cannot learn the
scaling and runs out of budget. After a rotation of the problem, `SEPARABLE` would fail as well.

## The state

The adaptation is `es.p.adaptation`: the mean `m`, the evolution paths `pc` and `ps`, and the
matrix (`C` and its eigendecomposition `B`, `d` for `COVARIANCE`; `M` for `MATRIX`; `A` for
`CHOLESKY` and `NATURAL_GRADIENT`). See the [reference](../reference/matrix_adaptation.md). The
matrix starts as the identity, or as `Settings.C0`
([warm starting](settings.md#the-initial-covariance-matrix-and-warm-starting)).

### The shape of the distribution: `metric()`

Every adaptation gives the shape of its search distribution in one form, without σ:
`es.p.adaptation.metric()`. A sample is x = m + σ B (d ∘ z), so the one-sigma ellipsoid around m
has the semi-axes σ dᵢ along the columns of B.

* `d`: the axis lengths, the square roots of the eigenvalues of C;
* `B`: the eigenvectors, or `None` for `SEPARABLE` and `NONE` (C is diagonal);
* `eigenvalues()` (d²), `inv_C()` (C⁻¹) and `inv_root_C()` (C^-1/2).

`COVARIANCE`, `SEPARABLE` and `NONE` keep d (and B), so the metric costs nothing. The other
adaptations compute it when asked, an O(d³) eigendecomposition, and reuse it until their matrix
changes. It describes the distribution the points were sampled from: `COVARIANCE` decomposes C only
every few generations in high dimension, and the metric follows the decomposition, not the latest C.

```python
import numpy as np
import modcma

rotation, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(5, 5)))
weights = 10.0 ** np.arange(5)  # the Hessian has condition 1e4

def rotated_ellipsoid(x):
    return float(np.sum(weights * (rotation @ x) ** 2))

es = modcma.ModularCMAES(modcma.settings_from_dict(5, target=1e-6, seed=1))
es.run(rotated_ellipsoid)
metric = es.p.adaptation.metric()
print(f"condition of C: {metric.eigenvalues().max() / metric.eigenvalues().min():.3g}")
print(f"longest semi-axis: {es.p.mutation.sigma * metric.d.max():.2e}")
```

On a convex-quadratic function C approaches the inverse Hessian up to a constant factor (Hansen
2016, Section 0.4, an assumption with strong empirical support): here the condition of C ends near
the Hessian's 10⁴. So `inv_C()` estimates the shape of the Hessian, not its scale, for the
adaptations that learn a full matrix.

## The options

<!-- options: MatrixAdaptationType -->
**`modcma.options.MatrixAdaptationType`**: The adaptation of the shape of the search distribution (`Modules.matrix_adaptation`; default COVARIANCE). The mutations are y = A z, with z from the sampler and A a factor of the covariance matrix, C = A A^T. The options differ in how A is adapted and in their cost per generation (docs/performance.md, "ModularCMAES matrix adaptations"). C starts as the identity, or as `Settings.C0` (warm starting).

| value | |
|---|---|
| `COVARIANCE` | The covariance matrix adaptation of the CMA-ES (Hansen & Ostermeier 2001), with rank-one and rank-mu update, the default. C is updated every generation and eigendecomposed when 1 / (10 d (c1 + cmu)) generations have passed since the last decomposition (every generation in low dimension; O(d^3), with LAPACK from `constants.lapack_min_dim` on). Keeps its metric (`Adaptation.metric()`), so the restart criteria ConditionC and NoEffectAxis are used, as with SEPARABLE and NONE. |
| `NONE` | No matrix adaptation (C = I): only the step size is adapted. Disables `Modules.active`, rejects `Settings.C0`. O(d lambda) per generation. |
| `MATRIX` | MA-ES (Beyer & Sendhoff 2017): adapts a matrix M with C = M M^T directly, without the evolution path p_c (its rank-one term reads p_sigma) and without eigendecomposition; O(d^2 lambda) per generation: at d = 1000, 956 us against 6.0 ms per generation for COVARIANCE (docs/performance.md, about 6 times faster). With a step-size rule other than CSA the rank-one term is skipped in a generation with h_sigma = 0 (#160). With `Modules.active` it uses the active update of COVARIANCE, written in M (MA-ES has none; #73). |
| `SEPARABLE` | sep-CMA-ES (Ros & Hansen 2008): adapts only the diagonal of C, O(d lambda) per generation, with pycma's separable learning rates c1 and cmu. For separable problems and very high dimension; it cannot learn correlations between variables. Of `Settings.C0` it uses the diagonal. |
| `CHOLESKY` | Cholesky-CMA-ES (Krause et al. 2016): adapts the lower triangular Cholesky factor of C with rank-one updates of the factor, without decomposing C; O(d^2 lambda) per generation. |
| `CMSA` | CMSA-ES (Beyer & Sendhoff 2008), covariance matrix self-adaptation: C is an exponential average of the selected y y^T with time constant 1 + d (d + 1) / (2 mueff), factorized with a Cholesky decomposition every generation. Meant for StepSizeAdaptation.SA; makes the default mu lambda / 4. Disables `Modules.active`: CMSA has no active variant. |
| `COVARIANCE_NO_EIGV` | The covariance update of COVARIANCE, with C factorized by a Cholesky decomposition instead of an eigendecomposition (cheaper), at the same interval. The path p_sigma uses the inverse Cholesky factor instead of C^-1/2 (the same up to a rotation); with CSA it gave the same results as COVARIANCE on the sphere and a rotated ellipsoid (d = 5 to 20; the deep dive on the matrix adaptations). There is no published counterpart. Chosen automatically when COVARIANCE is combined with a step-size adaptation other than CSA, and for lambda0 = 1. |
| `NATURAL_GRADIENT` | xNES (Glasmachers et al. 2010), exponential natural evolution strategies: A is updated with the matrix exponential of the natural gradient, A <- A expm(cc G) with G = sum_i w_i (z_i z_i^T - I) made trace-free (its trace goes to sigma) and xNES's cc = eta_B / 2 = (9 + 3 ln d) / (10 d sqrt(d)); a low-rank form keeps it O(d^2 k) when the k samples of the gradient (mu, or lambda with `Modules.active`) are fewer than d and d >= 16. With StepSizeAdaptation.XNES also xNES's update of sigma, with xNES's learning rate cs = eta_sigma (the other step-size adaptations keep their own cs; #72), and with RecombinationWeights.UTILITY (also selected by `Modules.active` with the default weights) xNES's utilities for the mean and the matrix: all three together are xNES. |
<!-- /options -->
