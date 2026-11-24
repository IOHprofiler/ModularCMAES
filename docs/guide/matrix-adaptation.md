# Matrix adaptation

The mutations of CMA-ES are drawn from N(0, σ² C): the covariance matrix C learns the shape of the
landscape (its scaling and correlations), which is what makes CMA-ES invariant to rotations and
efficient on ill-conditioned problems. `matrix_adaptation` chooses how C, or a factor A of it
(C = A Aᵀ), is learned.

## Which one to choose

| option | learns | cost per generation | time at d = 1000 | choose it for |
|---|---|---|---|---|
| `COVARIANCE` | full C | O(d²λ) + an O(d³) eigendecomposition every few generations | 4.2 ms | the default: the most robust choice up to a few hundred dimensions |
| `MATRIX` (MA-ES) | full M, C = M Mᵀ | O(d²λ) | 0.95 ms | high dimension, when the full covariance matters but CMA-ES is too slow |
| `NATURAL_GRADIENT` (xNES-like) | full A | O(d²λ) (exact low-rank update) | 0.87 ms | as MA-ES; natural-gradient based |
| `CHOLESKY` | Cholesky factor | O(d²λ) | 2.2 ms | as MA-ES |
| `COVARIANCE_NO_EIGV` | full C, Cholesky factor | O(d²λ) + O(d³) every few generations | 0.55 ms | step-size adaptations other than CSA (chosen automatically) |
| `CMSA` | full C, self-adapted | O(d²λ) + an O(d³) Cholesky factorization every generation | 1.9 ms | with `ssa=SA` (CMSA-ES) |
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
`CHOLESKY` and `NATURAL_GRADIENT`). See the [reference](../reference/matrix_adaptation.md).

## The options

<!-- options: MatrixAdaptationType -->
