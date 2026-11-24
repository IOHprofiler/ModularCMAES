# Performance

A modcma generation takes microseconds in low dimension and a few milliseconds at d = 1000; the
[benchmarks](../performance.md) compare it with pycma, `cmaes`, evosax and EvoTorch, and the matrix
adaptations with each other. This page is about getting the most out of it.

## Evaluate in batches

For a cheap objective, the Python call per point costs more than the optimizer (about 0.6 µs per
call). Write the objective for a `(k, d)` array and pass `vectorized=True`:

```python
import numpy as np
import modcma

w = 10.0 ** (6 * np.arange(20) / 19)

def ellipsoid_batch(X):
    return (X * X) @ w

es = modcma.ModularCMAES(modcma.Settings(20, budget=20_000, seed=1))
es.run(ellipsoid_batch, vectorized=True)
```

This is 2–3× faster than one call per point up to d ≈ 10. For an expensive objective, evaluate the
batches in parallel with [ask–tell](../getting-started/ask-tell.md).

## Choose the matrix adaptation

In high dimension the update of the full covariance matrix dominates. The alternatives are several
times cheaper per generation (d = 1000: default 4.2 ms; MA-ES 0.95 ms; `COVARIANCE_NO_EIGV`
0.55 ms; `SEPARABLE` 0.29 ms), at the price of a different, sometimes slower, search: see
[matrix adaptation](matrix-adaptation.md).

## LAPACK and BLAS

From d = 100 to 120 on, the eigendecomposition, Cholesky factorizations, matrix inverses and
matrix products use LAPACK and BLAS ([installation](../getting-started/installation.md#lapack-and-blas)).
The choice is visible and adjustable with the global `modcma.constants`:

```python
print(modcma.utils.build_info()["lapack"])  # "linked", "scipy" or "none"
modcma.constants.eigen_solver      # -1 automatic (default), 0 Eigen, 1 dsyevd, 2 dsyevr
modcma.constants.cholesky_solver   # -1 automatic (default), 0 Eigen, 1 dpotrf
modcma.constants.lapack_min_dim    # 120: eigendecomposition and Cholesky with LAPACK from here on
modcma.constants.blas              # -1 automatic (default), 0 Eigen, 1 BLAS
modcma.constants.blas_min_dim      # 100: products and inverses with BLAS/LAPACK from here on
```

Set `eigen_solver`, `cholesky_solver` and `blas` to 0 for results that do not depend on the LAPACK
library ([reproducibility](reproducibility.md)).

## Fast sampling

`Settings(..., fast_sampling=True)` draws the normal samples with xoshiro256++ and a ziggurat
instead of mt19937 and the inverse normal CDF. Sampling dominates the cheap matrix adaptations
(`SEPARABLE`, `NONE`, MA-ES), which become up to twice as fast per generation; the default gains up
to 10 %. It is a different random stream, so results differ from the default (statistically
equivalent); it is off by default.

## Many runs

Benchmarks and tuning run many independent optimizations. Each optimizer has its own random
generator and no shared state, so they can run in threads (truly parallel on free-threaded Python
3.14t) or processes; set `OPENBLAS_NUM_THREADS=1` in either case. See
[reproducibility and threads](reproducibility.md#threads).
