## Implementations (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| modcma (C++) | **1.9 µs** | **5.5 µs** | **83 µs** | **226 µs** | **575 µs** | **2.0 ms** | **5.4 ms** |
| modcma (C++, Eigen only, no LAPACK) | 1.9 µs | 5.5 µs | 83 µs | 226 µs | 855 µs | 5.1 ms | 18.5 ms |
| pycma | 86 µs | 110 µs | 310 µs | 406 µs | 781 µs | 2.5 ms | 7.1 ms |
| cmaes (Optuna) | 44 µs | 75 µs | 339 µs | 863 µs | 3.1 ms | 27.1 ms | 179.1 ms |
| evosax (JAX, jit + scan) | 1.5 µs | 7.2 µs | 182 µs | 511 µs | 1.7 ms | 10.9 ms | 47.2 ms |
| EvoTorch (PyTorch) | 143 µs | 148 µs | 195 µs | 326 µs | 833 µs | 4.5 ms | 13.5 ms |

## Speed-up

| time / time of modcma | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| pycma | 46× | 20× | 3.7× | 1.8× | 1.4× | 1.3× | 1.3× |
| cmaes (Optuna) | 24× | 14× | 4.1× | 3.8× | 5.4× | 14× | 33× |
| evosax (JAX, jit + scan) | 0.8× | 1.3× | 2.2× | 2.3× | 3.0× | 5.6× | 8.8× |
| EvoTorch (PyTorch) | 77× | 27× | 2.4× | 1.4× | 1.4× | 2.3× | 2.5× |

## Matrix adaptations (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| COVARIANCE (default) | **1.9 µs** | **5.5 µs** | **83 µs** | **226 µs** | **575 µs** | **2.0 ms** | **5.4 ms** |
| COVARIANCE, Eigen only (no LAPACK) | 1.9 µs | 5.5 µs | 83 µs | 226 µs | 855 µs | 5.1 ms | 18.5 ms |
| COVARIANCE, fast_sampling | 1.8 µs | 5.0 µs | 78 µs | 216 µs | 542 µs | 1.9 ms | 5.3 ms |
| COVARIANCE_NO_EIGV | 1.8 µs | 3.3 µs | 18 µs | 44 µs | 89 µs | 337 µs | 994 µs |
| MATRIX (MA-ES) | 1.7 µs | 2.9 µs | 14 µs | 42 µs | 129 µs | 643 µs | 2.3 ms |
| CHOLESKY | 1.7 µs | 3.5 µs | 18 µs | 47 µs | 132 µs | 693 µs | 2.6 ms |
| CMSA | 1.8 µs | 3.2 µs | 18 µs | 56 µs | 121 µs | 594 µs | 2.4 ms |
| NATURAL_GRADIENT (xNES) | 2.1 µs | 4.5 µs | 19 µs | 51 µs | 159 µs | 833 µs | 2.8 ms |
| SEPARABLE | 1.6 µs | 2.7 µs | 10 µs | 26 µs | 51 µs | 135 µs | 285 µs |
| NONE (step size only) | 1.6 µs | 2.6 µs | 10 µs | 26 µs | 50 µs | 132 µs | 280 µs |

## Python objective: vectorized vs per point (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| vectorized (one call per generation) | **1.9 µs** | **5.5 µs** | **83 µs** | **226 µs** | **575 µs** | **2.0 ms** | **5.4 ms** |
| one call per point | 5.6 µs | 12 µs | 95 µs | 234 µs | 582 µs | 2.0 ms | 5.5 ms |

## Threads

| threads | generations/s | speed-up |
|---|---|---|
| 1 | 175,755 | 1.00× |
| 2 | 288,637 | 1.64× |
| 4 | 491,025 | 2.79× |
| 6 | 566,334 | 3.22× |
| 8 | 673,441 | 3.83× |
| 10 | 756,239 | 4.30× |
