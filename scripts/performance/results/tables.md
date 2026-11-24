## Implementations (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| modcma (C++) | **1.9 µs** | **5.5 µs** | **83 µs** | **224 µs** | **573 µs** | **2.1 ms** | **6.0 ms** |
| modcma (C++, Eigen only, no LAPACK) | 1.9 µs | 5.5 µs | 83 µs | 231 µs | 960 µs | 7.7 ms | 32.0 ms |
| pycma | 87 µs | 111 µs | 312 µs | 409 µs | 792 µs | 2.6 ms | 7.1 ms |
| cmaes (Optuna) | 45 µs | 75 µs | 340 µs | 865 µs | 3.1 ms | 27.1 ms | 176.8 ms |
| evosax (JAX, jit + scan) | 1.6 µs | 7.2 µs | 182 µs | 512 µs | 1.7 ms | 10.9 ms | 48.9 ms |
| EvoTorch (PyTorch) | 148 µs | 150 µs | 197 µs | 330 µs | 855 µs | 4.7 ms | 13.6 ms |

## Speed-up

| time / time of modcma | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| pycma | 47× | 20× | 3.8× | 1.8× | 1.4× | 1.2× | 1.2× |
| cmaes (Optuna) | 24× | 14× | 4.1× | 3.9× | 5.5× | 13× | 30× |
| evosax (JAX, jit + scan) | 0.9× | 1.3× | 2.2× | 2.3× | 3.0× | 5.1× | 8.2× |
| EvoTorch (PyTorch) | 79× | 27× | 2.4× | 1.5× | 1.5× | 2.2× | 2.3× |

## Matrix adaptations (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| COVARIANCE (default) | **1.9 µs** | **5.5 µs** | **83 µs** | **224 µs** | **573 µs** | **2.1 ms** | **6.0 ms** |
| COVARIANCE, Eigen only (no LAPACK) | 1.9 µs | 5.5 µs | 83 µs | 231 µs | 960 µs | 7.7 ms | 32.0 ms |
| COVARIANCE, fast_sampling | 1.8 µs | 5.0 µs | 78 µs | 214 µs | 549 µs | 2.1 ms | 5.8 ms |
| COVARIANCE_NO_EIGV | 1.9 µs | 3.3 µs | 18 µs | 43 µs | 87 µs | 335 µs | 1.1 ms |
| MATRIX (MA-ES) | 1.6 µs | 3.0 µs | 14 µs | 37 µs | 86 µs | 332 µs | 956 µs |
| CHOLESKY | 1.8 µs | 3.5 µs | 18 µs | 45 µs | 118 µs | 593 µs | 2.2 ms |
| CMSA | 1.9 µs | 3.3 µs | 18 µs | 53 µs | 103 µs | 484 µs | 2.0 ms |
| NATURAL_GRADIENT (xNES) | 2.1 µs | 4.5 µs | 19 µs | 40 µs | 88 µs | 329 µs | 872 µs |
| SEPARABLE | 1.6 µs | 2.7 µs | 10 µs | 26 µs | 51 µs | 135 µs | 285 µs |
| NONE (step size only) | 1.6 µs | 2.7 µs | 10 µs | 26 µs | 51 µs | 133 µs | 281 µs |

## Python objective: vectorized vs per point (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| vectorized (one call per generation) | **1.9 µs** | **5.5 µs** | **83 µs** | **224 µs** | **573 µs** | **2.1 ms** | **6.0 ms** |
| one call per point | 5.5 µs | 13 µs | 95 µs | 236 µs | 573 µs | 2.1 ms | 5.9 ms |

## Threads

| threads | generations/s | speed-up |
|---|---|---|
| 1 | 175,755 | 1.00× |
| 2 | 288,637 | 1.64× |
| 4 | 491,025 | 2.79× |
| 6 | 566,334 | 3.22× |
| 8 | 673,441 | 3.83× |
| 10 | 756,239 | 4.30× |
