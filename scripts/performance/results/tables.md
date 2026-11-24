## Implementations (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| modcma (C++) | **1.9 µs** | **5.6 µs** | **83 µs** | **220 µs** | **520 µs** | **1.6 ms** | **4.2 ms** |
| modcma (C++, Eigen only, no LAPACK) | 1.9 µs | 5.5 µs | 83 µs | 227 µs | 857 µs | 5.1 ms | 18.4 ms |
| pycma | 87 µs | 111 µs | 312 µs | 409 µs | 792 µs | 2.6 ms | 7.1 ms |
| cmaes (Optuna) | 45 µs | 75 µs | 340 µs | 865 µs | 3.1 ms | 27.1 ms | 176.8 ms |
| evosax (JAX, jit + scan) | 1.6 µs | 7.2 µs | 182 µs | 512 µs | 1.7 ms | 10.9 ms | 48.9 ms |
| EvoTorch (PyTorch) | 148 µs | 150 µs | 197 µs | 330 µs | 855 µs | 4.7 ms | 13.6 ms |

## Speed-up

| time / time of modcma | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| pycma | 46× | 20× | 3.8× | 1.9× | 1.5× | 1.6× | 1.7× |
| cmaes (Optuna) | 24× | 13× | 4.1× | 3.9× | 6.0× | 17× | 42× |
| evosax (JAX, jit + scan) | 0.9× | 1.3× | 2.2× | 2.3× | 3.3× | 6.7× | 12× |
| EvoTorch (PyTorch) | 78× | 27× | 2.4× | 1.5× | 1.6× | 2.9× | 3.2× |

## Matrix adaptations (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| COVARIANCE (default) | **1.9 µs** | **5.6 µs** | **83 µs** | **220 µs** | **520 µs** | **1.6 ms** | **4.2 ms** |
| COVARIANCE, Eigen only (no LAPACK) | 1.9 µs | 5.5 µs | 83 µs | 227 µs | 857 µs | 5.1 ms | 18.4 ms |
| COVARIANCE, fast_sampling | 1.8 µs | 5.0 µs | 79 µs | 208 µs | 497 µs | 1.6 ms | 4.1 ms |
| COVARIANCE_NO_EIGV | 1.8 µs | 3.3 µs | 18 µs | 41 µs | 72 µs | 223 µs | 554 µs |
| MATRIX (MA-ES) | 1.6 µs | 2.9 µs | 14 µs | 37 µs | 85 µs | 330 µs | 953 µs |
| CHOLESKY | 1.8 µs | 3.5 µs | 18 µs | 45 µs | 117 µs | 592 µs | 2.2 ms |
| CMSA | 1.8 µs | 3.3 µs | 18 µs | 52 µs | 104 µs | 483 µs | 1.9 ms |
| NATURAL_GRADIENT (xNES) | 2.0 µs | 4.6 µs | 19 µs | 40 µs | 88 µs | 329 µs | 866 µs |
| SEPARABLE | 1.6 µs | 2.8 µs | 10 µs | 26 µs | 51 µs | 135 µs | 285 µs |
| NONE (step size only) | 1.6 µs | 2.7 µs | 10 µs | 26 µs | 50 µs | 132 µs | 280 µs |

## Python objective: vectorized vs per point (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| vectorized (one call per generation) | **1.9 µs** | **5.6 µs** | **83 µs** | **220 µs** | **520 µs** | **1.6 ms** | **4.2 ms** |
| one call per point | 5.7 µs | 13 µs | 94 µs | 232 µs | 539 µs | 1.7 ms | 4.4 ms |

## Threads

| threads | generations/s | speed-up |
|---|---|---|
| 1 | 175,755 | 1.00× |
| 2 | 288,637 | 1.64× |
| 4 | 491,025 | 2.79× |
| 6 | 566,334 | 3.22× |
| 8 | 673,441 | 3.83× |
| 10 | 756,239 | 4.30× |
