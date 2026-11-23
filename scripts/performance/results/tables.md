## Implementations (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| modcma (C++) | **1.9 µs** | **5.6 µs** | **83 µs** | **226 µs** | **899 µs** | **5.1 ms** | **18.4 ms** |
| pycma | 86 µs | 111 µs | 313 µs | 407 µs | 781 µs | 2.5 ms | 7.0 ms |
| cmaes (Optuna) | 44 µs | 76 µs | 341 µs | 864 µs | 3.1 ms | 27.1 ms | 174.2 ms |
| evosax (JAX, jit + scan) | 1.6 µs | 7.3 µs | 181 µs | 512 µs | 1.7 ms | 10.9 ms | 47.0 ms |
| EvoTorch (PyTorch) | 150 µs | 156 µs | 203 µs | 335 µs | 853 µs | 4.6 ms | 13.3 ms |

## Speed-up

| time / time of modcma | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| pycma | 45× | 20× | 3.8× | 1.8× | 0.9× | 0.5× | 0.4× |
| cmaes (Optuna) | 23× | 14× | 4.1× | 3.8× | 3.5× | 5.3× | 9.4× |
| evosax (JAX, jit + scan) | 0.8× | 1.3× | 2.2× | 2.3× | 1.9× | 2.1× | 2.5× |
| EvoTorch (PyTorch) | 79× | 28× | 2.4× | 1.5× | 0.9× | 0.9× | 0.7× |

## Matrix adaptations (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| COVARIANCE (default) | **1.9 µs** | **5.6 µs** | **83 µs** | **226 µs** | **899 µs** | **5.1 ms** | **18.4 ms** |
| COVARIANCE, fast_sampling | 1.8 µs | 5.1 µs | 79 µs | 216 µs | 876 µs | 5.0 ms | 18.3 ms |
| COVARIANCE_NO_EIGV | 1.8 µs | 3.3 µs | 18 µs | 44 µs | 109 µs | 499 µs | 1.7 ms |
| MATRIX (MA-ES) | 1.7 µs | 3.0 µs | 14 µs | 42 µs | 129 µs | 644 µs | 2.3 ms |
| CHOLESKY | 1.8 µs | 3.5 µs | 18 µs | 47 µs | 132 µs | 691 µs | 2.6 ms |
| CMSA | 1.8 µs | 3.3 µs | 18 µs | 56 µs | 182 µs | 1.4 ms | 9.1 ms |
| NATURAL_GRADIENT (xNES) | 2.0 µs | 4.4 µs | 56 µs | 311 µs | 2.1 ms | 28.7 ms | 224.0 ms |
| SEPARABLE | 1.6 µs | 2.7 µs | 10 µs | 26 µs | 51 µs | 135 µs | 285 µs |
| NONE (step size only) | 1.6 µs | 2.7 µs | 10 µs | 26 µs | 50 µs | 132 µs | 280 µs |

## Python objective: vectorized vs per point (µs per generation, median)

| implementation | d=2 | d=10 | d=50 | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|---|---|---|
| vectorized (one call per generation) | **1.9 µs** | **5.6 µs** | **83 µs** | **226 µs** | **899 µs** | **5.1 ms** | **18.4 ms** |
| one call per point | 5.7 µs | 13 µs | 95 µs | 235 µs | 879 µs | 5.1 ms | 18.5 ms |

## Threads

| threads | generations/s | speed-up |
|---|---|---|
| 1 | 175,755 | 1.00× |
| 2 | 288,637 | 1.64× |
| 4 | 491,025 | 2.79× |
| 6 | 566,334 | 3.22× |
| 8 | 673,441 | 3.83× |
| 10 | 756,239 | 4.30× |
