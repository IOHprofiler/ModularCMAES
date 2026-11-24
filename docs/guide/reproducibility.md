# Reproducibility and threads

## Seeds

Every optimizer has its own random number generator.

* `Settings(..., seed=s)` seeds the optimizer's own generator: the same seed gives the same run.
* Without a seed, the optimizer copies the global generator when it is created, so
  `modcma.utils.set_seed(s)` before creating it has the same effect (the way modcma 1.x and
  most scripts did it).

```python
import numpy as np
import modcma

def sphere(x):
    return float(np.sum(x**2))

a = modcma.ModularCMAES(modcma.Settings(5, budget=2_000, seed=7))
b = modcma.ModularCMAES(modcma.Settings(5, budget=2_000, seed=7))
a.run(sphere)
b.run(sphere)
assert a.p.stats.global_best.y == b.p.stats.global_best.y
```

`run`, `step`, `ask`/`tell` and vectorized evaluation give the same results for the same seed, bit
for bit, with every module.

## How far identical results go

The same seed gives the same run on the same machine and build. Across machines, results can
differ in the last bits, and runs then diverge:

* the compiler and the CPU decide whether fused multiply–add instructions are used (`build_info()`
  shows it);
* from dimension 100 (the default thresholds) the linear algebra uses LAPACK and BLAS, whose
  results depend on the library (Accelerate, OpenBLAS, MKL);
* `fast_sampling=True` uses a different random stream (xoshiro256++ and a ziggurat instead of
  mt19937 and the inverse normal CDF).

For results that must be identical across platforms below the LAPACK thresholds, nothing needs to
be done beyond fixing the seed; in high dimension, set `modcma.constants.eigen_solver = 0`,
`cholesky_solver = 0` and `blas = 0` to use Eigen only ([performance](performance.md)). The tests
of modcma keep golden trajectories per compiler to detect any change.

## Threads

Optimizers do not share state, so independent optimizers can run in parallel threads. On the
free-threaded build of CPython (3.14t) they run truly in parallel; with the GIL, threads help only
when the objective releases it.

```python
from concurrent.futures import ThreadPoolExecutor

def run(seed):
    es = modcma.ModularCMAES(modcma.Settings(10, budget=5_000, seed=seed))
    es.run(lambda X: np.sum(X**2, axis=1), vectorized=True)
    return es.p.stats.global_best.y

with ThreadPoolExecutor(max_workers=4) as pool:
    print(list(pool.map(run, range(8))))
```

The results do not depend on the number of threads or their order: each optimizer has its own
seed. On the free-threaded build, 10 threads ran 4.3× more optimizer generations per second than one
on a 10-core laptop ([benchmarks](../performance.md#parallel-threads)). With OpenBLAS, set
`OPENBLAS_NUM_THREADS=1` so the BLAS threads do not compete with yours.

Global settings (`modcma.constants`, `set_seed`, the static tolerances of the restart criteria) are
shared by all optimizers: set them before starting threads.
