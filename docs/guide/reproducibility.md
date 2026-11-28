# Reproducibility and threads

## Seeds

Every optimizer has its own random number generator.

* `Settings(..., seed=s)` seeds the optimizer's own generator: the same seed gives the same run.
* Without a seed, the optimizer copies the global generator when it is created, so
  `modcma.set_seed(s)` before creating it has the same effect (the way modcma 1.x and most
  scripts did it, as `c_maes.utils.set_seed(s)`).

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
for bit, with every module. So does a run that is saved with `pickle` and loaded, at any point
([saving and resuming a run](checkpointing.md)): the random generator is part of its state.

## How far identical results go

The same seed gives the same run on the same platform and toolchain (operating system, CPU,
compiler, C library). Across platforms, results can differ in the last bits, and runs then
diverge; the results are statistically the same. The last bits differ because:

* the compiler and the CPU decide whether fused multiply–add instructions are used (`build_info()`
  shows it);
* from dimension 100 (the default thresholds) the linear algebra uses LAPACK and BLAS, whose
  results depend on the library (Accelerate, OpenBLAS, MKL);
* `fast_sampling=True` uses a different random stream (xoshiro256++ and a ziggurat instead of
  mt19937 and the inverse normal CDF).

The random numbers of the samplers are the same on every platform: they come from mt19937
(xoshiro256++ with `fast_sampling`), turned into uniform and normal numbers, bounded integers and
permutations by the library's own code, not by the standard library's distributions, whose results
differ between libc++, libstdc++ and MSVC. Before 2.0 the `HALTON` and `SOBOL` samplers used `std::shuffle` and
`std::uniform_int_distribution`, so their runs differed between macOS and Linux from the first
generation (#116). The libm functions (`exp`, `log`, `cos`) also differ in the last bit between
the C libraries (Apple's and glibc) for a small share of arguments.

**An ambiguous eigenbasis amplifies these last bits** (#165). When C has equal eigenvalues, the
eigenvectors of that eigenspace are not unique, and the eigensolver's choice depends on the last
bits of C: two platforms that agree on C to a relative 1e-16 can return eigenvectors rotated by
tens of degrees against each other. `COVARIANCE` samples y = B D z with them (Hansen 2016,
Eqs. (38), (39); pycma does the same), so the samples then differ by O(1). Each generation adds µ
selected steps to C, and the path p_c lies in their span, so at the first decomposition (generation
2) C − aI has rank at most 2µ: with the default settings C has a repeated eigenvalue from d = 12 on
(d ≥ 2µ + 2; λ = 11, µ = 5), and runs on two platforms can differ from the third generation on. Below that, and with the samplers that
use a Cholesky factor (`COVARIANCE_NO_EIGV`, `CHOLESKY`, `CMSA`) or their own matrix (`MATRIX`,
`NATURAL_GRADIENT`), the last-bit differences grow slowly. The (1+1)-ES with
`always_compute_eigv=True` has the same amplification (C has a repeated eigenvalue until d − 1
successes). Measured between macOS (clang 21, arm64) and Linux (gcc 13, x86-64): the default
configuration differs by more than a relative 1e-3 from generation 3 on the sphere at d = 12 to
40, and on the ellipsoid at d = 20 and 40.

In the golden trajectories (d ≤ 10; macOS clang 21 against Linux gcc 13, σ and the best f at
generation 10 within a relative 1e-12), 238 of 241 runs with the default sampler agree, 15 of 16
with `HALTON` and 12 of 14 with `SOBOL`; the rest differ in the last bits.

So for results that agree across platforms up to these last bits, fix the seed and stay below
d = 12 with `COVARIANCE` and CSA (or use another matrix adaptation), and below the LAPACK
thresholds; in high dimension, set `modcma.constants.eigen_solver = 0`, `cholesky_solver = 0` and
`blas = 0` to use Eigen only ([performance](performance.md)). On the same platform and toolchain
the run is always the same. The tests of modcma keep golden trajectories per toolchain to detect
any change.

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
