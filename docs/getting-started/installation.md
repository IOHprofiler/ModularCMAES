# Installation

```bash
pip install modcma
```

There are wheels for Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows (x64, ARM64), for
CPython 3.10 to 3.14, including the free-threaded 3.14t, on which the optimizers run in parallel
threads without the GIL. The dependencies are numpy, scipy and ConfigSpace.

Check the installation:

```python
import modcma

print(modcma.__version__)
print(modcma.utils.build_info())  # compiler, FMA, and where LAPACK comes from
```

## LAPACK and BLAS

In high dimension (from d ≈ 100) the linear algebra uses LAPACK and BLAS, several times faster than
the built-in Eigen kernels ([performance](../guide/performance.md)). Nothing needs to be installed
for this:

| platform | LAPACK and BLAS | `build_info()["lapack"]` |
|---|---|---|
| macOS | Apple Accelerate, linked into the wheel | `"linked"` |
| Linux, Windows | scipy's (OpenBLAS in scipy's wheels), handed to the library at import | `"scipy"` |
| a C++ build without LAPACK | Eigen only | `"none"` |

!!! tip "Many optimizers at once"
    OpenBLAS uses every core by default. When you run many optimizers in parallel (processes or
    threads), set `OPENBLAS_NUM_THREADS=1` (or use `threadpoolctl`), so they do not compete for
    the cores.

## From source

A C++17 compiler and CMake ≥ 3.18 are needed; everything else (Eigen, nanobind) is fetched or
bundled.

```bash
git clone https://github.com/IOHprofiler/ModularCMAES.git
cd ModularCMAES
python -m pip install -e .
python -m unittest discover -s tests -t .
```

See [contributing](../contributing.md) for the development workflow (rebuilding, the C++ tests,
stubs), and [the C++ library](../guide/cpp.md) to use modcma from C++.

## modcma 1.x

Version 2.0 removed the pure-Python implementation (`modcma.ModularCMAES(func, d, ...)`,
`AskTellCMAES`, `evaluate_bbob`). It remains available as `pip install "modcma<2"` and on the
[`legacy-python`](https://github.com/IOHprofiler/ModularCMAES/tree/legacy-python) branch; see
[migrating from 1.x](../migration.md).
