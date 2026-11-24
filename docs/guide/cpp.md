# The C++ library

modcma is a C++17 library with a Python interface; the library can be used from C++ directly. The
API is the same as in Python: `parameters::Modules`, `parameters::Settings`, `ModularCMAES` with
`run` (`operator()`), `step`, `ask` and `tell`; points are Eigen vectors, and `ask()` returns the
points as the *columns* of a d × k matrix.

```cpp
--8<-- "examples/cpp/sphere.cpp"
```

`include/functions.hpp` has a few test functions (sphere, ellipse, Rosenbrock, Rastrigin) and a
random rotation; `examples/cpp/functions.cpp` runs them, shifted and rotated, with several matrix
adaptations.

## Building

The CMake target is `modcma_core` (a static library; its include directories and compile options
come with it). Add the repository as a subdirectory, or fetch it:

```cmake
--8<-- "examples/cpp/CMakeLists.txt"
```

```cmake
include(FetchContent)
FetchContent_Declare(modcma GIT_REPOSITORY https://github.com/IOHprofiler/ModularCMAES.git
                            GIT_TAG master)
FetchContent_MakeAvailable(modcma)
target_link_libraries(your_program PRIVATE modcma_core)
```

The dependencies are bundled (Eigen in `external/`). CMake options:

| option | default | effect |
|---|---|---|
| `MODCMA_LAPACK` | ON on macOS, OFF elsewhere | link LAPACK and BLAS (`find_package(LAPACK)`: Accelerate, OpenBLAS, ...) for the linear algebra in high dimension; without it, Eigen only |
| `MODCMA_NATIVE` | OFF | compile for the host CPU (`-march=native`) |
| `MODCMA_PYTHON` | OFF | also build the Python extension (needs Python and nanobind) |

The repository also builds `modcma_tests` (the C++ unit tests, `ctest`) and
`modcma_bench` (the benchmark harness of [the benchmarks](../performance.md)).

## Notes

* The objective is a `FunctionType`, `std::function<Float(Eigen::Ref<const Vector>)>`; `Float` is
  `double`.
* `rng::set_seed(s)` seeds the global generator; `settings.seed` gives an optimizer its own.
* The library is not header-only; link `modcma_core`.
* The C++ API is not yet versioned separately from the Python package: pin a tag.
