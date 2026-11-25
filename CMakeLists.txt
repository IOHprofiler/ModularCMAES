cmake_minimum_required(VERSION 3.12)

project(modcma C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# The library code, shared by the Python extension and the executables below
add_library(modcma_core STATIC ${SRC_FILES})
set_target_properties(modcma_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Programs that link the library compile its headers, which need C++17 too
target_compile_features(modcma_core PUBLIC cxx_std_17)
target_include_directories(modcma_core PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/external
)

option(MODCMA_NATIVE "Compile for the host CPU (-march=native); not for distributed binaries" OFF)
# LAPACK for the eigendecomposition (constants.eigen_solver): on by default on macOS, where Apple
# Accelerate is part of the system. Elsewhere the Python package uses scipy's LAPACK at run time
# (src/lapack.cpp), so the wheels need no LAPACK of their own.
if (APPLE)
  set(MODCMA_LAPACK_DEFAULT ON)
else()
  set(MODCMA_LAPACK_DEFAULT OFF)
endif()
option(MODCMA_LAPACK "Link LAPACK for the eigendecomposition" ${MODCMA_LAPACK_DEFAULT})

if (MODCMA_LAPACK)
  # Accelerate on macOS; OpenBLAS or the reference LAPACK elsewhere
  find_package(LAPACK REQUIRED)
  target_link_libraries(modcma_core PUBLIC ${LAPACK_LIBRARIES})
  target_compile_definitions(modcma_core PUBLIC MODCMA_LAPACK)
endif()

if (MSVC)
  target_compile_options(modcma_core PUBLIC /bigobj)
else()
  target_compile_options(modcma_core PUBLIC -fno-math-errno)
  if (MODCMA_NATIVE)
    target_compile_options(modcma_core PUBLIC -march=native)
  endif()
endif()

# The Python extension modcma.c_maes.cmaescpp (nanobind). scikit-build-core (pyproject.toml) sets
# SKBUILD and builds only this; MODCMA_PYTHON=ON builds it next to the C++ targets.
option(MODCMA_PYTHON "Build the Python extension" OFF)
if (SKBUILD OR MODCMA_PYTHON)
  # For an abi3 wheel (wheel.py-api in pyproject.toml) scikit-build-core sets
  # SKBUILD_SABI_COMPONENT to Development.SABIModule
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  file(GLOB BINDING_SOURCES ${PROJECT_SOURCE_DIR}/bindings/*.cpp)
  # NOMINSIZE: keep -O3 (nanobind's default -Os slows the bindings down, measured in
  # IOHexperimenter). STABLE_ABI: one wheel for CPython 3.12+ (ignored for older versions and
  # free-threaded builds). FREE_THREADED: on a free-threaded interpreter the module does not
  # re-enable the GIL; every optimizer has its own random generator (dev/overhaul-2.0.md, T4.1), so
  # optimizers can run in parallel threads (ignored on other interpreters).
  nanobind_add_module(cmaescpp NOMINSIZE STABLE_ABI FREE_THREADED ${BINDING_SOURCES})
  target_link_libraries(cmaescpp PRIVATE modcma_core)
  install(TARGETS cmaescpp LIBRARY DESTINATION modcma/c_maes)
endif()

if (SKBUILD)
  return()
endif()

# C++ unit tests for what Python cannot reach (ctest)
enable_testing()
add_executable(modcma_tests ${PROJECT_SOURCE_DIR}/tests/cpp/test_main.cpp)
target_link_libraries(modcma_tests PRIVATE modcma_core)
add_test(NAME modcma_tests COMMAND modcma_tests)

# Benchmark harness (dev/overhaul-2.0.md, T0.2): time per generation and heap allocations per evaluation
add_executable(modcma_bench ${PROJECT_SOURCE_DIR}/bench/bench.cpp)
target_link_libraries(modcma_bench PRIVATE modcma_core)
if (APPLE)
  # dyld only honours __interpose tuples from a dylib, not from the executable itself
  add_library(modcma_malloc_count SHARED ${PROJECT_SOURCE_DIR}/bench/malloc_count.c)
  target_link_libraries(modcma_bench PRIVATE modcma_malloc_count)
else()
  target_sources(modcma_bench PRIVATE ${PROJECT_SOURCE_DIR}/bench/malloc_count.c)
endif()
