cmake_minimum_required(VERSION 3.12)

project(modcma C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
list(FILTER SRC_FILES EXCLUDE REGEX ".*(interface|main)\\.cpp$")

# The library code, shared by the executables below
add_library(modcma_core STATIC ${SRC_FILES})
target_include_directories(modcma_core PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/external
)

option(MODCMA_NATIVE "Compile for the host CPU (-march=native); not for distributed binaries" OFF)

if (MSVC)
  target_compile_options(modcma_core PUBLIC /bigobj)
else()
  target_compile_options(modcma_core PUBLIC -fno-math-errno)
  if (MODCMA_NATIVE)
    target_compile_options(modcma_core PUBLIC -march=native)
  endif()
endif()

# Example/experiment driver
add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(main PRIVATE modcma_core)

# Benchmark harness (TASKS.md, T0.2): time per generation and heap allocations per evaluation
add_executable(modcma_bench ${PROJECT_SOURCE_DIR}/bench/bench.cpp)
target_link_libraries(modcma_bench PRIVATE modcma_core)
if (APPLE)
  # dyld only honours __interpose tuples from a dylib, not from the executable itself
  add_library(modcma_malloc_count SHARED ${PROJECT_SOURCE_DIR}/bench/malloc_count.c)
  target_link_libraries(modcma_bench PRIVATE modcma_malloc_count)
else()
  target_sources(modcma_bench PRIVATE ${PROJECT_SOURCE_DIR}/bench/malloc_count.c)
endif()
