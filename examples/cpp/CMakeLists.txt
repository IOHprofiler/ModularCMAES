# A C++ program that uses modcma: build it with
#   cmake -S examples/cpp -B build-example && cmake --build build-example && ./build-example/sphere
cmake_minimum_required(VERSION 3.18)
project(modcma_example CXX)

# The library from this repository (or from a copy of it, e.g. with FetchContent)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. modcma EXCLUDE_FROM_ALL)

add_executable(sphere sphere.cpp)
target_link_libraries(sphere PRIVATE modcma_core)

# The test functions of include/functions.hpp, shifted and rotated
add_executable(functions functions.cpp)
target_link_libraries(functions PRIVATE modcma_core)
