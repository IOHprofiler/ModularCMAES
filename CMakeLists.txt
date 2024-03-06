cmake_minimum_required(VERSION 3.12)

# Assign a name to the global project
project(modcma)

# Set the version of C/C++ (here C++17)
set(CMAKE_CXX_STANDARD 17)

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
list(FILTER  SRC_FILES  EXCLUDE REGEX ".*interface.cpp$")

add_executable(main ${SRC_FILES})

add_subdirectory(${PROJECT_SOURCE_DIR}/../IOHexperimenter ${CMAKE_CURRENT_BINARY_DIR}/ioh)

target_include_directories(main PUBLIC 
	${PROJECT_SOURCE_DIR}/include 
	${PROJECT_SOURCE_DIR}/external
	${PROJECT_SOURCE_DIR}/../IOHexperimenter/include
)

target_link_libraries(main PUBLIC ioh)

if (MSVC)
  target_compile_options(main PRIVATE /bigobj)
endif()
