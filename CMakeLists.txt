cmake_minimum_required(VERSION 3.12)

# Assign a name to the global project
project(modcma)

# Set the version of C/C++ (here C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
list(FILTER  SRC_FILES  EXCLUDE REGEX ".*interface.cpp$")
 
add_executable(main ${SRC_FILES}) 

target_include_directories(main PUBLIC 
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/external
)

if (MSVC)
  target_compile_options(main PRIVATE /bigobj)
endif() 
 