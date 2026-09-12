cmake_minimum_required(VERSION 4.4.0)

# Specify CUDA architectures as required since CMake 3.18 (policy CMP0104).
# Must happen before the call to "project".
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")

project(ParallelMeshExtractor
  VERSION 0.0.0
  LANGUAGES CXX CUDA
)

option(PME_BUILD_APP   "Build command line app." OFF)
option(PME_BUILD_TESTS "Build unit tests."       OFF)


# Set the C++ standard *globally* for all subdirectories, for the host and the device compiler
# alike, so that the CUDA sources see the same language as everything else.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ------------------------------------------------------------------------------------------------ #

enable_testing()

add_subdirectory(MeshExtraction)
add_subdirectory(MeshIO)
add_subdirectory(VolumeIO)

if(PME_BUILD_APP)
  add_subdirectory(app)
endif()

if(PME_BUILD_TESTS)
  add_subdirectory(tests)
endif()




