cmake_minimum_required(VERSION 4.4.0)
project(ParallelMeshExtractor
  VERSION 0.0.0
  LANGUAGES CXX CUDA
)

option(PME_BUILD_APP   "Build command line app." OFF)
option(PME_BUILD_TESTS "Build unit tests."       OFF)


# Set the C++ standard *globally* for all subdirectories.
set(CMAKE_CXX_STANDARD 17)

# ------------------------------------------------------------------------------------------------ #

enable_testing()

add_subdirectory(MeshExtraction)
add_subdirectory(VolumeIO)

if(PME_BUILD_APP)
  add_subdirectory(app)
endif()

if(PME_BUILD_TESTS)
  add_subdirectory(tests)
endif()




