

add_library(pme
  Chunk.h
  Chunkifier.h
  Chunkifier.cpp
  SlabSchedule.h
  SlabSchedule.cpp
  MeshAssembler.h
  MeshAssembler.cpp
  CpuChunkMeshExtractor.h
  CpuChunkMeshExtractor.cpp
  # GPU surface extraction.
  ChunkMesh.h
  CudaUtils.h
  MarchingCubesTables.h
  CudaChunkMeshExtractor.h
  CudaChunkMeshExtractor.cu
)

# The CUDA sources are compiled as C++17 just like the host ones, and the tables are kept inside
# the single translation unit that uses them so that no device link step is needed.
set_target_properties(pme PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE  ON
)
target_compile_features(pme PUBLIC cxx_std_17)

# Enable OpenMP in our library.
find_package(OpenMP REQUIRED)
target_link_libraries(pme PRIVATE OpenMP::OpenMP_CXX)

# If the Debug build is invoked, set the number of threads to 1.
target_compile_definitions(pme PRIVATE "$<$<CONFIG:Debug>:OMP_NUM_THREADS=1>")


