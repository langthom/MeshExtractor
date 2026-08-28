

add_library(pme
  Chunk.h
  Chunkifier.h
  Chunkifier.cpp
)

# Enable OpenMP in our library.
find_package(OpenMP REQUIRED)
target_link_libraries(pme PRIVATE OpenMP::OpenMP_CXX)

# If the Debug build is invoked, set the number of threads to 1.
target_compile_definitions(pme PRIVATE "$<$<CONFIG:Debug>:OMP_NUM_THREADS=1>")


