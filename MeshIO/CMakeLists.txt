
add_library(pme_mesh_io
  PLYStreamWriter.h
  PLYStreamWriter.cpp
)

# Only for the ChunkMesh convenience overload; the streaming interface itself deals in plain
# vectors and knows nothing of the extraction.
target_link_libraries(pme_mesh_io PUBLIC pme)
