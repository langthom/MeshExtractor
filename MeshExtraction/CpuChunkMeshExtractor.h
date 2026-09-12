#pragma once

#include <cstdint>
#include <vector>

#include "Chunk.h"
#include "ChunkMesh.h"
#include "Chunkifier.h"

namespace parallel_mesh_extractor {

  /// Extracts the isosurface of a single chunk on the host, producing exactly the mesh the CUDA
  /// extractor produces.
  ///
  /// The two differ only in how they reach that result. The GPU classifies every edge and every
  /// cell up front, prefix sums the counts and then fills the buffers, because it has to know
  /// where each thread writes before any of them run. On the host nothing of the sort is needed:
  /// the cells are walked in order and a grid edge is given its vertex the first time a triangle
  /// asks for it, which removes both scans. The vertices therefore come out in a different order
  /// than on the device -- same set, same triangles, different numbering.
  ///
  /// An instance owns the scratch it needs and is *not* thread safe. Extracting chunks in parallel
  /// means one extractor per thread, which is also how the host beats the device here: a chunk is
  /// far too small to fill a GPU, but twenty of them keep twenty cores busy.
  class CpuChunkMeshExtractor {
  public:

    /// Same layout constants as the CUDA extractor, and for the same reasons.
    static constexpr std::uint32_t CellDim   = Chunkifier::CoreSize;
    static constexpr std::uint32_t CornerDim = CellDim + 1;

    explicit CpuChunkMeshExtractor(float isoThreshold);

    /// Extracts the cells the chunk owns. "out" is cleared first and stays empty when the chunk
    /// carries no surface, which is a normal outcome rather than an error.
    void Extract(Chunkifier::DataChunk const& chunk, ChunkMesh& out);

  private:
    float IsoThreshold;

    /// Vertex index already assigned to each grid edge, or -1. Held across calls so that extracting
    /// many chunks allocates once.
    std::vector<std::int32_t> EdgeVertexIndex;
  };

} // namespace parallel_mesh_extractor
