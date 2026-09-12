#pragma once

#include <cstdint>
#include <memory>
#include "Chunk.h"
#include "ChunkMesh.h"
#include "Chunkifier.h"

namespace parallel_mesh_extractor {

  /// Extracts the isosurface of a single chunk on the GPU, by Marching Cubes.
  ///
  /// The extractor owns every device buffer it needs and sizes them for the worst case a chunk can
  /// produce, so extracting a whole volume allocates exactly once and can never overflow. All CUDA
  /// state is hidden behind a pointer to implementation, which keeps this header compilable by a
  /// plain C++ compiler and lets the rest of the library stay free of CUDA includes.
  ///
  /// Only the cells a chunk *owns* are extracted, i.e. the ones based at the local indices
  /// [GhostWidth, ChunkSize - GhostWidth). Those form an exact partition of the volume's cells
  /// across all chunks, so concatenating the results neither duplicates nor drops a triangle.
  class CudaChunkMeshExtractor {
  public:

    /// Number of cells along one axis of a chunk that are actually extracted.
    static constexpr std::uint32_t CellDim = Chunkifier::CoreSize;

    /// Number of grid corners along one axis that those cells touch. One more than CellDim,
    /// because a cell spans one voxel further than its own base corner.
    static constexpr std::uint32_t CornerDim = CellDim + 1;

    /// Upper bounds on what a single chunk can emit. Every grid edge carries at most one vertex,
    /// and every cell at most 5 triangles.
    static constexpr std::uint32_t MaxVertices  = 3 * CornerDim * CornerDim * CornerDim;
    static constexpr std::uint32_t MaxTriangles = 5 * CellDim * CellDim * CellDim;

    explicit CudaChunkMeshExtractor(float isoThreshold);
    ~CudaChunkMeshExtractor();

    CudaChunkMeshExtractor(CudaChunkMeshExtractor&&) noexcept;
    CudaChunkMeshExtractor& operator=(CudaChunkMeshExtractor&&) noexcept;
    CudaChunkMeshExtractor(CudaChunkMeshExtractor const&) = delete;
    CudaChunkMeshExtractor& operator=(CudaChunkMeshExtractor const&) = delete;

    /// Uploads the chunk, extracts its isosurface and reads the result back.
    ///
    /// "out" is cleared first and is left empty when the chunk carries no surface. That is a normal
    /// outcome rather than an error: the chunkifier culls on the value range of a chunk, which is a
    /// conservative test and can let a chunk through whose owned cells happen to contain no
    /// crossing at all.
    void Extract(Chunkifier::DataChunk const& chunk, ChunkMesh& out);

    /// Whether a usable CUDA device is present. Lets a test suite degrade to skipping the GPU
    /// cases on a machine without one, rather than failing to run at all.
    static bool IsCudaAvailable();

  private:
    struct Impl;
    std::unique_ptr<Impl> P;
  };

} // namespace parallel_mesh_extractor
