#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include "Chunk.h"

namespace parallel_mesh_extractor {

  /// Splits a dense volume into the overlapping chunks consumed by the surface extraction.
  ///
  /// Every chunk is allocated as a ChunkSize^3 array, of which only the inner CoreSize^3 region is
  /// owned by the chunk. The remaining GhostWidth-voxel shell replicates the data of the
  /// neighboring chunks, which is what allows the extraction to evaluate the cells along a chunk
  /// boundary without any cross-chunk lookup, and therefore to emit geometry that connects
  /// seamlessly to the geometry of the neighbor.
  ///
  /// Consequently the chunks advance by CoreSize -- not by ChunkSize -- along every axis, and two
  /// adjacent chunks overlap by 2*GhostWidth voxels. Where the shell reaches outside of the volume
  /// (at the very first and very last chunk of every axis), the chunk is filled with a
  /// configurable background value instead.
  class Chunkifier {
  public:

    /// Total edge length of the allocated chunk. Deliberately kept a power of two so that the
    /// extraction kernels can index it with bit shifts rather than integer divisions.
    static constexpr std::uint32_t ChunkSize = 64;

    /// Thickness of the ghost shell replicated from the neighboring chunks on each of the 6 sides.
    /// A single layer is what the extraction needs to evaluate the boundary cells and the gradients
    /// on them.
    static constexpr std::uint32_t GhostWidth = 1;

    /// Edge length of the region a chunk owns, which is also the chunk-to-chunk stride.
    static constexpr std::uint32_t CoreSize = ChunkSize - 2 * GhostWidth;

    using DataChunk = Chunk<ChunkSize>;

    class ChunkIterator {
    public:
      ChunkIterator(float const* data = nullptr, float backgroundValue = 0.0f) noexcept;

      void ComputeChunkingOffsets(std::array<std::uint32_t, 3> const& dataDim);

      DataChunk operator*() const;

      ChunkIterator& operator++();

      std::int32_t GetChunkIndex(void) const;

    private:
      std::int32_t ChunkIndex = -1;
      float const* Data = nullptr;
      float BackgroundValue = 0.0f;
      std::array<std::uint32_t, 3> DataDims;
      std::vector<std::array<std::uint32_t, 3>> CoreOrigins;
    };

    Chunkifier(float const* data, std::array<std::uint32_t, 3> const& dim,
               float backgroundValue = 0.0f) noexcept;

    ChunkIterator begin() const;

    ChunkIterator end() const;

  private:
    float const* Data;
    std::array<std::uint32_t, 3> Dimensions;
    float BackgroundValue;
  };

} // namespace parallel_mesh_extractor


bool operator!=(parallel_mesh_extractor::Chunkifier::ChunkIterator const& lhs,
                parallel_mesh_extractor::Chunkifier::ChunkIterator const& rhs);
