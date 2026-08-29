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

    struct ChunkingDataCollection {
      float const* Data;
      std::array<std::uint32_t, 3> Dimensions;
      std::vector<std::array<std::int64_t, 3>> CoreOrigins;
      std::vector<std::array<float, 2>> ValueRanges;
      float ISOThreshold;
      float BackgroundValue;
    };

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
      ChunkIterator(ChunkingDataCollection const* chunkingDataCollectionPtr) noexcept;

      DataChunk operator*() const;

      ChunkIterator& operator++();

      bool Equals(ChunkIterator const& other) const;

    private:
      /// Advance onto the next chunk whose value range contains the ISO threshold, so that the
      /// iterator always rests on a chunk that is going to be materialized.
      void SkipCulledChunks();

      /// Whether this iterator has run out of chunks, which is also true for the sentinel that
      /// end() hands out.
      bool AtEnd() const;

      ChunkingDataCollection const* ChunkingDataPtr;

      std::vector<std::array<std::int64_t, 3>>::const_iterator CoreOriginsIterator, CoreOriginsEnd;
      std::vector<std::array<float, 2>>::const_iterator ValueRangesIterator;
    };

    Chunkifier(float const* data, std::array<std::uint32_t, 3> const& dim,
               float isoThreshold = 0.0f, float backgroundValue = 0.0f) noexcept;

    void ComputeChunking(std::array<std::uint32_t, 3> const& dataDim);

    ChunkIterator begin() const;

    ChunkIterator end() const;

  private:
    ChunkingDataCollection ChunkingData;
  };

} // namespace parallel_mesh_extractor


bool operator!=(parallel_mesh_extractor::Chunkifier::ChunkIterator const& lhs,
                parallel_mesh_extractor::Chunkifier::ChunkIterator const& rhs);
