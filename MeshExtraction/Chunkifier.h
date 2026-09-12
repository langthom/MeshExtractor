#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include "Chunk.h"
#include "SlabSchedule.h"

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

      /// Dimensions of the *whole* volume, not of the slab held in "Data". The chunk grid is laid
      /// out over the volume, so that a chunk's core origin means the same thing no matter which
      /// slab produced it.
      std::array<std::uint32_t, 3> Dimensions;

      /// The slab in hand and the chunk layers it is responsible for. Every read of "Data"
      /// subtracts Slab.ZBegin from the global z to reach the slice actually held.
      SlabWindow Slab;

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

    /// Chunks a volume held in memory in its entirety.
    Chunkifier(float const* data, std::array<std::uint32_t, 3> const& dim,
               float isoThreshold = 0.0f, float backgroundValue = 0.0f) noexcept;

    /// Chunks one slab of a volume that is too large to hold at once.
    ///
    /// "slabData" holds the slices [slab.ZBegin, slab.ZBegin + slab.ZCount) of a volume whose
    /// overall dimensions are "dim", laid out exactly as the corresponding part of the full volume
    /// would be. Exactly the chunk layers named by "slab" are handed out.
    ///
    /// Which layers a slab is *responsible* for is the caller's decision rather than something
    /// inferred from the slices in hand, and deliberately so: a short final layer needs so few
    /// slices that more than one slab can be capable of producing it, and a chunkifier guessing
    /// from capability alone would emit it twice. PlanSlabs assigns every layer to one slab, and
    /// CoversLayer below states what a slab has to hold to honour that assignment.
    Chunkifier(float const* slabData, std::array<std::uint32_t, 3> const& dim,
               SlabWindow const& slab,
               float isoThreshold = 0.0f, float backgroundValue = 0.0f) noexcept;

    void ComputeChunking(std::array<std::uint32_t, 3> const& dataDim);

    /// Whether a slab holds every slice the given chunk layer needs. This is a statement about
    /// capability, not about ownership: several slabs may be able to produce the same layer, which
    /// is why the layer assignment is passed in rather than derived from this.
    static bool CoversLayer(std::int64_t tileZ, std::array<std::uint32_t, 3> const& dim,
                            std::int64_t slabZBegin, std::int64_t slabZEnd);

    ChunkIterator begin() const;

    ChunkIterator end() const;

  private:
    ChunkingDataCollection ChunkingData;
  };

} // namespace parallel_mesh_extractor


bool operator!=(parallel_mesh_extractor::Chunkifier::ChunkIterator const& lhs,
                parallel_mesh_extractor::Chunkifier::ChunkIterator const& rhs);
