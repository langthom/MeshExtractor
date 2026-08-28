#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include "Chunk.h"

namespace parallel_mesh_extractor {

  class Chunkifier {
  public:

    static constexpr std::uint32_t ChunkSize = 64;

    using DataChunk = Chunk<ChunkSize>;

    class ChunkIterator {
    public:
      ChunkIterator(float const* data = nullptr) noexcept;

      void ComputeChunkingOffsets(std::array<std::uint32_t, 3> const& dataDim);

      DataChunk operator*() const;

      ChunkIterator& operator++();

      std::int32_t GetChunkIndex(void) const;
    
    private:
      std::int32_t ChunkIndex = -1;
      float const* Data = nullptr;
      std::array<std::uint32_t, 3> DataDims;
      std::vector<std::array<std::uint32_t, 3>> ChunkOrigins;
    };

    Chunkifier(float const* data, std::array<std::uint32_t, 3> const& dim) noexcept;

    ChunkIterator begin() const;

    ChunkIterator end() const;

  private:
    float const* Data;
    std::array<std::uint32_t, 3> Dimensions;
  };

} // namespace parallel_mesh_extractor


bool operator!=(parallel_mesh_extractor::Chunkifier::ChunkIterator const& lhs,
                parallel_mesh_extractor::Chunkifier::ChunkIterator const& rhs);






