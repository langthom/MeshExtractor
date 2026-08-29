#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace parallel_mesh_extractor {

  /// A cubic block of voxel data of edge length "Size", ready to be handed to the surface
  /// extraction kernels.
  ///
  /// The block is not a plain partition of the volume: its outermost voxel layer is a ghost shell
  /// replicating the data of the neighboring blocks, so that the extraction can evaluate the cells
  /// along the block boundary without ever reading outside of this allocation. The region the
  /// block actually owns is therefore the inner part of "data", starting at "CoreOrigin" in global
  /// voxel coordinates.
  template<std::uint32_t Size=64>
  struct Chunk {
    static constexpr std::int32_t const ChunkDimSize = static_cast<std::int32_t>(Size);

    /// Global voxel coordinate of the first owned voxel, i.e., of the local index
    /// (GhostWidth, GhostWidth, GhostWidth). The ghost shell below it samples the global
    /// coordinate CoreOrigin-1, which is outside of the volume for the very first chunks.
    std::array<std::int64_t, 3> CoreOrigin{0, 0, 0};

    float data[Size][Size][Size];

    /// Set every voxel of the chunk to a single scalar value.
    void Fill(float value) {
      std::fill_n(&this->data[0][0][0], static_cast<std::size_t>(Size) * Size * Size, value);
    }
  };

} // namespace parallel_mesh_extractor
