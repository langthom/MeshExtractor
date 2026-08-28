#pragma once

#include <cstring>

namespace parallel_mesh_extractor {

  template<std::uint32_t Size=64>
  struct Chunk {
    static constexpr std::int32_t const ChunkDimSize = static_cast<std::int32_t>(Size);

    float data[Size][Size][Size];

    Chunk() {
      std::memset(this->data, 0, Size*Size*Size*sizeof(float));
    }
  };

} // namespace parallel_mesh_extractor
