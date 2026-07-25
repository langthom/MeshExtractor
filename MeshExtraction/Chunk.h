#pragma once

namespace parallel_mesh_extractor {

  template<unsigned Size=64>
  struct Chunk {
    float data[Size][Size][Size];
  };

} // namespace parallel_mesh_extractor
