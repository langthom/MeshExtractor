#pragma once

#include "SliceChunkedVolumeDataIO.h"

namespace parallel_mesh_extractor {

  class SliceChunkedMHDIO : public SliceChunkedVolumeDataIO<SliceChunkedMHDIO> {
  public:

    void ReadMetaDataImpl();

    BufferType ReadSlicesImpl(unsigned int begin, unsigned int end) const;

  private:
    std::int64_t offset;
  };

} // namespace parallel_mesh_extractor

