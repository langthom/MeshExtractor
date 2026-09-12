#pragma once

#include <vector>

#include "SliceChunkedVolumeDataIO.h"

namespace parallel_mesh_extractor {

  class SliceChunkedMHDIO : public SliceChunkedVolumeDataIO<SliceChunkedMHDIO> {
  public:

    void ReadMetaDataImpl();

    BufferType ReadSlicesImpl(unsigned int begin, unsigned int end) const;

    bool ReadSlicesIntoImpl(unsigned int begin, unsigned int numberOfSlices, float* destination);

  private:
    /// Reused staging for the raw, still natively typed bytes. Held across calls so that reading a
    /// volume in slabs does not allocate per call.
    std::vector<char> rawStaging;

    /// Number of bytes to skip at the start of the raw data file before the voxels begin. Comes
    /// from the optional "HeaderSize" key and is zero when the file carries no header of its own.
    /// Not to be confused with "Offset", which is the image origin in world coordinates and lives
    /// in the meta data.
    std::int64_t headerSize = 0;
  };

} // namespace parallel_mesh_extractor

