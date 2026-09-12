#pragma once

#include <array>
#include <filesystem>
#include <memory>

#include "VolumeMetaData.h"

namespace parallel_mesh_extractor {

  template<class DerivedIO>
  class SliceChunkedVolumeDataIO {
  public:

    using BufferType = std::unique_ptr<float[]>;

    void SetFilePath(std::filesystem::path const& volumeDataFilePath) {
      this->volumeDataFilePath = volumeDataFilePath;
    }

    VolumeMetaData const& GetMetaData() const {
      return this->metaData;
    }

    VolumeMetaData const& ReadMetaData() {
      static_cast<DerivedIO*>(this)->ReadMetaDataImpl();
      return this->metaData;
    }

    BufferType ReadSlices(unsigned int begin, unsigned int numberOfSlices) const {
      return static_cast<DerivedIO const*>(this)->ReadSlicesImpl(begin, numberOfSlices);
    }

    /// Reads slices directly into a buffer the caller already owns.
    ///
    /// Same conversion as ReadSlices -- every voxel is widened to float one element at a time, not
    /// reinterpreted -- but written straight to its destination. ReadSlices has to allocate a
    /// buffer to hand back, which a caller assembling a larger slab then copies out of and throws
    /// away; over a whole volume that is two extra passes across several gigabytes.
    ///
    /// "destination" must have room for numberOfSlices * width * height floats. Returns false when
    /// the request is out of range or the read fails, leaving the destination untouched.
    bool ReadSlicesInto(unsigned int begin, unsigned int numberOfSlices, float* destination) {
      return static_cast<DerivedIO*>(this)->ReadSlicesIntoImpl(begin, numberOfSlices, destination);
    }

  protected:
    SliceChunkedVolumeDataIO() = default;

  protected:
    std::filesystem::path volumeDataFilePath;
    VolumeMetaData metaData;
  };

} // namespace parallel_mesh_extractor

