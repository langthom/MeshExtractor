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

  protected:
    SliceChunkedVolumeDataIO() = default;

  protected:
    std::filesystem::path volumeDataFilePath;
    VolumeMetaData metaData;
  };

} // namespace parallel_mesh_extractor

