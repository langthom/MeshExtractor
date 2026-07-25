#include "VolumeMetaData.h"

namespace pme = parallel_mesh_extractor;

namespace {
  template<class VoxelType> size_t sizeofCalculator() { return sizeof(VoxelType); }
}

std::array<std::size_t, 3> pme::VolumeMetaData::ComputeOffsets(void) const {
  return {
    1ull,
    static_cast<std::size_t>(this->dim[0]),
    static_cast<std::size_t>(this->dim[0]) * static_cast<std::size_t>(this->dim[1])
  };
}

size_t pme::VolumeMetaData::GetElementSizeInBytes(void) const {
  return pme::DISPATCH_VOXEL_TYPE(pme::ALL_VOXEL_TYPES, this->voxelType, sizeofCalculator);
}


