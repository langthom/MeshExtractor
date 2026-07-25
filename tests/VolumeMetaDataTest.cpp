
#include <string>
#include "doctest.h"
#include "../VolumeIO/VolumeMetaData.h"

namespace {
  template<class VoxelType> std::string voxelTypeString();
  template<> std::string voxelTypeString<std::int8_t  >() { return "std::int8_t";   }
  template<> std::string voxelTypeString<std::int16_t >() { return "std::int16_t";  }
  template<> std::string voxelTypeString<std::int32_t >() { return "std::int32_t";  }
  template<> std::string voxelTypeString<std::int64_t >() { return "std::int64_t";  }
  template<> std::string voxelTypeString<std::uint8_t >() { return "std::uint8_t";  }
  template<> std::string voxelTypeString<std::uint16_t>() { return "std::uint16_t"; }
  template<> std::string voxelTypeString<std::uint32_t>() { return "std::uint32_t"; }
  template<> std::string voxelTypeString<std::uint64_t>() { return "std::uint64_t"; }
  template<> std::string voxelTypeString<float        >() { return "std::float_t";  }
  template<> std::string voxelTypeString<double       >() { return "std::double_t"; }
}

namespace pme = parallel_mesh_extractor;

TEST_CASE("Dispatch a function over the voxel type") {
  // For this test, we provided the templated function above which returns a string
  // describing the voxel type. This function lacks a specialization for an unrecognized
  // voxel type, for which we expect the dispatcher to return Nothing.
  using VT = pme::VoxelType;
  std::array<VT, 10> const voxelTypes{
    VT::INT8,  VT::INT16,  VT::INT32,  VT::INT64,  VT::FLOAT32,
    VT::UINT8, VT::UINT16, VT::UINT32, VT::UINT64, VT::FLOAT64,
  };
  std::array<std::string, 10> const expectedResults{
    "std::int8_t",  "std::int16_t",  "std::int32_t",  "std::int64_t",  "std::float_t",
    "std::uint8_t", "std::uint16_t", "std::uint32_t", "std::uint64_t", "std::double_t",
  };

  // Check the known and supported voxel types.
  for (int i = 0; i < 10; ++i) {
    auto const computedVoxelType = pme::DISPATCH_VOXEL_TYPE(pme::ALL_VOXEL_TYPES, voxelTypes[i], voxelTypeString);
    CHECK(expectedResults[i] == computedVoxelType);
  }

  // Check invalid voxel types.
  for (int invalid : {-1, 0, 1}) {
    CHECK("" == pme::DISPATCH_VOXEL_TYPE(pme::ALL_VOXEL_TYPES, static_cast<VT>(invalid), voxelTypeString));
  }
}

TEST_CASE("Get the element size of the voxel type") {
  using VT = pme::VoxelType;
  std::array<VT, 10> const voxelTypes{
    VT::INT8,  VT::INT16,  VT::INT32,  VT::INT64,  VT::FLOAT32,
    VT::UINT8, VT::UINT16, VT::UINT32, VT::UINT64, VT::FLOAT64,
  };
  std::array<size_t, 10> elementSizes{
    sizeof(std::int8_t),  sizeof(std::int16_t),  sizeof(std::int32_t),  sizeof(std::int64_t),  sizeof(float),
    sizeof(std::uint8_t), sizeof(std::uint16_t), sizeof(std::uint32_t), sizeof(std::uint64_t), sizeof(double),
  };

  // Check the supported voxel types.
  for (int i = 0; i < 10; ++i) {
    pme::VolumeMetaData vmd;
    vmd.voxelType = voxelTypes[i];
    CHECK(elementSizes[i] == vmd.GetElementSizeInBytes());
  }

  // Check invalid voxel types.
  for (int invalid : {-1, 0, 1}) {
    pme::VolumeMetaData vmd;
    vmd.voxelType = static_cast<VT>(invalid);
    CHECK(0ull == vmd.GetElementSizeInBytes());
  }
}

TEST_CASE("Compute the offsets for jumping to different columns, rows, or slices") {
  auto vec = [](std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return std::array<std::uint32_t, 3>{x, y, z};
  };
  auto off = [](std::size_t x, std::size_t y, std::size_t z) {
    return std::array<std::size_t, 3>{x, y, z};
  };

  std::vector<std::array<std::uint32_t, 3>> dimensionsVec{
    // Invalid volume, all dimensions are zero.
    vec(0, 0, 0),
    // Invalid volume, two volume dimensions are zero.
    vec(0, 0, 1), vec(0, 1, 0), vec(1, 0, 0),
    // Invalid volume, one volume dimension is zero.
    vec(0, 1, 1), vec(1, 0, 1), vec(1, 1, 0),
    // Valid volume cases.
    vec(1, 1, 1), vec(2, 3, 4), vec(4, 3, 2)
  };
  
  std::vector<std::array<std::size_t, 3>> expectedOffsetsVec{
    // Invalid volume, all dimensions are zero.
    off(1, 0, 0),
    // Invalid volume, two volume dimensions are zero.
    off(1, 0, 0), off(1, 0, 0), off(1, 1, 0),
    // Invalid volume, one volume dimension is zero.
    off(1, 0, 0), off(1, 1, 0), off(1, 1, 1),
    // Valid volume cases.
    off(1, 1, 1), off(1, 2, 6), off(1, 4, 12)
  };

  for (int i = 0; i < dimensionsVec.size(); ++i) {
    parallel_mesh_extractor::VolumeMetaData vmd;
    vmd.dim = dimensionsVec[i];

    auto const computedOffsets = vmd.ComputeOffsets();
    auto const expectedOffsets = expectedOffsetsVec[i];

    CHECK(expectedOffsets[0] == computedOffsets[0]);
    CHECK(expectedOffsets[1] == computedOffsets[1]);
    CHECK(expectedOffsets[2] == computedOffsets[2]);
  }
}


