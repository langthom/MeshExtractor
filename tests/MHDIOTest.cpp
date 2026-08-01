
#include <filesystem>
#include <numeric>
#include <fstream>
#include <sstream>
#include "doctest.h"
#include "../VolumeIO/SliceChunkedMHDIO.h"

#include <iostream>

namespace {
  template<class VoxelType> std::string voxelTypeString();
  template<> std::string voxelTypeString<std::int8_t  >() { return "MET_CHAR";   }
  template<> std::string voxelTypeString<std::int16_t >() { return "MET_SHORT";  }
  template<> std::string voxelTypeString<std::int32_t >() { return "MET_INT";    }
  template<> std::string voxelTypeString<std::uint8_t >() { return "MET_UCHAR";  }
  template<> std::string voxelTypeString<std::uint16_t>() { return "MET_USHORT"; }
  template<> std::string voxelTypeString<std::uint32_t>() { return "MET_UINT";   }
  template<> std::string voxelTypeString<float        >() { return "MET_FLOAT";  }
  template<> std::string voxelTypeString<double       >() { return "MET_DOUBLE"; }

  template<class VoxelType>
  void fillBuffer(char* buffer, size_t bufferElements) {
    auto b = reinterpret_cast<VoxelType*>(buffer);
    std::iota(b, b+bufferElements, VoxelType(0));
  }

  namespace pme = parallel_mesh_extractor;

  class ConstructTemporaryMHDFile {
    using VT = pme::VoxelType;
    std::filesystem::path mhdPath, volPath;

  public:

    ConstructTemporaryMHDFile(
      char writeMask                                   = 0xFF,
      std::string mhdName                              = "test",
      std::string volName                              = "test",
      parallel_mesh_extractor::VolumeMetaData metaData = ConstructTemporaryMHDFile::DefaultMetaData(),
      size_t payloadSize                               = 1,
      int offset                                       = 0,
      std::string objectType                           = "Image",
      int ndims                                        = 3,
      std::string auxiliaryLine                        = ""
    ) {
      auto tempFilePath = std::filesystem::temp_directory_path();
      this->mhdPath = tempFilePath / (mhdName + ".mhd");
      this->volPath = tempFilePath / (volName + ".vol");

      using SupportedVoxelTypes = pme::VoxelTypeSelection<
        VT::INT8, VT::INT16, VT::INT32, VT::UINT8, VT::UINT16, VT::UINT32, VT::FLOAT32, VT::FLOAT64
      >;
      auto et = pme::DISPATCH_VOXEL_TYPE(SupportedVoxelTypes, metaData.voxelType, voxelTypeString);

      std::ostringstream metaStr;
      if (writeMask & 0x01) metaStr << "ObjectType      = " << objectType << '\n';
      if (writeMask & 0x02) metaStr << "NDims           = " << ndims      << '\n';
      if (writeMask & 0x04) metaStr << "Offset          = " << offset << '\n';
      if (writeMask & 0x08) metaStr << "ElementSpacing  = " << metaData.spacing[0] << ' ' << metaData.spacing[1] << ' ' << metaData.spacing[2] << '\n';
      if (writeMask & 0x10) metaStr << "DimSize         = " << metaData.dim[0]     << ' ' << metaData.dim[1]     << ' ' << metaData.dim[2]     << '\n';
      if (writeMask & 0x20) metaStr << "ElementType     = " << et << '\n';
      if (writeMask & 0x40) metaStr << "ElementDataFile = " << this->volPath << '\n';
      if (auxiliaryLine != "")  metaStr << auxiliaryLine << '\n';

      std::ofstream mhdFile{this->mhdPath};
      mhdFile << metaStr.str();
      
      if (writeMask & 0x80) {
        auto buf = std::make_unique<char[]>(payloadSize * metaData.GetElementSizeInBytes() + offset);
        parallel_mesh_extractor::DISPATCH_VOXEL_TYPE(parallel_mesh_extractor::ALL_VOXEL_TYPES,
                                                     metaData.voxelType, fillBuffer,
                                                     buf.get() + offset, payloadSize);
      
        std::ofstream volFile{this->volPath, std::ios::binary};
        volFile.write(buf.get(), payloadSize * metaData.GetElementSizeInBytes() + offset);
      }
    }

    ~ConstructTemporaryMHDFile() {
      std::filesystem::remove(this->mhdPath);
      std::filesystem::remove(this->volPath);
    }

    std::filesystem::path GetMHDPath(void) const {
      return this->mhdPath;
    }

    static parallel_mesh_extractor::VolumeMetaData DefaultMetaData(void) {
      parallel_mesh_extractor::VolumeMetaData vmd;
      vmd.dim       = {1, 1, 1};
      vmd.spacing   = {1, 1, 1};
      vmd.voxelType = parallel_mesh_extractor::VoxelType::UINT16;
      return vmd;
    }
  };

} // anonymous namespace

TEST_CASE("MHD meta data parsing fails") {
  std::string mhd = "test";
  std::string vol = "test";
  parallel_mesh_extractor::VolumeMetaData vmd = ConstructTemporaryMHDFile::DefaultMetaData();

  // Case 1: The requested MHD file does not exist.
  parallel_mesh_extractor::SliceChunkedMHDIO scmhd;
  scmhd.SetFilePath("blablubb.mhd");
  CHECK_THROWS_AS(scmhd.ReadMetaData(), std::filesystem::filesystem_error const&);

  auto testFile = [&vmd](auto&&... args) {
    auto tmpMHD = ConstructTemporaryMHDFile(args...);
    auto mhdIO  = parallel_mesh_extractor::SliceChunkedMHDIO();
    mhdIO.SetFilePath(tmpMHD.GetMHDPath());
    CHECK_THROWS_AS(mhdIO.ReadMetaData(), std::runtime_error const&);
    vmd = parallel_mesh_extractor::VolumeMetaData{};
  };

  // Case 2: Malformed file.
  // -- Non Key=Value line.
  testFile(0x7F, mhd, vol, vmd, 1, 0, "Image", 3, "blubb");
  // -- No ObjectType is given.
  testFile(0x7E);
  // -- No NDims given.
  testFile(0x7D);
  // -- No ElementSpacing given.
  testFile(0x7B);
  // -- No DimSize given.
  testFile(0x77);
  // -- No ElementType given.
  testFile(0x6F);
  // -- No ElementDataFile given.
  testFile(0x5F);

  // Case 3: File form is good but elements are invalid.
  // -- ObjectType is not "Image".
  testFile(0x7F, mhd, vol, vmd, 1, 0, "Invalid", 3, "");
  // -- NDims is not equal to 3.
  testFile(0x7F, mhd, vol, vmd, 1, 0, "Image", 2, "");
  // -- ElementSpacing is not positive.
  auto spacings = std::vector<std::array<float, 3>>{
    {-1, -1, -1}, {0, -1, -1}, {-1, 0, -1}, {-1, -1, 0}, {0, 0, -1}, {0, -1, 0}, {-1, 0, 0},
    {0, 1, 1}, {1, 0, 1}, {1, 1, 0}
  };
  for (auto spacing : spacings) {
    vmd.spacing = spacing;
    testFile(0x7F, mhd, vol, vmd, 1, 0, "Image", 3, "");
  }
  // -- ElementType is invalid.
  vmd.voxelType = static_cast<parallel_mesh_extractor::VoxelType>(0);
  testFile(0x7F, mhd, vol, vmd, 1, 0, "Image", 3, "");
  
  // Case 4: Regarding the linked data file.
  // -- ElementDataFile does not exist.
  testFile(0x7F, mhd, "non existing file", vmd, 1, 0, "Image", 3, "");
  // -- The size of the ElementDataFile does not match the meta data.
  testFile(0xFF, mhd, vol, vmd, 17, 0, "Image", 3, "");
}

TEST_CASE("MHD meta data parsing succeeds") {
  parallel_mesh_extractor::VolumeMetaData vmd;
  vmd.dim       = {2, 3, 4};
  vmd.spacing   = {0.01, 0.01, 0.01};
  vmd.voxelType = parallel_mesh_extractor::VoxelType::UINT16;

  auto mhd = ConstructTemporaryMHDFile(0xFF, "test", "test", vmd, 2*3*4, 0, "Image", 3, "");
  auto mhdIO = parallel_mesh_extractor::SliceChunkedMHDIO();
  mhdIO.SetFilePath(mhd.GetMHDPath());

  // Parsing should succeed and not throw any exception.
  CHECK_NOTHROW(mhdIO.ReadMetaData());

  // The read fields should also be equal.
  auto parsed_vmd = mhdIO.GetMetaData();
  CHECK_EQ(parsed_vmd.dim,       vmd.dim);
  CHECK_EQ(parsed_vmd.spacing,   vmd.spacing);
  CHECK_EQ(parsed_vmd.voxelType, vmd.voxelType);
}

TEST_CASE("MHD volume reading fails") {
  parallel_mesh_extractor::VolumeMetaData vmd;
  vmd.dim       = {2, 3, 4};
  vmd.spacing   = {0.01, 0.01, 0.01};
  vmd.voxelType = parallel_mesh_extractor::VoxelType::UINT16;

  std::vector<std::pair<unsigned int, unsigned int>> invalidSliceSpecs{
    {4, 0}, {4, 1}, {5, 0}, {5, 1},
    {0, 0}, {1, 0}, {2, 0}, {3, 0},
    {0, 5}, {0, 6}, {1, 4}, {1, 5},
    {2, 3}, {2, 4}, {3, 2}, {3, 3},
  };

  for (int offset : {0, 2048}) {
    auto size = vmd.GetNumberOfVoxels();
    auto expectedBuffer = std::make_unique<std::uint16_t[]>(size);
    fillBuffer<std::uint16_t>(reinterpret_cast<char*>(expectedBuffer.get()), size);

    auto mhd = ConstructTemporaryMHDFile(0xFF, "test", "test", vmd, size, offset, "Image", 3, "");
    auto mhdIO = parallel_mesh_extractor::SliceChunkedMHDIO();
    mhdIO.SetFilePath(mhd.GetMHDPath());
    CHECK_NOTHROW(mhdIO.ReadMetaData());

    for (auto const [beg, num] : invalidSliceSpecs) {
      CHECK_EQ(nullptr, mhdIO.ReadSlices(beg, num));
    }
  }
}

TEST_CASE("MHD volume reading succeeds") {
  parallel_mesh_extractor::VolumeMetaData vmd;
  vmd.dim       = {2, 3, 4};
  vmd.spacing   = {0.01, 0.01, 0.01};
  vmd.voxelType = parallel_mesh_extractor::VoxelType::UINT16;

  // Case 1: Create and read a small entire volume (i.e., all slices in primary dimension) at a given offset.
  for (int offset : {0, 2048}) {
    auto size = vmd.GetNumberOfVoxels();
    auto expectedBuffer = std::make_unique<std::uint16_t[]>(size);
    fillBuffer<std::uint16_t>(reinterpret_cast<char*>(expectedBuffer.get()), size);

    auto mhd = ConstructTemporaryMHDFile(0xFF, "test", "test", vmd, size, offset, "Image", 3, "");
    auto mhdIO = parallel_mesh_extractor::SliceChunkedMHDIO();
    mhdIO.SetFilePath(mhd.GetMHDPath());
    CHECK_NOTHROW(mhdIO.ReadMetaData());

    auto numberOfSlices = mhdIO.GetMetaData().dim[2];
    auto readBuffer = mhdIO.ReadSlices(0, numberOfSlices);
    CHECK(readBuffer != nullptr);

    for (int i = 0; i < size; ++i) {
      CHECK_EQ(expectedBuffer[i], readBuffer[i]);
    }
  }

  // Case 2: Read only a single slice.
  auto sliceSize      = vmd.dim[0] * vmd.dim[1];
  auto size           = vmd.GetNumberOfVoxels();
  auto expectedBuffer = std::make_unique<std::uint16_t[]>(size);
  fillBuffer<std::uint16_t>(reinterpret_cast<char*>(expectedBuffer.get()), size);

  auto mhd = ConstructTemporaryMHDFile(0xFF, "test", "test", vmd, size, 0, "Image", 3, "");
  auto mhdIO = parallel_mesh_extractor::SliceChunkedMHDIO();
  mhdIO.SetFilePath(mhd.GetMHDPath());
  CHECK_NOTHROW(mhdIO.ReadMetaData());

  for (unsigned int start : {0, 1, 2, 3}) {
    auto readBuffer = mhdIO.ReadSlices(start, 1);
    CHECK(readBuffer != nullptr);
    auto expectedSlice = expectedBuffer.get() + start * sliceSize;

    for (int i = 0; i < sliceSize; ++i) {
      CHECK_EQ(expectedSlice[i], readBuffer[i]);
    }
  }
}

