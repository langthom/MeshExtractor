
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
      int headerSize                                   = 0,
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
      // "Offset" is the image origin, a triple in the unit of the spacing. The number of bytes to
      // skip before the voxel data is the separate, optional "HeaderSize", which is only written
      // when it is actually non-zero -- so the default case also exercises its absence.
      if (writeMask & 0x04) metaStr << "Offset          = " << metaData.origin[0] << ' ' << metaData.origin[1] << ' ' << metaData.origin[2] << '\n';
      if (headerSize != 0)  metaStr << "HeaderSize      = " << headerSize << '\n';
      if (writeMask & 0x08) metaStr << "ElementSpacing  = " << metaData.spacing[0] << ' ' << metaData.spacing[1] << ' ' << metaData.spacing[2] << '\n';
      if (writeMask & 0x10) metaStr << "DimSize         = " << metaData.dim[0]     << ' ' << metaData.dim[1]     << ' ' << metaData.dim[2]     << '\n';
      if (writeMask & 0x20) metaStr << "ElementType     = " << et << '\n';
      if (writeMask & 0x40) metaStr << "ElementDataFile = " << this->volPath << '\n';
      if (auxiliaryLine != "")  metaStr << auxiliaryLine << '\n';

      std::ofstream mhdFile{this->mhdPath};
      mhdFile << metaStr.str();
      
      if (writeMask & 0x80) {
        auto buf = std::make_unique<char[]>(payloadSize * metaData.GetElementSizeInBytes() + headerSize);
        parallel_mesh_extractor::DISPATCH_VOXEL_TYPE(parallel_mesh_extractor::ALL_VOXEL_TYPES,
                                                     metaData.voxelType, fillBuffer,
                                                     buf.get() + headerSize, payloadSize);
      
        std::ofstream volFile{this->volPath, std::ios::binary};
        volFile.write(buf.get(), payloadSize * metaData.GetElementSizeInBytes() + headerSize);
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
  // Each mask below clears exactly one of the required keys. The labels used to be shifted by one
  // bit against the mask they annotate; since every missing required key throws the same way, the
  // cases still passed while naming the wrong field.
  // -- No ObjectType is given.      (0x01)
  testFile(0x7E);
  // -- No NDims given.              (0x02)
  testFile(0x7D);
  // -- No Offset given.             (0x04)
  testFile(0x7B);
  // -- No ElementSpacing given.     (0x08)
  testFile(0x77);
  // -- No DimSize given.            (0x10)
  testFile(0x6F);
  // -- No ElementType given.        (0x20)
  testFile(0x5F);
  // -- No ElementDataFile given.    (0x40)
  testFile(0x3F);

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
  CHECK_EQ(parsed_vmd.origin,    vmd.origin);
}

TEST_CASE("A relative ElementDataFile is resolved next to the MHD file") {
  // MHD files in the wild name their raw file as a bare filename sitting beside them. Resolving
  // that against the working directory of whoever is running would make a perfectly ordinary pair
  // of files readable only from one particular directory.
  parallel_mesh_extractor::VolumeMetaData vmd;
  vmd.dim       = {2, 3, 4};
  vmd.spacing   = {1, 1, 1};
  vmd.voxelType = parallel_mesh_extractor::VoxelType::UINT16;

  auto const directory = std::filesystem::temp_directory_path() / "pme_relative_mhd";
  std::filesystem::create_directories(directory);

  auto const mhdPath = directory / "scan.mhd";
  auto const rawPath = directory / "scan.raw";

  auto const size = vmd.GetNumberOfVoxels();
  {
    auto buffer = std::make_unique<std::uint16_t[]>(size);
    fillBuffer<std::uint16_t>(reinterpret_cast<char*>(buffer.get()), size);
    std::ofstream raw(rawPath, std::ios::binary);
    raw.write(reinterpret_cast<char const*>(buffer.get()),
              static_cast<std::streamsize>(size * sizeof(std::uint16_t)));
  }
  {
    std::ofstream mhd(mhdPath);
    mhd << "ObjectType      = Image\n"
        << "NDims           = 3\n"
        << "Offset          = 0 0 0\n"
        << "ElementSpacing  = 1 1 1\n"
        << "DimSize         = 2 3 4\n"
        << "ElementType     = MET_USHORT\n"
        << "ElementDataFile = scan.raw\n";   // beside the MHD, named without any directory
  }

  auto mhdIO = parallel_mesh_extractor::SliceChunkedMHDIO();
  mhdIO.SetFilePath(mhdPath);
  REQUIRE_NOTHROW(mhdIO.ReadMetaData());

  auto const read = mhdIO.ReadSlices(0, vmd.dim[2]);
  REQUIRE(read != nullptr);

  auto expected = std::make_unique<std::uint16_t[]>(size);
  fillBuffer<std::uint16_t>(reinterpret_cast<char*>(expected.get()), size);
  for (std::uintmax_t i = 0; i < size; ++i) CHECK_EQ(expected[i], read[i]);

  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
}

TEST_CASE("The MHD offset is read as the image origin") {
  // "Offset" names where the first voxel sits in world coordinates, in the same unit as the
  // spacing, and everything extracted from the volume is translated by it. It is a triple, and it
  // is routinely negative on scans whose reconstruction volume is centred on the rotation axis.
  parallel_mesh_extractor::VolumeMetaData vmd;
  vmd.dim       = {2, 3, 4};
  vmd.spacing   = {0.25f, 0.25f, 0.5f};
  vmd.voxelType = parallel_mesh_extractor::VoxelType::UINT16;

  auto check = [&vmd](std::array<float, 3> const& origin) {
    vmd.origin = origin;
    auto mhd = ConstructTemporaryMHDFile(0xFF, "test", "test", vmd, 2 * 3 * 4, 0, "Image", 3, "");
    auto mhdIO = parallel_mesh_extractor::SliceChunkedMHDIO();
    mhdIO.SetFilePath(mhd.GetMHDPath());

    REQUIRE_NOTHROW(mhdIO.ReadMetaData());
    CHECK_EQ(mhdIO.GetMetaData().origin, origin);
  };

  check({0.0f, 0.0f, 0.0f});
  check({1.5f, 2.25f, 3.125f});
  check({-10.0f, -0.5f, 0.0f});
  check({-100.5f, -100.5f, -200.25f});
}

TEST_CASE("An MHD offset that is not a triple is rejected") {
  // The value list is what has to be long enough, not the destination array. Guarding the
  // destination instead would make this line read past the end of the parsed values rather than
  // report a malformed file.
  parallel_mesh_extractor::VolumeMetaData vmd = ConstructTemporaryMHDFile::DefaultMetaData();

  for (auto const& truncated : {"Offset = 1", "Offset = 1 2", "ElementSpacing = 1", "DimSize = 1 1"}) {
    // The auxiliary line is appended after the well formed one, and the later assignment wins.
    auto mhd = ConstructTemporaryMHDFile(0xFF, "test", "test", vmd, 1, 0, "Image", 3, truncated);
    auto mhdIO = parallel_mesh_extractor::SliceChunkedMHDIO();
    mhdIO.SetFilePath(mhd.GetMHDPath());

    INFO("offending line: ", truncated);
    CHECK_THROWS_AS(mhdIO.ReadMetaData(), std::runtime_error const&);
  }
}

TEST_CASE("The MHD header size skips a header in the raw file and defaults to zero") {
  // "HeaderSize" is the number of bytes of the raw file that are not voxels. It is optional, and a
  // file that does not name it simply has no header -- which is what every other case in this
  // suite relies on.
  parallel_mesh_extractor::VolumeMetaData vmd;
  vmd.dim       = {2, 3, 4};
  vmd.spacing   = {1, 1, 1};
  vmd.voxelType = parallel_mesh_extractor::VoxelType::UINT16;

  auto const size = vmd.GetNumberOfVoxels();
  auto expected = std::make_unique<std::uint16_t[]>(size);
  fillBuffer<std::uint16_t>(reinterpret_cast<char*>(expected.get()), size);

  SUBCASE("absent means no header") {
    auto mhd = ConstructTemporaryMHDFile(0xFF, "test", "test", vmd, size, 0, "Image", 3, "");
    auto mhdIO = parallel_mesh_extractor::SliceChunkedMHDIO();
    mhdIO.SetFilePath(mhd.GetMHDPath());

    REQUIRE_NOTHROW(mhdIO.ReadMetaData());
    auto const read = mhdIO.ReadSlices(0, vmd.dim[2]);
    REQUIRE(read != nullptr);
    for (std::uintmax_t i = 0; i < size; ++i) CHECK_EQ(expected[i], read[i]);
  }

  SUBCASE("a header is skipped, leaving the voxels unchanged") {
    auto mhd = ConstructTemporaryMHDFile(0xFF, "test", "test", vmd, size, 2048, "Image", 3, "");
    auto mhdIO = parallel_mesh_extractor::SliceChunkedMHDIO();
    mhdIO.SetFilePath(mhd.GetMHDPath());

    REQUIRE_NOTHROW(mhdIO.ReadMetaData());
    auto const read = mhdIO.ReadSlices(0, vmd.dim[2]);
    REQUIRE(read != nullptr);
    for (std::uintmax_t i = 0; i < size; ++i) CHECK_EQ(expected[i], read[i]);
  }

  SUBCASE("a negative header size is rejected") {
    auto mhd = ConstructTemporaryMHDFile(0xFF, "test", "test", vmd, size, 0, "Image", 3, "HeaderSize = -8");
    auto mhdIO = parallel_mesh_extractor::SliceChunkedMHDIO();
    mhdIO.SetFilePath(mhd.GetMHDPath());
    CHECK_THROWS_AS(mhdIO.ReadMetaData(), std::runtime_error const&);
  }
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

