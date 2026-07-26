
#include <filesystem>
#include <numeric>
#include <fstream>
#include <sstream>
#include "doctest.h"
#include "../VolumeIO/SliceChunkedMHDIO.h"

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
      parallel_mesh_extractor::VolumeMetaData metaData = parallel_mesh_extractor::VolumeMetaData{},
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
      if (writeMask & 0x08) metaStr << "Offset          = " << offset << '\n';
      if (writeMask & 0x10) metaStr << "ElementSpacing  = " << metaData.spacing[0] << ' ' << metaData.spacing[1] << ' ' << metaData.spacing[2] << '\n';
      if (writeMask & 0x20) metaStr << "DimSize         = " << metaData.dim[0]     << ' ' << metaData.dim[1]     << ' ' << metaData.dim[2]     << '\n';
      if (writeMask & 0x40) metaStr << "ElementType     = " << et << '\n';
      if (writeMask & 0x80) metaStr << "ElementDataFile = " << this->volPath << '\n';
      if (auxiliaryLine != "")  metaStr << auxiliaryLine << '\n';

      std::ofstream mhdFile{this->mhdPath};
      mhdFile << metaStr.str();
      
      if (writeMask & 0x80) {
        auto buf = std::make_unique<char[]>(payloadSize * metaData.GetElementSizeInBytes());
        parallel_mesh_extractor::DISPATCH_VOXEL_TYPE(parallel_mesh_extractor::ALL_VOXEL_TYPES,
                                                     metaData.voxelType, fillBuffer,
                                                     buf.get(), payloadSize);
      
        std::ofstream volFile{this->volPath, std::ios::binary};
        volFile.write(buf.get(), payloadSize * metaData.GetElementSizeInBytes());
      }
    }

    ~ConstructTemporaryMHDFile() {
      std::filesystem::remove(this->mhdPath);
      std::filesystem::remove(this->volPath);
    }

    std::filesystem::path GetMHDPath(void) const {
      return this->mhdPath;
    }
  };

} // anonymous namespace

TEST_CASE("MHD meta data parsing fails") {
  std::string mhd = "test";
  std::string vol = "test";
  parallel_mesh_extractor::VolumeMetaData vmd;

  // Case 1: The requested MHD file does not exist.
  parallel_mesh_extractor::SliceChunkedMHDIO scmhd;
  scmhd.SetFilePath("blablubb.mhd");
  CHECK_THROWS_AS(scmhd.ReadMetaData(), std::filesystem::filesystem_error const&);

  // Case 1.5: Malformed file.
  {
    auto tmpMHD = ConstructTemporaryMHDFile(0xFF, mhd, vol, vmd, 1, 0, "Image", 3, "blubb");
    auto mhdIO  = parallel_mesh_extractor::SliceChunkedMHDIO();
    mhdIO.SetFilePath(tmpMHD.GetMHDPath());
    CHECK_THROWS_AS(mhdIO.ReadMetaData(), std::runtime_error const&);
  }

  auto missingMetaDataElementTest = [](char mask) {
    auto tmpMHD = ConstructTemporaryMHDFile(mask);
    auto mhdIO  = parallel_mesh_extractor::SliceChunkedMHDIO();
    mhdIO.SetFilePath(tmpMHD.GetMHDPath());
    CHECK_THROWS_AS(mhdIO.ReadMetaData(), std::runtime_error const&);
  };

  // Case 2: No ObjectType is given.
  missingMetaDataElementTest(0xfe);
 
  // Case 3: ObjectType is given but is not image.
  
  // Case 4: NDims is not given.
  missingMetaDataElementTest(0xfd);

  // Case 5: NDims is given but is not equal to 3.

  // Case 6: No ElementSpacing is given.
  missingMetaDataElementTest(0xef);

  // Case 7: ElementSpacing is given but spacing is invalid (negative or zero on any axis).
  
  // Case 8: No DimSize is given.
  missingMetaDataElementTest(0xdf);

  // Case 9: DimSize is given but is invalid (zero on any axis).
  
  // Case 10: No ElementType is given.
  missingMetaDataElementTest(0xbf);

  // Case 11: ElementType is given but is invalid.
  
  // Case 12: No ElementDataFile is given.
  missingMetaDataElementTest(0x7f);

  // Case 13: ElementDataFile is given but does not exist.
  
  // Case 14: The size of the ElementDataFile does not correspond to the product of DimSize + Offset

}

TEST_CASE("MHD meta data parsing succeeds") {
  CHECK(false);
}

TEST_CASE("MHD volume reading fails") {
  CHECK(false);
}

TEST_CASE("MHD volume reading succeeds") {
  CHECK(false);
}

