
#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string_view>
#include "SliceChunkedMHDIO.h"

namespace impl {
  using KeyValues = std::pair<std::string_view, std::vector<std::string_view>>;
  std::optional<KeyValues> parseMHDLine(std::string_view line);

  template<class T, bool EnforceGTZero=true>
  void parseNumeric(std::string_view sv, T& value) {
    auto const [_, ec] = std::from_chars(sv.data(), sv.data()+sv.size(), value);
    if (ec != std::errc{} || (EnforceGTZero && (value < 0))) {
      throw std::runtime_error("Invalid numerical value encounterd");
    }
  }

  template<char Mask, class T, bool EnforceGTZero=true>
  void checkValues(std::string_view sv, T& value, char& maskValue) {
    impl::parseNumeric<T, EnforceGTZero>(sv, value);
    maskValue |= Mask;
  }

  template<char Mask, class T, size_t Size, bool EnforceGTZero=true>
  void checkValues(std::vector<std::string_view> const& svs, std::array<T, Size>& values, char& maskValue) {
    // Guard the *parsed* list, not the destination: the destination is a fixed size array, so
    // asking it for its size can never fail and would let a line carrying too few values read past
    // the end of "svs" below.
    if (svs.size() < Size) {
      throw std::runtime_error("Number of expected values not fulfilled.");
    }
    for (int i = 0; i < Size; ++i) {
      impl::parseNumeric<T, EnforceGTZero>(svs[i], values[i]);
    }
    maskValue |= Mask;
  }

  template<class VoxelType>
  void CastArrayToFloat(float* target, char const* source, size_t numElements) {
    VoxelType const* s = reinterpret_cast<VoxelType const*>(source);
    std::transform(s, s + numElements, target, [](VoxelType const x){ return static_cast<float>(x); });
  }
}

using Buffer = parallel_mesh_extractor::SliceChunkedMHDIO::BufferType;

void parallel_mesh_extractor::SliceChunkedMHDIO::ReadMetaDataImpl() {
  // Check file access.
  std::error_code fileError;
  auto const mhdStatus   = std::filesystem::status(this->volumeDataFilePath, fileError);
  auto const mhdFileType = mhdStatus.type();

  if (mhdFileType != std::filesystem::file_type::regular) {
    throw std::filesystem::filesystem_error("Requested MHD file is not a regular file.",
                                            this->volumeDataFilePath,
                                            fileError);
  }

  // Read the file into a key-value storage.
  // Whenever a non " key = value " line is encountered, this is consider a malformed file.
  std::map<std::string, std::string> metaKV;
  std::ifstream mhdFile{this->volumeDataFilePath};
  if (!mhdFile.is_open() || !mhdFile.good()) {
    fileError = std::io_errc::stream;
    throw std::filesystem::filesystem_error("Opening MHD file did not work.",
                                            this->volumeDataFilePath,
                                            fileError);
  }

  // Reset the optional fields, so that re-reading with this object cannot inherit a header size or
  // an origin from the file it parsed before.
  this->headerSize      = 0;
  this->metaData.origin = {0.0f, 0.0f, 0.0f};

  // Where the MHD file itself lives. An MHD normally names its raw file relative to itself, so
  // this is what such a name has to be resolved against -- noted before the parse loop, which
  // replaces the path with the data file's.
  auto const mhdDirectory = this->volumeDataFilePath.parent_path();

  char requiredEntriesGiven = 0x0;
  std::string line;
  while (std::getline(mhdFile, line)) {
    auto const keyValuePair = impl::parseMHDLine(line);
    if (!keyValuePair.has_value()) {
      throw std::runtime_error("Malformed MHD file.");
    }

    auto const [key, values] = keyValuePair.value();
    if (key == "ObjectType") {
      if (values.front() != "Image") {
        throw std::runtime_error("Unrecognized ObjectType");
      }
      requiredEntriesGiven |= 0x01;
    } else if (key == "NDims") {
      int ndims = 0;
      impl::checkValues<0x02>(values.front(), ndims, requiredEntriesGiven);
      if (ndims != 3) {
        throw std::runtime_error("Only three-dimensional images are permitted.");
      }
    } else if (key == "Offset") {
      // The image origin: the world coordinate of the first voxel, in the same unit as the
      // spacing. A scan whose reconstruction volume is centred on the rotation axis has a negative
      // origin on every axis, so the positivity check has to be off for this one.
      impl::checkValues<0x04, float, 3, false>(values, this->metaData.origin, requiredEntriesGiven);
    } else if (key == "HeaderSize") {
      // Optional: how many bytes of the raw file to skip before the voxels start. Absent means the
      // file is nothing but voxels.
      impl::parseNumeric<std::int64_t>(values.front(), this->headerSize);
    } else if (key == "ElementSpacing") {
      impl::checkValues<0x08>(values, this->metaData.spacing, requiredEntriesGiven);
    } else if (key == "DimSize") {
      impl::checkValues<0x10>(values, this->metaData.dim, requiredEntriesGiven);
    } else if (key == "ElementType") {
      std::map<std::string_view, VoxelType> const supportedElementTypes{
        {"MET_CHAR",  VoxelType::INT8},  {"MET_SHORT",  VoxelType::INT16},  {"MET_INT",  VoxelType::INT32},
        {"MET_UCHAR", VoxelType::UINT8}, {"MET_USHORT", VoxelType::UINT16}, {"MET_UINT", VoxelType::UINT32},
        {"MET_FLOAT", VoxelType::FLOAT32}, {"MET_DOUBLE", VoxelType::FLOAT64}
      };
      auto it = supportedElementTypes.find(values.front());
      if (it == supportedElementTypes.cend()) {
        throw std::runtime_error("Unrecognized element type.");
      }
      this->metaData.voxelType = it->second;
      requiredEntriesGiven |= 0x20;
    } else if (key == "ElementDataFile") {
      // Resolved against the directory holding the MHD, not against the working directory of
      // whoever is running. Almost every MHD in the wild names its raw file as a bare filename
      // sitting next to it, and resolving that against the process's current directory would make
      // reading such a pair depend on where the command happened to be issued from.
      std::filesystem::path dataFile = values.front();
      this->volumeDataFilePath = dataFile.is_absolute() ? dataFile : mhdDirectory / dataFile;
      requiredEntriesGiven |= 0x40;
    } else {
      // Additional potentially valid but ignored MHD key-value pair.
    }
  }

  if (requiredEntriesGiven != 0x7F) {
    throw std::runtime_error("Missing data field");
  }

  // Test the existance of the linked data file and check that its size maches the meta data.
  if (!std::filesystem::exists(this->volumeDataFilePath)) {
    throw std::runtime_error("Volume data file does not exists.");
  }
  
  std::uintmax_t const dataSize     = std::filesystem::file_size(this->volumeDataFilePath);
  std::uintmax_t const expectedSize = 
    this->metaData.GetNumberOfVoxels() * this->metaData.GetElementSizeInBytes() + this->headerSize;

  if (dataSize != expectedSize) {
    throw std::runtime_error("Meta data does not match the size of the data file.");
  }
}
 

Buffer parallel_mesh_extractor::SliceChunkedMHDIO::ReadSlicesImpl(unsigned int begin, unsigned int numberOfSlices) const {
  // In MHD, there may be metadata stored at the beginning of the file up to the specified offset
  // in bytes (which is zero by default). After that, the data is stored as contiguous block of data.

  if (begin                  >= this->metaData.dim[2]) { return nullptr; }
  if (begin + numberOfSlices >  this->metaData.dim[2]) { return nullptr; }
  if (numberOfSlices         >  this->metaData.dim[2]) { return nullptr; }
  if (numberOfSlices         ==                     0) { return nullptr; }

  auto const sliceSize      = this->metaData.dim[0] * this->metaData.dim[1];
  auto const elementSize    = this->metaData.GetElementSizeInBytes();
  auto const sliceSizeBytes = sliceSize * elementSize;


  std::ifstream mhdFile{this->volumeDataFilePath, std::ios::binary};
  if (!mhdFile.seekg(this->headerSize + begin * sliceSizeBytes, std::ios::beg)) {
    return nullptr;
  }

  auto rawBuffer = std::make_unique<char[]>(numberOfSlices * sliceSizeBytes);
  if (!mhdFile.read(rawBuffer.get(), numberOfSlices * sliceSizeBytes)) {
    return nullptr;
  }

  Buffer buffer{ new Buffer::element_type[numberOfSlices * sliceSize] };
  parallel_mesh_extractor::DISPATCH_VOXEL_TYPE(
    parallel_mesh_extractor::ALL_VOXEL_TYPES,
    this->metaData.voxelType,
    impl::CastArrayToFloat,
    buffer.get(), rawBuffer.get(), numberOfSlices * sliceSize
  );
  return buffer;
}

bool parallel_mesh_extractor::SliceChunkedMHDIO::ReadSlicesIntoImpl(unsigned int begin,
                                                                   unsigned int numberOfSlices,
                                                                   float* destination) {
  // Same range rules as ReadSlicesImpl; reporting them the same way keeps the two interchangeable.
  if (destination            ==               nullptr) { return false; }
  if (begin                  >= this->metaData.dim[2]) { return false; }
  if (begin + numberOfSlices >  this->metaData.dim[2]) { return false; }
  if (numberOfSlices         >  this->metaData.dim[2]) { return false; }
  if (numberOfSlices         ==                     0) { return false; }

  auto const sliceSize      = static_cast<std::size_t>(this->metaData.dim[0]) * this->metaData.dim[1];
  auto const elementSize    = this->metaData.GetElementSizeInBytes();
  auto const sliceSizeBytes = sliceSize * elementSize;
  auto const totalBytes     = static_cast<std::size_t>(numberOfSlices) * sliceSizeBytes;

  std::ifstream mhdFile{this->volumeDataFilePath, std::ios::binary};
  if (!mhdFile.seekg(this->headerSize + static_cast<std::int64_t>(begin) * sliceSizeBytes, std::ios::beg)) {
    return false;
  }

  this->rawStaging.resize(totalBytes);
  if (!mhdFile.read(this->rawStaging.data(), static_cast<std::streamsize>(totalBytes))) {
    return false;
  }

  // The voxels are still in their native type here. Widening them is a per element conversion --
  // the same one ReadSlicesImpl performs -- and emphatically not a reinterpretation of the bytes:
  // a uint16 of 1730 has to become the float 1730.0, not whatever those two bytes happen to spell
  // as half of a float. The only thing saved over ReadSlices is the buffer in between.
  parallel_mesh_extractor::DISPATCH_VOXEL_TYPE(
    parallel_mesh_extractor::ALL_VOXEL_TYPES,
    this->metaData.voxelType,
    impl::CastArrayToFloat,
    destination, this->rawStaging.data(), static_cast<std::size_t>(numberOfSlices) * sliceSize
  );
  return true;
}

// ---------------------------------------------------------------------------------------------- //

std::optional<impl::KeyValues> impl::parseMHDLine(std::string_view line) {
  auto trim = [](std::string_view str) -> std::string_view {
    while (!str.empty() && (std::isspace(str.front()) || str.front() == '\"')) {
      str.remove_prefix(1);
    }
    while (!str.empty() && (std::isspace(str.back()) || str.back() == '\"')) {
      str.remove_suffix(1);
    }
    return str;
  };

  // Find the = sign separating the key and its values.
  auto const eqPos = line.find('=');
  if (eqPos == std::string_view::npos) {
    return std::nullopt; // Malformed file if no = sign is present.
  }

  // Everything up to the = sign except spaces forms the key.
  auto key = trim(line.substr(0, eqPos));
  auto val = trim(line.substr(eqPos+1));

  if (key.empty() || val.empty()) {
    return std::nullopt;
  }

  // Also split up the value further by spaces.
  std::vector<std::string_view> values;
  while (!val.empty()) {
    std::size_t spacePos = 0;
    while (spacePos < val.size() && !std::isspace(val[spacePos])) {
      ++spacePos;
    }

    values.push_back(val.substr(0, spacePos));

    val.remove_prefix(spacePos);
    val = trim(val);
  }

  return std::make_pair(key, values);
}



