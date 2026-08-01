
#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string_view>
#include "SliceChunkedMHDIO.h"

#include <iostream>

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
    if (values.size() < Size) {
      throw std::runtime_error("Number of expected values not fulfilled.");
    }
    for (int i = 0; i < Size; ++i) {
      impl::parseNumeric<T, EnforceGTZero>(svs[i], values[i]);
    }
    maskValue |= Mask;
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
      impl::checkValues<0x04>(values.front(), this->offset, requiredEntriesGiven);
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
      this->volumeDataFilePath = values.front();
      std::cout << "path: " << this->volumeDataFilePath << "\n";
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
    this->metaData.GetNumberOfVoxels() * this->metaData.GetElementSizeInBytes() + this->offset;

  if (dataSize != expectedSize) {
    throw std::runtime_error("Meta data does not match the size of the data file.");
  }
}
 

Buffer parallel_mesh_extractor::SliceChunkedMHDIO::ReadSlicesImpl(unsigned int begin, unsigned int end) const {
  // If 
  return nullptr;
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



