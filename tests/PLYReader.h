#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "doctest.h"

namespace ply_reader {

  struct ParsedPLY {
    std::string Format;
    std::uint64_t DeclaredVertices = 0;
    std::uint64_t DeclaredFaces = 0;
    std::size_t HeaderBytes = 0;
    std::vector<std::array<float, 3>> Positions;
    std::vector<std::uint32_t> Indices;
  };

  /// A small PLY reader written independently of the writer, so that agreeing with it says
  /// something. It understands only the one layout the writer emits, and complains otherwise.
  inline ParsedPLY parsePLY(std::filesystem::path const& path) {
    std::ifstream file(path, std::ios::binary);
    REQUIRE(file.good());

    std::string const content{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

    auto const headerEnd = content.find("end_header\n");
    REQUIRE(headerEnd != std::string::npos);

    ParsedPLY parsed;
    parsed.HeaderBytes = headerEnd + std::strlen("end_header\n");

    std::istringstream header(content.substr(0, headerEnd));
    std::string line;
    std::string element;

    while (std::getline(header, line)) {
      std::istringstream words(line);
      std::string keyword;
      words >> keyword;

      if (keyword == "format") {
        words >> parsed.Format;
      } else if (keyword == "element") {
        words >> element;
        if (element == "vertex") words >> parsed.DeclaredVertices;
        else if (element == "face") words >> parsed.DeclaredFaces;
      }
    }

    if (parsed.Format == "binary_little_endian") {
      std::size_t offset = parsed.HeaderBytes;

      parsed.Positions.resize(parsed.DeclaredVertices);
      auto const vertexBytes = parsed.DeclaredVertices * 3 * sizeof(float);
      REQUIRE(content.size() >= offset + vertexBytes);
      if (vertexBytes > 0) std::memcpy(parsed.Positions.data(), content.data() + offset, vertexBytes);
      offset += vertexBytes;

      for (std::uint64_t i = 0; i < parsed.DeclaredFaces; ++i) {
        REQUIRE(content.size() >= offset + 1 + 3 * sizeof(std::uint32_t));
        auto const corners = static_cast<std::uint8_t>(content[offset]);
        REQUIRE(corners == 3);
        offset += 1;

        std::uint32_t triangle[3];
        std::memcpy(triangle, content.data() + offset, 3 * sizeof(std::uint32_t));
        offset += 3 * sizeof(std::uint32_t);

        parsed.Indices.insert(parsed.Indices.end(), triangle, triangle + 3);
      }

      // Nothing may trail the declared elements.
      CHECK(offset == content.size());
    } else {
      std::istringstream body(content.substr(parsed.HeaderBytes));
      for (std::uint64_t i = 0; i < parsed.DeclaredVertices; ++i) {
        std::array<float, 3> vertex{};
        body >> vertex[0] >> vertex[1] >> vertex[2];
        REQUIRE(body.good());
        parsed.Positions.push_back(vertex);
      }
      for (std::uint64_t i = 0; i < parsed.DeclaredFaces; ++i) {
        int corners = 0;
        std::uint32_t triangle[3];
        body >> corners >> triangle[0] >> triangle[1] >> triangle[2];
        REQUIRE(corners == 3);
        parsed.Indices.insert(parsed.Indices.end(), triangle, triangle + 3);
      }
    }

    return parsed;
  }

} // namespace ply_reader
