#include "MeshAssembler.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace pme = parallel_mesh_extractor;

namespace {

  /// Reduce a position to the bit patterns of its coordinates.
  ///
  /// The two encodings of zero compare equal as floats but differ bit for bit, so a vertex landing
  /// on -0.0 in one chunk and on +0.0 in another would otherwise fail to merge. Normalising them
  /// costs one comparison per coordinate and removes the whole question.
  std::array<std::uint32_t, 3> keyOf(std::array<float, 3> const& position) {
    std::array<std::uint32_t, 3> key{};
    for (int axis = 0; axis < 3; ++axis) {
      float value = position[axis];
      if (value == 0.0f) value = 0.0f;
      std::memcpy(&key[axis], &value, sizeof(float));
    }
    return key;
  }

  float coordinateOf(std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(float));
    return value;
  }

} // namespace

std::size_t pme::MeshAssembler::PositionKeyHash::operator()(PositionKey const& key) const noexcept {
  // FNV-1a over the three words. The inputs are float bit patterns, whose low bits vary far more
  // than their high ones, so a hash that merely concatenated them would cluster badly.
  std::size_t hash = 1469598103934665603ull;
  for (auto const word : key) {
    hash ^= static_cast<std::size_t>(word);
    hash *= 1099511628211ull;
  }
  return hash;
}

void pme::MeshAssembler::Add(ChunkMesh const& chunkMesh,
                             std::vector<std::array<float, 3>>& newVertices,
                             std::vector<std::uint32_t>& faceIndices) {
  newVertices.clear();
  faceIndices.clear();

  if (chunkMesh.Positions.empty()) return;

  this->Remap.assign(chunkMesh.Positions.size(), 0);

  for (std::size_t local = 0; local < chunkMesh.Positions.size(); ++local) {
    auto const key = keyOf(chunkMesh.Positions[local]);

    auto const existing = this->Lookup.find(key);
    if (existing != this->Lookup.end()) {
      this->Remap[local] = existing->second;
      continue;
    }

    // PLY addresses vertices with 32 bit indices, so a mesh beyond that is not representable and
    // has to be reported rather than silently wrapped around.
    if (this->EmittedVertices > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("mesh exceeds the 32 bit vertex indexing the output format uses");
    }

    auto const global = static_cast<std::uint32_t>(this->EmittedVertices++);
    this->Lookup.emplace(key, global);
    this->Remap[local] = global;
    newVertices.push_back(chunkMesh.Positions[local]);
  }

  faceIndices.reserve(chunkMesh.Indices.size());
  for (auto const local : chunkMesh.Indices) faceIndices.push_back(this->Remap[local]);

  this->EmittedTriangles += chunkMesh.TriangleCount();
}

void pme::MeshAssembler::PruneBelowZ(float voxelZ) {
  for (auto it = this->Lookup.begin(); it != this->Lookup.end(); ) {
    if (coordinateOf(it->first[2]) < voxelZ) {
      it = this->Lookup.erase(it);
    } else {
      ++it;
    }
  }
}
