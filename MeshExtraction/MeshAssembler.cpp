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

  /// Whether two positions weld into one vertex, which is the same test the merge cache applies.
  /// Asking it of a triangle's corners decides whether that triangle survives the weld, without
  /// having to enter any of them into the cache first.
  bool weldsTogether(std::array<float, 3> const& a, std::array<float, 3> const& b) {
    return keyOf(a) == keyOf(b);
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

  if (chunkMesh.Indices.empty()) return;

  // Nothing is entered into the cache before a surviving triangle asks for it, so a vertex that
  // only ever appears in collapsed triangles is never emitted and leaves no orphan behind.
  this->Remap.assign(chunkMesh.Positions.size(), Unassigned);

  auto const globalIndexOf = [&](std::uint32_t local) -> std::uint32_t {
    if (this->Remap[local] != Unassigned) return this->Remap[local];

    auto const key = keyOf(chunkMesh.Positions[local]);

    auto const existing = this->Lookup.find(key);
    if (existing != this->Lookup.end()) {
      this->Remap[local] = existing->second;
      return existing->second;
    }

    // PLY addresses vertices with 32 bit indices, so a mesh beyond that is not representable and
    // has to be reported rather than silently wrapped around. The last index is spent on the
    // "not yet assigned" marker above, which costs one vertex out of four billion.
    if (this->EmittedVertices >= std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("mesh exceeds the 32 bit vertex indexing the output format uses");
    }

    auto const global = static_cast<std::uint32_t>(this->EmittedVertices++);
    this->Lookup.emplace(key, global);
    this->Remap[local] = global;
    newVertices.push_back(chunkMesh.Positions[local]);
    return global;
  };

  faceIndices.reserve(chunkMesh.Indices.size());

  for (std::size_t base = 0; base + 3 <= chunkMesh.Indices.size(); base += 3) {
    auto const& v0 = chunkMesh.Positions[chunkMesh.Indices[base + 0]];
    auto const& v1 = chunkMesh.Positions[chunkMesh.Indices[base + 1]];
    auto const& v2 = chunkMesh.Positions[chunkMesh.Indices[base + 2]];

    // Two corners welding into one leave a triangle that is a line or a point. It carries no
    // surface and no orientation, and every consumer downstream -- normals, decimation, the
    // manifold checks -- has to special case it, so it is dropped here instead.
    //
    // Dropping it does not open the surface. The triangle's two non-self edges are the same edge
    // traversed both ways, so they cancel against each other and the directed edge balance that
    // makes the mesh closed is left exactly as it was. What remains is a vertex the surface
    // pinches itself at, which Marching Cubes produces anyway.
    if (weldsTogether(v0, v1) || weldsTogether(v1, v2) || weldsTogether(v2, v0)) {
      ++this->DroppedTriangles;
      continue;
    }

    faceIndices.push_back(globalIndexOf(chunkMesh.Indices[base + 0]));
    faceIndices.push_back(globalIndexOf(chunkMesh.Indices[base + 1]));
    faceIndices.push_back(globalIndexOf(chunkMesh.Indices[base + 2]));
    ++this->EmittedTriangles;
  }
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
