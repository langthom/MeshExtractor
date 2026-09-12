#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "ChunkMesh.h"

namespace parallel_mesh_extractor {

  /// Joins the meshes of individually extracted chunks into one mesh, merging the vertices they
  /// have in common, and does so without ever holding the finished mesh.
  ///
  /// The extraction emits a vertex per grid edge *per chunk*, so a vertex on the boundary between
  /// two chunks arrives once from each of them. Merging those is what turns a pile of chunk meshes
  /// into a surface whose connectivity runs across the whole volume.
  ///
  /// Rather than accumulating, each Add hands back only the vertices it had not seen before,
  /// together with the chunk's triangles already renumbered onto the global indexing. A caller can
  /// write both straight out and keep nothing, which is what makes a volume larger than memory
  /// possible.
  ///
  /// Vertices are matched on the exact bit pattern of their position, which is the right test
  /// rather than a tolerant one: two chunks sharing a grid edge read the same two voxel values and
  /// evaluate the same interpolation, so their vertices agree to the last bit. An epsilon would
  /// paper over a regression instead of reporting it. Positions must therefore be fed in *voxel*
  /// coordinates, exactly as the extraction produces them; scaling to millimetres belongs after
  /// the welding, not before it.
  ///
  /// Where the data puts a corner value exactly on the isovalue, several grid edges interpolate to
  /// the same point, and merging by position then collapses them into one vertex -- turning the
  /// triangles between them into degenerate ones. That is the honest consequence of welding by
  /// position and matches what the extraction already does with such data.
  class MeshAssembler {
  public:

    /// Merges one chunk mesh into the assembly.
    ///
    /// "newVertices" and "faceIndices" are cleared first and then filled with, respectively, the
    /// positions not seen before and the chunk's triangles as three global indices each. The
    /// global index of a vertex is its position in the concatenation of every "newVertices" handed
    /// out so far, so a caller writing them out in order needs no further bookkeeping.
    void Add(ChunkMesh const& chunkMesh,
             std::vector<std::array<float, 3>>& newVertices,
             std::vector<std::uint32_t>& faceIndices);

    /// Forgets cached vertices lying below a voxel z coordinate.
    ///
    /// Two chunks can only share a vertex if their tiles touch, and two chunk layers touch in
    /// exactly one plane of grid corners. Once a layer is finished, nothing below that plane can
    /// be referenced again, so dropping it keeps the cache proportional to a slice rather than to
    /// the volume. The comparison is strict, because a vertex sitting exactly on the plane is
    /// precisely the one the next layer will want.
    void PruneBelowZ(float voxelZ);

    /// Vertices and triangles handed out so far, i.e. the counts the finished mesh will carry.
    std::uint64_t VertexCount() const { return this->EmittedVertices; }
    std::uint64_t TriangleCount() const { return this->EmittedTriangles; }

    /// How many vertices the merge cache is currently holding. Of no use to a caller, but it is
    /// the only way to observe that pruning does anything.
    std::size_t CachedVertexCount() const { return this->Lookup.size(); }

  private:
    /// A position reduced to the bit patterns of its three coordinates.
    using PositionKey = std::array<std::uint32_t, 3>;

    struct PositionKeyHash {
      std::size_t operator()(PositionKey const& key) const noexcept;
    };

    std::unordered_map<PositionKey, std::uint32_t, PositionKeyHash> Lookup;

    std::uint64_t EmittedVertices = 0;
    std::uint64_t EmittedTriangles = 0;

    /// Scratch for the chunk local to global index mapping, kept across calls so that merging a
    /// few thousand chunks does not allocate a few thousand times.
    std::vector<std::uint32_t> Remap;
  };

} // namespace parallel_mesh_extractor
