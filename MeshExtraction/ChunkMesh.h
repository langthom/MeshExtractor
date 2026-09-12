#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace parallel_mesh_extractor {

  /// The isosurface of a single chunk, as an indexed triangle mesh.
  ///
  /// Positions are in *global voxel coordinates*, i.e. in the coordinate system of the whole
  /// volume rather than of the chunk, so that the meshes of two chunks can simply be concatenated.
  /// Converting to millimetres by scaling with the voxel spacing is left to whoever writes the mesh
  /// out: keeping the extraction unit free is what makes the vertex of a shared grid edge come out
  /// bit identical in both chunks that see it, which in turn is what makes the assembled mesh
  /// watertight.
  ///
  /// A vertex is emitted once per grid edge it sits on, no matter how many of the up to four cells
  /// meeting at that edge reference it. Within a chunk the mesh is therefore properly indexed;
  /// across chunks, vertices on the shared boundary still appear once per chunk and have to be
  /// welded by the caller.
  struct ChunkMesh {
    std::vector<std::array<float, 3>> Positions;

    /// Three indices into "Positions" per triangle.
    std::vector<std::uint32_t> Indices;

    std::size_t TriangleCount() const { return this->Indices.size() / 3; }

    void Clear() {
      this->Positions.clear();
      this->Indices.clear();
    }

    bool IsEmpty() const { return this->Indices.empty(); }
  };

} // namespace parallel_mesh_extractor
