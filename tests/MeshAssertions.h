#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <vector>

#include "../MeshExtraction/ChunkMesh.h"

namespace mesh_assertions {

  using Vertex = std::array<float, 3>;

  /// An undirected edge of a triangle mesh, as an ordered index pair.
  using Edge = std::pair<std::uint32_t, std::uint32_t>;

  inline Edge undirected(std::uint32_t a, std::uint32_t b) {
    return (a < b) ? Edge{a, b} : Edge{b, a};
  }

  inline Vertex subtract(Vertex const& a, Vertex const& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  }

  inline Vertex cross(Vertex const& a, Vertex const& b) {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
  }

  inline double dot(Vertex const& a, Vertex const& b) {
    return static_cast<double>(a[0]) * b[0]
         + static_cast<double>(a[1]) * b[1]
         + static_cast<double>(a[2]) * b[2];
  }

  inline double length(Vertex const& a) { return std::sqrt(dot(a, a)); }

  /// Twice the area vector of a triangle, which doubles as its unnormalized normal.
  inline Vertex triangleNormal(parallel_mesh_extractor::ChunkMesh const& mesh, std::size_t triangle) {
    auto const& v0 = mesh.Positions[mesh.Indices[3 * triangle + 0]];
    auto const& v1 = mesh.Positions[mesh.Indices[3 * triangle + 1]];
    auto const& v2 = mesh.Positions[mesh.Indices[3 * triangle + 2]];
    return cross(subtract(v1, v0), subtract(v2, v0));
  }

  inline double surfaceArea(parallel_mesh_extractor::ChunkMesh const& mesh) {
    double area = 0.0;
    for (std::size_t triangle = 0; triangle < mesh.TriangleCount(); ++triangle) {
      area += 0.5 * length(triangleNormal(mesh, triangle));
    }
    return area;
  }

  /// Volume enclosed by a closed mesh, by the divergence theorem. Positive when the triangles are
  /// wound so that their normals point outwards, negative when they point inwards -- which makes
  /// this a check of the winding convention as much as of the geometry.
  inline double enclosedVolume(parallel_mesh_extractor::ChunkMesh const& mesh) {
    double volume = 0.0;
    for (std::size_t triangle = 0; triangle < mesh.TriangleCount(); ++triangle) {
      auto const& v0 = mesh.Positions[mesh.Indices[3 * triangle + 0]];
      auto const& v1 = mesh.Positions[mesh.Indices[3 * triangle + 1]];
      auto const& v2 = mesh.Positions[mesh.Indices[3 * triangle + 2]];
      volume += dot(v0, cross(v1, v2)) / 6.0;
    }
    return volume;
  }

  /// How often each directed triangle edge occurs. A mesh is a closed, consistently oriented
  /// surface exactly when every directed edge occurs once and its reverse occurs once.
  inline std::map<Edge, int> directedEdgeCounts(parallel_mesh_extractor::ChunkMesh const& mesh) {
    std::map<Edge, int> counts;
    for (std::size_t triangle = 0; triangle < mesh.TriangleCount(); ++triangle) {
      auto const a = mesh.Indices[3 * triangle + 0];
      auto const b = mesh.Indices[3 * triangle + 1];
      auto const c = mesh.Indices[3 * triangle + 2];
      ++counts[Edge{a, b}];
      ++counts[Edge{b, c}];
      ++counts[Edge{c, a}];
    }
    return counts;
  }

  /// Watertightness and manifoldness are separate properties, and Marching Cubes separates them in
  /// practice: with the classic Lorensen tables the surface never has a hole, but it may pinch
  /// itself, leaving an edge shared by four triangles rather than two. Such a mesh is still a
  /// closed boundary -- it encloses a well defined volume and nothing leaks through it -- it just
  /// is not a manifold. Conflating the two would either forbid a legitimate, documented outcome or
  /// let an actual hole pass, so the report keeps them apart.
  struct ManifoldReport {
    /// No boundary: every edge is shared by an even number of triangles, at least two.
    bool IsWatertight = true;

    /// Every edge is shared by exactly two triangles.
    bool IsManifold = true;

    /// Every directed edge occurs as often as its reverse, i.e. the triangles around each edge
    /// pair up into opposing traversals.
    bool IsConsistentlyOriented = true;

    std::size_t BoundaryEdges = 0;     ///< shared by fewer than two, or by an odd number
    std::size_t NonManifoldEdges = 0;  ///< shared by more than two
    std::size_t UniqueEdges = 0;
  };

  inline ManifoldReport analyzeManifold(parallel_mesh_extractor::ChunkMesh const& mesh) {
    auto const directed = directedEdgeCounts(mesh);

    std::map<Edge, int> undirectedCounts;
    for (auto const& [edge, count] : directed) {
      undirectedCounts[undirected(edge.first, edge.second)] += count;
    }

    ManifoldReport report;
    report.UniqueEdges = undirectedCounts.size();

    for (auto const& [edge, count] : undirectedCounts) {
      (void)edge;
      if (count < 2 || count % 2 != 0) { ++report.BoundaryEdges;    report.IsWatertight = false; }
      if (count > 2)                   { ++report.NonManifoldEdges; report.IsManifold   = false; }
    }
    report.IsManifold = report.IsManifold && report.IsWatertight;

    for (auto const& [edge, count] : directed) {
      // The triangles meeting at an edge must traverse it in opposing directions in equal numbers;
      // an imbalance means their normals disagree.
      auto const reverse = directed.find(Edge{edge.second, edge.first});
      int const reverseCount = (reverse == directed.end()) ? 0 : reverse->second;
      if (count != reverseCount) report.IsConsistentlyOriented = false;
    }

    return report;
  }

  /// V - E + F. For a closed surface of genus 0 this is 2, which pins the connectivity far more
  /// tightly than counting triangles does.
  inline long eulerCharacteristic(parallel_mesh_extractor::ChunkMesh const& mesh) {
    std::set<std::uint32_t> usedVertices;
    std::set<Edge> edges;

    for (std::size_t triangle = 0; triangle < mesh.TriangleCount(); ++triangle) {
      auto const a = mesh.Indices[3 * triangle + 0];
      auto const b = mesh.Indices[3 * triangle + 1];
      auto const c = mesh.Indices[3 * triangle + 2];
      usedVertices.insert(a); usedVertices.insert(b); usedVertices.insert(c);
      edges.insert(undirected(a, b));
      edges.insert(undirected(b, c));
      edges.insert(undirected(c, a));
    }

    return static_cast<long>(usedVertices.size())
         - static_cast<long>(edges.size())
         + static_cast<long>(mesh.TriangleCount());
  }

  /// Key identifying a vertex position by its exact bit pattern.
  ///
  /// Welding on exact equality rather than within a tolerance is deliberate. Two chunks that share
  /// a grid edge read the same two voxel values and evaluate the same interpolation, so their
  /// vertices agree to the last bit. If they ever stop agreeing, welding must fail loudly instead
  /// of papering over it with an epsilon.
  using PositionKey = std::array<std::uint32_t, 3>;

  inline PositionKey positionKey(Vertex const& v) {
    PositionKey key{};
    std::memcpy(key.data(), v.data(), sizeof(key));
    return key;
  }

  /// Concatenate a mesh onto another, shifting its indices.
  inline void append(parallel_mesh_extractor::ChunkMesh& target,
                     parallel_mesh_extractor::ChunkMesh const& source) {
    auto const offset = static_cast<std::uint32_t>(target.Positions.size());
    target.Positions.insert(target.Positions.end(), source.Positions.begin(), source.Positions.end());
    for (auto const index : source.Indices) target.Indices.push_back(index + offset);
  }

  /// Merge vertices that occupy exactly the same position, producing the mesh as it would look if
  /// the chunk boundaries had never existed.
  inline parallel_mesh_extractor::ChunkMesh weldByExactPosition(
      parallel_mesh_extractor::ChunkMesh const& mesh) {
    parallel_mesh_extractor::ChunkMesh welded;
    std::map<PositionKey, std::uint32_t> lookup;
    std::vector<std::uint32_t> remap(mesh.Positions.size());

    for (std::size_t i = 0; i < mesh.Positions.size(); ++i) {
      auto const key = positionKey(mesh.Positions[i]);
      auto const existing = lookup.find(key);
      if (existing != lookup.end()) {
        remap[i] = existing->second;
        continue;
      }
      auto const fresh = static_cast<std::uint32_t>(welded.Positions.size());
      lookup.emplace(key, fresh);
      welded.Positions.push_back(mesh.Positions[i]);
      remap[i] = fresh;
    }

    welded.Indices.reserve(mesh.Indices.size());
    for (auto const index : mesh.Indices) welded.Indices.push_back(remap[index]);

    return welded;
  }

  /// Drop triangles that collapsed to a point or a line. The extraction emits them on purpose --
  /// culling them would tear the index structure -- but they carry no surface, and a check on the
  /// *geometry* of a mesh has to look past them.
  inline parallel_mesh_extractor::ChunkMesh withoutDegenerateTriangles(
      parallel_mesh_extractor::ChunkMesh const& mesh) {
    parallel_mesh_extractor::ChunkMesh result;
    result.Positions = mesh.Positions;

    for (std::size_t triangle = 0; triangle < mesh.TriangleCount(); ++triangle) {
      auto const a = mesh.Indices[3 * triangle + 0];
      auto const b = mesh.Indices[3 * triangle + 1];
      auto const c = mesh.Indices[3 * triangle + 2];
      if (a == b || b == c || c == a) continue;
      result.Indices.push_back(a);
      result.Indices.push_back(b);
      result.Indices.push_back(c);
    }
    return result;
  }

  /// A canonical form of a triangle soup, so that two extractions can be compared without caring
  /// about the order in which the triangles came out or which corner each one starts at.
  using CanonicalTriangle = std::array<PositionKey, 3>;

  inline std::vector<CanonicalTriangle> canonicalTriangles(
      parallel_mesh_extractor::ChunkMesh const& mesh) {
    std::vector<CanonicalTriangle> triangles;
    triangles.reserve(mesh.TriangleCount());

    for (std::size_t triangle = 0; triangle < mesh.TriangleCount(); ++triangle) {
      CanonicalTriangle corners = {
        positionKey(mesh.Positions[mesh.Indices[3 * triangle + 0]]),
        positionKey(mesh.Positions[mesh.Indices[3 * triangle + 1]]),
        positionKey(mesh.Positions[mesh.Indices[3 * triangle + 2]]),
      };
      // Rotate to the smallest corner first. A rotation preserves the winding, so two triangles
      // that differ only in which corner they start at compare equal, while a flipped one does not.
      auto const smallest = static_cast<std::size_t>(
        std::min_element(corners.begin(), corners.end()) - corners.begin());
      CanonicalTriangle rotated = {corners[smallest],
                                   corners[(smallest + 1) % 3],
                                   corners[(smallest + 2) % 3]};
      triangles.push_back(rotated);
    }

    std::sort(triangles.begin(), triangles.end());
    return triangles;
  }

} // namespace mesh_assertions
