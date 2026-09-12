
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "doctest.h"
#include "../MeshExtraction/MarchingCubesTables.h"

namespace mc = parallel_mesh_extractor::marching_cubes;

namespace {

  /// A directed edge of the triangle mesh a case emits, given as the two cell edges its endpoints
  /// sit on. The whole table can be reasoned about purely combinatorially in these terms, without
  /// ever placing a vertex in space.
  using Segment = std::pair<int, int>;

  /// All directed triangle edges of a case, in emission order.
  std::vector<Segment> segmentsOf(std::uint32_t caseIndex) {
    std::vector<Segment> segments;
    auto const& row = mc::TriTable[caseIndex];

    for (int i = 0; i < 16 && row[i] >= 0; i += 3) {
      segments.emplace_back(row[i + 0], row[i + 1]);
      segments.emplace_back(row[i + 1], row[i + 2]);
      segments.emplace_back(row[i + 2], row[i + 0]);
    }
    return segments;
  }

  /// The 4 corners lying on each of the 6 cell faces, derived from the corner offsets so that the
  /// faces cannot drift out of sync with the corner numbering.
  std::array<std::vector<int>, 6> faceCorners() {
    std::array<std::vector<int>, 6> faces;

    for (int axis = 0; axis < 3; ++axis) {
      for (int side = 0; side < 2; ++side) {
        for (int corner = 0; corner < 8; ++corner) {
          if (mc::CornerOffset[corner][axis] == side) faces[2 * axis + side].push_back(corner);
        }
      }
    }
    return faces;
  }

  /// Face index in the order used by mc::FaceEdges, i.e. -x, +x, -y, +y, -z, +z.
  bool edgeIsOnFace(int edge, int face) {
    auto const& edges = mc::FaceEdges[face];
    return std::find(edges.begin(), edges.end(), edge) != edges.end();
  }

  std::string describe(std::vector<Segment> const& segments) {
    std::ostringstream out;
    for (auto const& [from, to] : segments) out << from << "->" << to << " ";
    return out.str();
  }

} // namespace

// ------------------------------------- table well-formedness --------------------------------- //

TEST_CASE("Every triangle table row is a well formed, terminated triangle list") {
  for (std::uint32_t caseIndex = 0; caseIndex < 256; ++caseIndex) {
    INFO("case ", caseIndex);
    auto const& row = mc::TriTable[caseIndex];

    // Find the terminator. Everything before it is payload, everything after it has to stay -1,
    // so that a kernel scanning for the terminator cannot walk into stale entries.
    int length = 0;
    while (length < 16 && row[length] >= 0) ++length;

    REQUIRE(length % 3 == 0);

    // At most 5 triangles fit into a cell, which is what leaves room for the terminator in a row
    // of 16. A 6th triangle would silently lose its terminator.
    CHECK(length <= 15);

    for (int i = 0; i < length; ++i) {
      INFO("entry ", i);
      CHECK(row[i] >= 0);
      CHECK(row[i] < 12);
    }
    for (int i = length; i < 16; ++i) {
      INFO("entry ", i);
      CHECK(row[i] == -1);
    }

    // No triangle may reference the same cell edge twice: that would be a degenerate triangle
    // baked into the table rather than one caused by the data.
    for (int i = 0; i < length; i += 3) {
      CHECK(row[i + 0] != row[i + 1]);
      CHECK(row[i + 1] != row[i + 2]);
      CHECK(row[i + 2] != row[i + 0]);
    }
  }
}

TEST_CASE("The empty cases emit nothing and every other case emits something") {
  // A cell whose corners all share a sign carries no surface, and those are the only two such
  // cases -- the extraction culls exactly on this property, so a stray entry here would make it
  // emit geometry for a homogeneous cell.
  CHECK(mc::TriangleCount(0) == 0);
  CHECK(mc::TriangleCount(255) == 0);

  for (std::uint32_t caseIndex = 1; caseIndex < 255; ++caseIndex) {
    INFO("case ", caseIndex);
    CHECK(mc::TriangleCount(caseIndex) > 0);
  }
}

TEST_CASE("The derived edge mask agrees with the triangle table") {
  // EdgeMask is what a host side reference implementation interpolates on. It is derived from
  // TriTable, so this pins the derivation rather than a second transcription.
  for (std::uint32_t caseIndex = 0; caseIndex < 256; ++caseIndex) {
    INFO("case ", caseIndex);

    std::uint16_t expected = 0;
    for (auto const& [from, to] : segmentsOf(caseIndex)) {
      expected |= static_cast<std::uint16_t>(1u << from);
      expected |= static_cast<std::uint16_t>(1u << to);
    }
    CHECK(mc::EdgeMask(caseIndex) == expected);
  }
}

TEST_CASE("Every cell edge a case uses actually separates two corners of opposite sign") {
  // A vertex may only be placed on an edge whose two corners disagree, since that is the only
  // place the isosurface can cross. An entry violating this would put a vertex on an edge the
  // extraction never even classifies as active, and the index lookup would read a stale slot.
  for (std::uint32_t caseIndex = 0; caseIndex < 256; ++caseIndex) {
    for (int edge = 0; edge < 12; ++edge) {
      if ((mc::EdgeMask(caseIndex) & (1u << edge)) == 0) continue;

      auto const [cornerA, cornerB] = mc::EdgeCorners[edge];
      bool const insideA = (caseIndex >> cornerA) & 1u;
      bool const insideB = (caseIndex >> cornerB) & 1u;

      INFO("case ", caseIndex, ", edge ", edge);
      CHECK(insideA != insideB);
    }
  }
}

TEST_CASE("The edge ownership table matches the edge endpoints") {
  // EdgeToOwner is what turns a cell edge into the shared grid edge carrying its vertex. It has to
  // name the edge's minimum corner and the axis it runs along, otherwise two cells sharing an edge
  // would look up different vertices and the mesh would fall apart along that edge.
  for (int edge = 0; edge < 12; ++edge) {
    INFO("edge ", edge);

    auto const [cornerA, cornerB] = mc::EdgeCorners[edge];
    auto const& offsetA = mc::CornerOffset[cornerA];
    auto const& offsetB = mc::CornerOffset[cornerB];
    auto const& owner   = mc::EdgeToOwner[edge];

    // The two corners must differ along the named axis and agree along the other two.
    for (int axis = 0; axis < 3; ++axis) {
      if (axis == owner.Axis) {
        CHECK(offsetA[axis] != offsetB[axis]);
      } else {
        CHECK(offsetA[axis] == offsetB[axis]);
      }
    }

    // The owner offset is the componentwise minimum of the two corner offsets.
    for (int axis = 0; axis < 3; ++axis) {
      CHECK(owner.Offset[axis] == std::min(offsetA[axis], offsetB[axis]));
    }
  }
}

TEST_CASE("The face edge table lists exactly the edges lying in each face") {
  auto const corners = faceCorners();

  for (int face = 0; face < 6; ++face) {
    INFO("face ", face);
    CHECK(corners[face].size() == 4);

    std::set<int> expected;
    for (int edge = 0; edge < 12; ++edge) {
      auto const [cornerA, cornerB] = mc::EdgeCorners[edge];
      bool const onFaceA = std::find(corners[face].begin(), corners[face].end(), cornerA) != corners[face].end();
      bool const onFaceB = std::find(corners[face].begin(), corners[face].end(), cornerB) != corners[face].end();
      if (onFaceA && onFaceB) expected.insert(edge);
    }

    std::set<int> const listed(mc::FaceEdges[face].begin(), mc::FaceEdges[face].end());
    CHECK(listed == expected);
  }
}

// ------------------------------------ complement symmetry ------------------------------------ //

TEST_CASE("Complementary cases are triangulated independently, and that is intentional") {
  // One might expect the triangle count to be invariant under flipping every corner sign, since
  // that only swaps which side is inside. It is not, and the reason is worth pinning down here
  // rather than rediscovering it in front of a broken mesh.
  //
  // Where a configuration is ambiguous -- the classic example being two corners touching only at
  // a body diagonal -- the surface may either separate them or join them, and both readings are
  // consistent with the corner signs. This table resolves a case and its complement differently:
  // case 5 separates its two corners with 2 triangles, case 250 joins them with 4. The result is
  // still watertight, which is what the face consistency test below establishes, but the topology
  // of the extracted surface is not uniquely determined by the data. Resolving that would take
  // MC33 style subcase tables, and it is a deliberate non-goal here.
  //
  // The count is pinned so that editing the table surfaces as a failure here.
  std::vector<std::uint32_t> asymmetric;
  for (std::uint32_t caseIndex = 0; caseIndex < 256; ++caseIndex) {
    if (mc::TriangleCount(caseIndex) != mc::TriangleCount(255 - caseIndex)) {
      asymmetric.push_back(caseIndex);
    }
  }
  CHECK(asymmetric.size() == 88);

  // A configuration with fewer than 2 corners on a side has no room for an ambiguity, and one
  // split 4/4 happens not to trigger it in this table. Only the counts in between do.
  for (auto const caseIndex : asymmetric) {
    int const insideCount = __builtin_popcount(caseIndex);
    INFO("case ", caseIndex, " has ", insideCount, " inside corners");
    CHECK(insideCount >= 2);
    CHECK(insideCount <= 6);
    CHECK(insideCount != 4);
  }
}

TEST_CASE("A case and its complement use the same set of cell edges") {
  // The isosurface crosses exactly the edges whose corners disagree, and complementing preserves
  // disagreement. Only the winding is allowed to differ.
  for (std::uint32_t caseIndex = 0; caseIndex < 256; ++caseIndex) {
    INFO("case ", caseIndex, " vs ", 255 - caseIndex);
    CHECK(mc::EdgeMask(caseIndex) == mc::EdgeMask(255 - caseIndex));
  }
}

// --------------------------------- manifoldness inside a cell -------------------------------- //

TEST_CASE("The triangles of a case form a manifold patch whose boundary lies on the cell faces") {
  // This is the local half of the watertightness argument: inside a cell the triangles have to fit
  // together without overlapping, and wherever the patch ends it must end on a cell face, where a
  // neighbouring cell can pick it up. A patch ending in mid air is a hole in the final mesh.
  for (std::uint32_t caseIndex = 0; caseIndex < 256; ++caseIndex) {
    INFO("case ", caseIndex, ": ", describe(segmentsOf(caseIndex)));

    auto const segments = segmentsOf(caseIndex);

    std::map<Segment, int> directedCount;
    for (auto const& segment : segments) ++directedCount[segment];

    // No directed segment twice: two triangles running along the same edge in the same direction
    // means they overlap rather than share it.
    for (auto const& [segment, count] : directedCount) {
      INFO("segment ", segment.first, "->", segment.second);
      CHECK(count == 1);
    }

    for (auto const& [segment, count] : directedCount) {
      (void)count;
      auto const reverse = Segment{segment.second, segment.first};
      bool const isInterior = directedCount.count(reverse) != 0;
      if (isInterior) continue;

      // A boundary segment must lie inside one of the 6 cell faces, so that the neighbouring cell
      // sharing that face continues the surface there.
      bool onSomeFace = false;
      for (int face = 0; face < 6 && !onSomeFace; ++face) {
        onSomeFace = edgeIsOnFace(segment.first, face) && edgeIsOnFace(segment.second, face);
      }

      INFO("boundary segment ", segment.first, "->", segment.second, " lies on no cell face");
      CHECK(onSomeFace);
    }
  }
}

// ------------------------- face consistency, i.e. no holes between cells --------------------- //

TEST_CASE("What a case emits on a face is determined by that face's four corner signs alone") {
  // This is the other half of the watertightness argument, and the one that actually rules out the
  // classic Marching Cubes holes.
  //
  // Two cells sharing a face see the very same 4 corner values, hence the same 4 signs, but they
  // generally sit in different full cases. If the segments a case leaves on a face depended on the
  // 4 corners *outside* that face, the two cells could resolve the face differently and leave a
  // crack between them. Grouping all 256 cases by the signs of one face and demanding they agree
  // there is exactly the condition that forbids it -- and it needs no neighbour bookkeeping.
  auto const corners = faceCorners();

  for (int face = 0; face < 6; ++face) {
    // Segments seen so far for each of the 16 sign patterns of this face, plus the case that
    // established them, so that a failure can name both offenders.
    std::map<int, std::pair<std::uint32_t, std::set<Segment>>> byPattern;

    for (std::uint32_t caseIndex = 0; caseIndex < 256; ++caseIndex) {
      int pattern = 0;
      for (int i = 0; i < 4; ++i) {
        if ((caseIndex >> corners[face][i]) & 1u) pattern |= (1 << i);
      }

      std::set<Segment> onFace;
      for (auto const& segment : segmentsOf(caseIndex)) {
        if (edgeIsOnFace(segment.first, face) && edgeIsOnFace(segment.second, face)) {
          onFace.insert(segment);
        }
      }

      auto const existing = byPattern.find(pattern);
      if (existing == byPattern.end()) {
        byPattern.emplace(pattern, std::make_pair(caseIndex, onFace));
        continue;
      }

      INFO("face ", face, ", corner pattern ", pattern,
           ": case ", existing->second.first, " and case ", caseIndex, " disagree");
      CHECK(onFace == existing->second.second);
    }
  }
}
