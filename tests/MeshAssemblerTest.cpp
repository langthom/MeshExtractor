
#include <array>
#include <cstdint>
#include <vector>

#include "doctest.h"
#include "MeshAssertions.h"
#include "../MeshExtraction/MeshAssembler.h"

namespace pme = parallel_mesh_extractor;
namespace ma  = mesh_assertions;

namespace {

  /// A chunk mesh built by hand, so that a merge can be reasoned about vertex by vertex.
  pme::ChunkMesh meshOf(std::vector<std::array<float, 3>> positions,
                        std::vector<std::uint32_t> indices) {
    pme::ChunkMesh mesh;
    mesh.Positions = std::move(positions);
    mesh.Indices = std::move(indices);
    return mesh;
  }

  /// Merge one mesh and return what the assembler handed back, so a test can read it directly.
  struct Emitted {
    std::vector<std::array<float, 3>> Vertices;
    std::vector<std::uint32_t> Indices;
  };

  Emitted add(pme::MeshAssembler& assembler, pme::ChunkMesh const& mesh) {
    Emitted emitted;
    assembler.Add(mesh, emitted.Vertices, emitted.Indices);
    return emitted;
  }

  /// Rebuild the finished mesh from the pieces the assembler handed out, which is what a caller
  /// streaming them to a file ends up with.
  void appendEmitted(pme::ChunkMesh& assembled, Emitted const& emitted) {
    assembled.Positions.insert(assembled.Positions.end(),
                               emitted.Vertices.begin(), emitted.Vertices.end());
    assembled.Indices.insert(assembled.Indices.end(),
                             emitted.Indices.begin(), emitted.Indices.end());
  }

} // namespace

// ------------------------------------- merging vertices -------------------------------------- //

TEST_CASE("A single chunk passes through with its own vertices renumbered from zero") {
  pme::MeshAssembler assembler;

  auto const mesh = meshOf({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2});
  auto const emitted = add(assembler, mesh);

  CHECK(emitted.Vertices == mesh.Positions);
  CHECK(emitted.Indices == std::vector<std::uint32_t>{0, 1, 2});
  CHECK(assembler.VertexCount() == 3);
  CHECK(assembler.TriangleCount() == 1);
}

TEST_CASE("Shared positions merge, and the later chunk's triangles point at the earlier vertices") {
  // The whole job: the second chunk brings two vertices the first already contributed, so only its
  // genuinely new one comes out, and its triangle refers back to the originals.
  pme::MeshAssembler assembler;

  auto const first  = meshOf({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2});
  auto const second = meshOf({{1, 0, 0}, {0, 1, 0}, {1, 1, 0}}, {0, 1, 2});

  auto const emittedFirst  = add(assembler, first);
  auto const emittedSecond = add(assembler, second);

  CHECK(emittedFirst.Vertices.size() == 3);
  REQUIRE(emittedSecond.Vertices.size() == 1);
  CHECK(emittedSecond.Vertices[0] == std::array<float, 3>{1, 1, 0});

  // (1,0,0) and (0,1,0) were global 1 and 2; the new (1,1,0) becomes global 3.
  CHECK(emittedSecond.Indices == std::vector<std::uint32_t>{1, 2, 3});

  CHECK(assembler.VertexCount() == 4);
  CHECK(assembler.TriangleCount() == 2);
}

TEST_CASE("Chunks with nothing in common are simply concatenated") {
  pme::MeshAssembler assembler;

  auto const first  = meshOf({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2});
  auto const second = meshOf({{9, 9, 9}, {8, 9, 9}, {9, 8, 9}}, {0, 1, 2});

  add(assembler, first);
  auto const emitted = add(assembler, second);

  CHECK(emitted.Vertices == second.Positions);
  CHECK(emitted.Indices == std::vector<std::uint32_t>{3, 4, 5});
  CHECK(assembler.VertexCount() == 6);
}

TEST_CASE("An empty chunk mesh contributes nothing") {
  // The extraction hands back an empty mesh for a chunk the culling let through without a surface,
  // so this is a normal occurrence rather than an edge case.
  pme::MeshAssembler assembler;

  auto const emitted = add(assembler, pme::ChunkMesh{});
  CHECK(emitted.Vertices.empty());
  CHECK(emitted.Indices.empty());
  CHECK(assembler.VertexCount() == 0);
  CHECK(assembler.TriangleCount() == 0);
}

TEST_CASE("The two encodings of zero describe the same vertex") {
  // -0.0f and +0.0f compare equal as floats but differ bit for bit. Matching on raw bits without
  // normalising them would split a vertex sitting on an axis into two.
  pme::MeshAssembler assembler;

  auto const positive = meshOf({{0.0f, 1.0f, 2.0f}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2});
  auto const negative = meshOf({{-0.0f, 1.0f, 2.0f}, {5, 5, 5}, {6, 6, 6}}, {0, 1, 2});

  add(assembler, positive);
  auto const emitted = add(assembler, negative);

  INFO("the shared vertex should not have been emitted a second time");
  CHECK(emitted.Vertices.size() == 2);
  CHECK(emitted.Indices[0] == 0);
  CHECK(assembler.VertexCount() == 5);
}

// --------------------------------- triangles that collapse ----------------------------------- //

TEST_CASE("A triangle whose corners weld together is dropped rather than emitted degenerate") {
  // Where a corner value lands exactly on the isovalue -- the rule for an integer volume read at
  // an integer isovalue -- several grid edges interpolate to that same corner, and welding merges
  // them. The triangle between them is then a line, carries no surface, and is left out.
  pme::MeshAssembler assembler;

  auto const mesh = meshOf({{4, 4, 4}, {4, 4, 4}, {5, 4, 4}}, {0, 1, 2});
  auto const emitted = add(assembler, mesh);

  CHECK(emitted.Indices.empty());
  CHECK(assembler.TriangleCount() == 0);
  CHECK(assembler.DroppedTriangleCount() == 1);

  INFO("a vertex no surviving triangle refers to has no business in the file either");
  CHECK(emitted.Vertices.empty());
  CHECK(assembler.VertexCount() == 0);
}

TEST_CASE("Dropping a collapsed triangle leaves its neighbours untouched") {
  // The collapsed triangle sits between two intact ones and shares vertices with both, so this
  // pins down that only the triangle goes and none of the geometry around it.
  pme::MeshAssembler assembler;

  auto const mesh = meshOf({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}},
                           {0, 1, 2,      // intact
                            1, 3, 3,      // collapses onto the edge 1-3
                            1, 3, 2});    // intact
  auto const emitted = add(assembler, mesh);

  CHECK(emitted.Vertices.size() == 4);
  CHECK(emitted.Indices == std::vector<std::uint32_t>{0, 1, 2, 1, 3, 2});
  CHECK(assembler.TriangleCount() == 2);
  CHECK(assembler.DroppedTriangleCount() == 1);
}

TEST_CASE("Corners that collapse across chunks are dropped too") {
  // The two corners are distinct within the chunk and only become one vertex through the weld, so
  // the test has to run through the assembler rather than through the chunk's own indexing.
  pme::MeshAssembler assembler;

  auto const first = meshOf({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2});
  add(assembler, first);

  // Two of the chunk's own positions hold the same point. They are separate vertices as far as the
  // chunk is concerned, and only the weld makes the triangle's last two corners one and the same.
  auto const second = meshOf({{1, 0, 0}, {0, 1, 0}, {0, 1, 0}}, {0, 1, 2});
  auto const emitted = add(assembler, second);

  CHECK(emitted.Vertices.empty());
  CHECK(emitted.Indices.empty());
  CHECK(assembler.TriangleCount() == 1);
  CHECK(assembler.DroppedTriangleCount() == 1);
}

TEST_CASE("A collapsed triangle does not open the surface it was part of") {
  // Dropping the triangle has to leave the mesh as closed as it was: its two non-self edges are
  // one edge traversed both ways and cancel against each other, so no boundary appears.
  pme::MeshAssembler assembler;
  pme::ChunkMesh assembled;

  // A tetrahedron, with one extra triangle pinched flat against one of its edges.
  auto const mesh = meshOf({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                           {0, 2, 1,
                            0, 1, 3,
                            0, 3, 2,
                            1, 2, 3,
                            1, 3, 3});
  appendEmitted(assembled, add(assembler, mesh));

  REQUIRE(assembled.TriangleCount() == 4);

  auto const report = ma::analyzeManifold(assembled);
  CHECK(report.IsWatertight);
  CHECK(report.IsManifold);
  CHECK(report.IsConsistentlyOriented);
  CHECK(ma::eulerCharacteristic(assembled) == 2);
}

TEST_CASE("The two encodings of zero collapse a triangle just as an exact repeat does") {
  // Welding normalises -0.0 onto +0.0, so the collapse test has to use the same normalisation or
  // it would pass a triangle through that the weld then turns degenerate anyway.
  pme::MeshAssembler assembler;

  auto const mesh = meshOf({{0.0f, 1.0f, 2.0f}, {-0.0f, 1.0f, 2.0f}, {5, 5, 5}}, {0, 1, 2});
  auto const emitted = add(assembler, mesh);

  CHECK(emitted.Indices.empty());
  CHECK(assembler.DroppedTriangleCount() == 1);
}

// -------------------------------------- pruning the cache ------------------------------------ //

TEST_CASE("Pruning drops what can no longer be referenced and keeps the shared plane") {
  // The comparison has to be strict: a vertex sitting exactly on the plane two chunk layers share
  // is the one the next layer is about to ask for.
  pme::MeshAssembler assembler;

  auto const mesh = meshOf({{0, 0, 61.5f}, {0, 0, 62.0f}, {0, 0, 62.5f}}, {0, 1, 2});
  add(assembler, mesh);
  REQUIRE(assembler.CachedVertexCount() == 3);

  assembler.PruneBelowZ(62.0f);
  CHECK(assembler.CachedVertexCount() == 2);

  // What survived still merges.
  auto const next = meshOf({{0, 0, 62.0f}, {0, 0, 62.5f}, {1, 1, 63.0f}}, {0, 1, 2});
  auto const emitted = add(assembler, next);

  REQUIRE(emitted.Vertices.size() == 1);
  CHECK(emitted.Indices == std::vector<std::uint32_t>{1, 2, 3});
}

TEST_CASE("A pruned vertex met again is emitted afresh") {
  // Pruning is an assertion by the caller that a vertex cannot come back. If one does anyway, it
  // has to become a new vertex rather than a dangling index -- the mesh is then merely duplicated
  // there, not broken.
  pme::MeshAssembler assembler;

  auto const mesh = meshOf({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2});
  add(assembler, mesh);

  assembler.PruneBelowZ(100.0f);
  CHECK(assembler.CachedVertexCount() == 0);

  auto const emitted = add(assembler, mesh);
  CHECK(emitted.Vertices.size() == 3);
  CHECK(emitted.Indices == std::vector<std::uint32_t>{3, 4, 5});
  CHECK(assembler.VertexCount() == 6);
}

TEST_CASE("Pruning nothing away leaves the cache intact") {
  pme::MeshAssembler assembler;

  auto const mesh = meshOf({{0, 0, 10}, {1, 0, 11}, {0, 1, 12}}, {0, 1, 2});
  add(assembler, mesh);

  assembler.PruneBelowZ(0.0f);
  CHECK(assembler.CachedVertexCount() == 3);
}

// ------------------------------ agreement with an independent weld ---------------------------- //

TEST_CASE("Assembling agrees with an independently written weld") {
  // The oracle in MeshAssertions is a separate implementation kept deliberately apart from this
  // one, so that a bug in either shows up as a disagreement rather than cancelling out.
  pme::MeshAssembler assembler;
  pme::ChunkMesh assembled;

  // Three overlapping pieces, each sharing part of its vertices with the ones before it.
  std::vector<pme::ChunkMesh> pieces = {
    meshOf({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}, {0, 1, 2, 1, 3, 2}),
    meshOf({{1, 0, 0}, {1, 1, 0}, {2, 0, 0}, {2, 1, 0}}, {0, 2, 1, 2, 3, 1}),
    meshOf({{0, 1, 0}, {1, 1, 0}, {0, 2, 0}, {1, 2, 0}}, {0, 1, 2, 1, 3, 2}),
  };

  pme::ChunkMesh concatenated;
  for (auto const& piece : pieces) {
    appendEmitted(assembled, add(assembler, piece));
    ma::append(concatenated, piece);
  }

  auto const oracle = ma::weldByExactPosition(concatenated);

  CHECK(assembled.Positions.size() == oracle.Positions.size());
  CHECK(assembled.TriangleCount() == oracle.TriangleCount());
  CHECK(assembler.VertexCount() == assembled.Positions.size());
  CHECK(assembler.TriangleCount() == assembled.TriangleCount());

  // Same geometry, and the same connectivity once both are read as triangles of positions.
  CHECK(ma::canonicalTriangles(assembled) == ma::canonicalTriangles(oracle));
}
