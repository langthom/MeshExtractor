
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <vector>

#include <cuda_runtime.h>

#include "doctest.h"
#include "MeshAssertions.h"
#include "../MeshExtraction/CudaChunkMeshExtractor.h"
#include "../MeshExtraction/CudaUtils.h"

namespace pme = parallel_mesh_extractor;
namespace ma  = mesh_assertions;

namespace {

  // As in the chunkifier tests, the layout is pinned with literals rather than derived from the
  // production constants, so that moving those shows up here as a failure.
  constexpr std::int64_t ChunkSize  = 64;
  constexpr std::int64_t GhostWidth = 1;
  constexpr std::int64_t CellDim    = ChunkSize - 2 * GhostWidth;  // 62 owned cells per axis
  constexpr std::int64_t CornerDim  = CellDim + 1;                 // 63 corners they touch

  using Chunk = pme::Chunkifier::DataChunk;
  using Field = std::function<float(std::int64_t, std::int64_t, std::int64_t)>;

  /// True when a CUDA device is present. Without one the GPU cases report and return rather than
  /// fail, so that the suite still runs to completion on a machine that has no card.
  bool cudaMissing() {
    if (pme::CudaChunkMeshExtractor::IsCudaAvailable()) return false;
    MESSAGE("no CUDA device available, skipping the GPU extraction case");
    return true;
  }

  /// Build a chunk whose voxels are sampled from a field in *global* voxel coordinates. Local index
  /// l of an axis maps to the global coordinate (CoreOrigin + l - GhostWidth), which is the same
  /// contract the chunkifier fills a chunk by.
  std::unique_ptr<Chunk> makeChunk(std::array<std::int64_t, 3> const& coreOrigin, Field const& field) {
    auto chunk = std::make_unique<Chunk>();
    chunk->CoreOrigin = coreOrigin;

    for (std::int64_t z = 0; z < ChunkSize; ++z) {
      for (std::int64_t y = 0; y < ChunkSize; ++y) {
        for (std::int64_t x = 0; x < ChunkSize; ++x) {
          chunk->data[z][y][x] = field(coreOrigin[0] + x - GhostWidth,
                                       coreOrigin[1] + y - GhostWidth,
                                       coreOrigin[2] + z - GhostWidth);
        }
      }
    }
    return chunk;
  }

  Field constantField(float value) {
    return [value](std::int64_t, std::int64_t, std::int64_t) { return value; };
  }

  /// A solid ball, as a density: high in the interior, falling through zero at the surface and
  /// negative outside. Reading the field this way -- material dense, surroundings not -- is what
  /// makes the extraction's winding come out as the usual outward normal.
  Field ballField(std::array<double, 3> const& center, double radius) {
    return [center, radius](std::int64_t x, std::int64_t y, std::int64_t z) {
      double const dx = static_cast<double>(x) - center[0];
      double const dy = static_cast<double>(y) - center[1];
      double const dz = static_cast<double>(z) - center[2];
      return static_cast<float>(radius - std::sqrt(dx * dx + dy * dy + dz * dz));
    };
  }

  pme::ChunkMesh extract(Chunk const& chunk, float isoThreshold) {
    pme::CudaChunkMeshExtractor extractor(isoThreshold);
    pme::ChunkMesh mesh;
    extractor.Extract(chunk, mesh);
    return mesh;
  }

  /// Every index has to address a vertex that was actually emitted. Checked before anything else
  /// looks at the linkage, because an out of range index would make the rest read out of bounds.
  void requireIndicesInRange(pme::ChunkMesh const& mesh) {
    for (std::size_t i = 0; i < mesh.Indices.size(); ++i) {
      if (mesh.Indices[i] >= mesh.Positions.size()) {
        INFO("index ", i, " is ", mesh.Indices[i], " but only ",
             mesh.Positions.size(), " vertices were emitted");
        REQUIRE(mesh.Indices[i] < mesh.Positions.size());
      }
    }
  }

} // namespace

// Everything below needs a CUDA device. The cases skip themselves when none is present, and the
// suite name lets a machine without a card exclude them outright.
TEST_SUITE_BEGIN("gpu");

// -------------------------------------- nothing to extract ----------------------------------- //

TEST_CASE("A homogeneous chunk carries no surface") {
  if (cudaMissing()) return;

  auto const chunk = makeChunk({0, 0, 0}, constantField(500.0f));

  SUBCASE("isovalue below the data") {
    auto const mesh = extract(*chunk, 100.0f);
    CHECK(mesh.IsEmpty());
    CHECK(mesh.Positions.empty());
  }

  SUBCASE("isovalue above the data") {
    auto const mesh = extract(*chunk, 900.0f);
    CHECK(mesh.IsEmpty());
  }

  SUBCASE("isovalue exactly on the data") {
    // The convention is that a corner is inside when its value is *less than* the isovalue, so a
    // chunk sitting exactly on the isovalue is entirely outside and carries no surface. Pinning it
    // here keeps the chunkifier's inclusive culling and the extraction's strict comparison from
    // drifting apart unnoticed.
    auto const mesh = extract(*chunk, 500.0f);
    CHECK(mesh.IsEmpty());
  }
}

TEST_CASE("The chunk's lower ghost layer is never read") {
  if (cudaMissing()) return;

  // The cells a chunk owns are based at local index GhostWidth and up, so their corners never reach
  // the lowest voxel layer. Here that layer is the only thing sitting on the far side of the
  // isovalue: if the extraction touched it, it would find a surface. It must find none.
  auto chunk = makeChunk({0, 0, 0}, constantField(-1.0f));
  for (std::int64_t y = 0; y < ChunkSize; ++y) {
    for (std::int64_t x = 0; x < ChunkSize; ++x) chunk->data[0][y][x] = +1.0f;
  }

  auto const mesh = extract(*chunk, 0.0f);
  CHECK(mesh.IsEmpty());
}

// ------------------------------------ a single cell of surface ------------------------------- //

TEST_CASE("A single inside voxel produces the octahedron around it") {
  if (cudaMissing()) return;

  // One voxel below the isovalue in an otherwise uniform chunk is a corner of the 8 cells meeting
  // at it, and each of them cuts that corner off with a single triangle. The 6 grid edges leaving
  // the voxel carry the vertices, exactly halfway along each because the isovalue is the midpoint
  // of the two values.
  //
  // The vertex count is the real subject here: the 8 triangles have 24 corners between them, but
  // the 4 triangles meeting at a grid edge must all resolve to the *same* vertex, leaving 6. A
  // triangle soup with an index buffer bolted on would report 24.
  std::array<std::int64_t, 3> const insideVoxel = {30, 31, 32};

  auto chunk = makeChunk({0, 0, 0}, constantField(+1.0f));
  chunk->data[insideVoxel[2] + GhostWidth][insideVoxel[1] + GhostWidth][insideVoxel[0] + GhostWidth] = -1.0f;

  auto const mesh = extract(*chunk, 0.0f);

  REQUIRE(mesh.TriangleCount() == 8);
  REQUIRE(mesh.Positions.size() == 6);
  requireIndicesInRange(mesh);

  // Half a voxel out from the inside corner, one vertex in each of the 6 axis directions.
  std::set<ma::Vertex> expected;
  for (int axis = 0; axis < 3; ++axis) {
    for (float step : {-0.5f, +0.5f}) {
      ma::Vertex vertex = {static_cast<float>(insideVoxel[0]),
                           static_cast<float>(insideVoxel[1]),
                           static_cast<float>(insideVoxel[2])};
      vertex[axis] += step;
      expected.insert(vertex);
    }
  }
  std::set<ma::Vertex> const emitted(mesh.Positions.begin(), mesh.Positions.end());
  CHECK(emitted == expected);

  // And the linkage is that of an octahedron: closed, consistently oriented, genus 0.
  auto const report = ma::analyzeManifold(mesh);
  CHECK(report.BoundaryEdges == 0);
  CHECK(report.NonManifoldEdges == 0);
  CHECK(report.IsConsistentlyOriented);
  CHECK(ma::eulerCharacteristic(mesh) == 2);
}

TEST_CASE("The core origin places the vertices in global coordinates") {
  if (cudaMissing()) return;

  // The same local configuration in a chunk that owns a different part of the volume has to come
  // out shifted by exactly that chunk's core origin -- that is what lets chunk meshes simply be
  // concatenated.
  std::array<std::int64_t, 3> const coreOrigin = {CellDim, 2 * CellDim, 3 * CellDim};

  auto chunk = makeChunk(coreOrigin, constantField(+1.0f));
  chunk->data[32][31][30] = -1.0f;

  auto const mesh = extract(*chunk, 0.0f);
  REQUIRE(mesh.Positions.size() == 6);

  for (auto const& position : mesh.Positions) {
    CHECK(position[0] >= static_cast<float>(coreOrigin[0]));
    CHECK(position[1] >= static_cast<float>(coreOrigin[1]));
    CHECK(position[2] >= static_cast<float>(coreOrigin[2]));
  }

  // Local voxel (30, 31, 32) is the global voxel (coreOrigin + local - GhostWidth).
  ma::Vertex const insideCorner = {
    static_cast<float>(coreOrigin[0] + 30 - GhostWidth),
    static_cast<float>(coreOrigin[1] + 31 - GhostWidth),
    static_cast<float>(coreOrigin[2] + 32 - GhostWidth),
  };
  for (auto const& position : mesh.Positions) {
    CHECK(ma::length(ma::subtract(position, insideCorner)) == doctest::Approx(0.5));
  }
}

// ------------------------------------- an analytic plane ------------------------------------- //

TEST_CASE("A plane comes out flat, complete and consistently wound") {
  if (cudaMissing()) return;

  // An axis aligned plane at a deliberately non-integer height, so that no corner value ever lands
  // exactly on the isovalue and the geometry stays free of degenerate triangles. The field is a
  // density: dense below the plane, empty above it.
  constexpr double planeZ = 31.37;
  auto const chunk = makeChunk({0, 0, 0}, [](std::int64_t, std::int64_t, std::int64_t z) {
    return static_cast<float>(planeZ - static_cast<double>(z));
  });

  auto const mesh = extract(*chunk, 0.0f);

  REQUIRE_FALSE(mesh.IsEmpty());
  requireIndicesInRange(mesh);

  // Every vertex lies in the plane. The field varies only along z, so the interpolation is exact
  // there and only rounding separates the result from the analytic height.
  for (auto const& position : mesh.Positions) {
    CHECK(position[2] == doctest::Approx(planeZ).epsilon(1e-6));
  }

  // The plane cuts every one of the chunk's owned cells at that height, and each contributes its
  // full unit square, so the total area is the owned footprint exactly.
  CHECK(ma::surfaceArea(mesh) == doctest::Approx(static_cast<double>(CellDim * CellDim)).epsilon(1e-5));

  // The winding convention: the surface normal points towards the corners that compare less than
  // the isovalue, which reading the field as a density means out of the material and into the
  // empty space. The material is below the plane here, so every normal points to +z. A flipped
  // table row would leave the area intact and show up right here.
  for (std::size_t triangle = 0; triangle < mesh.TriangleCount(); ++triangle) {
    auto const normal = ma::triangleNormal(mesh, triangle);
    INFO("triangle ", triangle, " normal z = ", normal[2]);
    CHECK(normal[2] > 0.0f);
    CHECK(std::abs(normal[0]) < 1e-4f);
    CHECK(std::abs(normal[1]) < 1e-4f);
  }
}

// ---------------------------------------- a sphere ------------------------------------------- //

TEST_CASE("A sphere comes out as a closed, correctly sized surface") {
  if (cudaMissing()) return;

  // Placed so that the whole sphere sits well inside the region the chunk owns, with the isovalue
  // offset slightly off the grid to keep corner values away from it.
  constexpr double radius = 20.0;
  std::array<double, 3> const center = {31.5, 31.5, 31.5};

  auto const chunk = makeChunk({0, 0, 0}, ballField(center, radius));
  auto const mesh = extract(*chunk, 0.0f);

  REQUIRE_FALSE(mesh.IsEmpty());
  requireIndicesInRange(mesh);

  SUBCASE("every vertex sits on the sphere") {
    double worst = 0.0;
    for (auto const& position : mesh.Positions) {
      ma::Vertex const offset = {static_cast<float>(position[0] - center[0]),
                                 static_cast<float>(position[1] - center[1]),
                                 static_cast<float>(position[2] - center[2])};
      worst = std::max(worst, std::abs(ma::length(offset) - radius));
    }
    INFO("largest radial deviation: ", worst);
    CHECK(worst < 0.02);
  }

  SUBCASE("the mesh is a closed, consistently oriented surface") {
    // Nothing here depends on the geometry: it is purely a statement about the index buffer, which
    // is only correct if the edge ownership resolved every shared vertex to the same slot.
    auto const report = ma::analyzeManifold(mesh);
    INFO("boundary edges: ", report.BoundaryEdges, ", non manifold edges: ", report.NonManifoldEdges);
    CHECK(report.BoundaryEdges == 0);
    CHECK(report.NonManifoldEdges == 0);
    CHECK(report.IsConsistentlyOriented);
  }

  SUBCASE("the mesh is topologically a sphere") {
    CHECK(ma::eulerCharacteristic(mesh) == 2);
  }

  SUBCASE("no vertex is emitted that no triangle uses") {
    // An active grid edge is always referenced by at least one of the cells around it, so an
    // orphan vertex can only come from the edge slot arithmetic addressing the wrong slot.
    std::set<std::uint32_t> used(mesh.Indices.begin(), mesh.Indices.end());
    CHECK(used.size() == mesh.Positions.size());
  }

  SUBCASE("no position is emitted twice") {
    // One vertex per grid edge is the whole point of the indexed output. Duplicates would mean the
    // extraction fell back to a triangle soup with an index buffer bolted on.
    std::set<ma::PositionKey> distinct;
    for (auto const& position : mesh.Positions) distinct.insert(ma::positionKey(position));
    CHECK(distinct.size() == mesh.Positions.size());
  }

  SUBCASE("the area and the enclosed volume match the analytic sphere") {
    constexpr double pi = 3.14159265358979323846;
    CHECK(ma::surfaceArea(mesh) == doctest::Approx(4.0 * pi * radius * radius).epsilon(0.02));

    // Positive, because the winding puts the normals on the outside. The sign is as much the
    // subject of this check as the magnitude.
    CHECK(ma::enclosedVolume(mesh)
          == doctest::Approx(4.0 / 3.0 * pi * radius * radius * radius).epsilon(0.02));
  }
}

// ------------------------------- corner values on the isovalue ------------------------------- //

TEST_CASE("Values landing exactly on the isovalue keep the linkage intact") {
  if (cudaMissing()) return;

  // A single voxel exactly at the isovalue, surrounded by voxels below it. Every one of the six
  // grid edges meeting that voxel is then active, and every one of them interpolates to the voxel
  // itself: all six vertices land on the same point and the eight triangles around it collapse.
  //
  // The extraction emits them anyway, which is the deliberate policy: the triangles still form a
  // valid closed octahedron in the index buffer, and dropping them would tear a hole into exactly
  // the linkage this extraction exists to produce. Only the geometry degenerates.
  std::array<std::int64_t, 3> const peak = {31, 31, 31};

  auto chunk = makeChunk({0, 0, 0}, constantField(-1.0f));
  chunk->data[peak[2] + GhostWidth][peak[1] + GhostWidth][peak[0] + GhostWidth] = 0.0f;

  auto const mesh = extract(*chunk, 0.0f);

  requireIndicesInRange(mesh);
  CHECK(mesh.Positions.size() == 6);
  CHECK(mesh.TriangleCount() == 8);

  // Geometrically collapsed: every vertex is the peak voxel itself.
  ma::Vertex const expected = {static_cast<float>(peak[0]),
                               static_cast<float>(peak[1]),
                               static_cast<float>(peak[2])};
  for (auto const& position : mesh.Positions) CHECK(position == expected);

  CHECK(ma::surfaceArea(mesh) == doctest::Approx(0.0));

  // Topologically intact: an octahedron, closed and consistently oriented.
  auto const report = ma::analyzeManifold(mesh);
  CHECK(report.BoundaryEdges == 0);
  CHECK(report.NonManifoldEdges == 0);
  CHECK(report.IsConsistentlyOriented);
  CHECK(ma::eulerCharacteristic(mesh) == 2);
}

// ------------------------------------ reproducibility ---------------------------------------- //

TEST_CASE("Extracting the same chunk twice yields identical output") {
  if (cudaMissing()) return;

  // Every write offset comes from a prefix sum rather than from an atomic counter, so the output
  // is bit for bit reproducible. This is the test that fails the day that changes.
  auto const chunk = makeChunk({0, 0, 0}, ballField({31.5, 31.5, 31.5}, 18.0));

  pme::CudaChunkMeshExtractor extractor(0.0f);
  pme::ChunkMesh first, second;
  extractor.Extract(*chunk, first);
  extractor.Extract(*chunk, second);

  REQUIRE_FALSE(first.IsEmpty());
  CHECK(first.Positions == second.Positions);
  CHECK(first.Indices == second.Indices);
}

TEST_CASE("An extractor reused across chunks keeps no state between them") {
  if (cudaMissing()) return;

  // The device buffers are allocated once and reused, so a chunk could in principle be
  // contaminated by whatever the previous one left behind. Interleaving two very different chunks
  // and returning to the first is what would expose it.
  auto const sphere = makeChunk({0, 0, 0}, ballField({31.5, 31.5, 31.5}, 18.0));
  auto const plane  = makeChunk({0, 0, 0}, [](std::int64_t, std::int64_t, std::int64_t z) {
    return static_cast<float>(static_cast<double>(z) - 20.5);
  });
  auto const empty  = makeChunk({0, 0, 0}, constantField(7.0f));

  pme::CudaChunkMeshExtractor extractor(0.0f);

  pme::ChunkMesh firstPass, secondPass, planeMesh, emptyMesh;
  extractor.Extract(*sphere, firstPass);
  extractor.Extract(*plane,  planeMesh);
  extractor.Extract(*empty,  emptyMesh);
  extractor.Extract(*sphere, secondPass);

  REQUIRE_FALSE(firstPass.IsEmpty());
  REQUIRE_FALSE(planeMesh.IsEmpty());
  CHECK(emptyMesh.IsEmpty());

  CHECK(firstPass.Positions == secondPass.Positions);
  CHECK(firstPass.Indices == secondPass.Indices);
}

TEST_CASE("Repeated extraction does not leak device memory") {
  if (cudaMissing()) return;

  pme::CudaChunkMeshExtractor extractor(0.0f);
  auto const chunk = makeChunk({0, 0, 0}, ballField({31.5, 31.5, 31.5}, 18.0));

  pme::ChunkMesh mesh;
  extractor.Extract(*chunk, mesh);  // settle any lazy runtime allocation before measuring

  std::size_t freeBefore = 0, total = 0;
  REQUIRE(cudaMemGetInfo(&freeBefore, &total) == cudaSuccess);

  for (int i = 0; i < 100; ++i) extractor.Extract(*chunk, mesh);

  std::size_t freeAfter = 0;
  REQUIRE(cudaMemGetInfo(&freeAfter, &total) == cudaSuccess);
  CHECK(freeAfter == freeBefore);
}

// ----------------------------------------- robustness ---------------------------------------- //

TEST_CASE("Non finite voxels damage their own neighbourhood and nothing else") {
  if (cudaMissing()) return;

  // Every comparison against a NaN is false, so a NaN voxel classifies as outside. The
  // classification therefore stays total and deterministic. Interpolating across such a voxel does
  // not: (iso - NaN) / (v - NaN) is a NaN, and so is the infinity case, so the vertices on the
  // edges touching a damaged voxel come out non finite.
  //
  // That is inherent to linear interpolation rather than something worth branching on in the
  // kernel, and this test pins the two properties that do matter: the result is reproducible, and
  // the damage stays strictly local instead of contaminating the rest of the surface.
  auto const clean = makeChunk({0, 0, 0}, ballField({31.5, 31.5, 31.5}, 18.0));

  auto damaged = makeChunk({0, 0, 0}, ballField({31.5, 31.5, 31.5}, 18.0));
  damaged->data[20][20][20] = std::numeric_limits<float>::quiet_NaN();
  damaged->data[21][20][20] = std::numeric_limits<float>::infinity();
  damaged->data[22][20][20] = -std::numeric_limits<float>::infinity();

  pme::CudaChunkMeshExtractor extractor(0.0f);
  pme::ChunkMesh cleanMesh, first, second;

  REQUIRE_NOTHROW(extractor.Extract(*clean, cleanMesh));
  REQUIRE_NOTHROW(extractor.Extract(*damaged, first));
  REQUIRE_NOTHROW(extractor.Extract(*damaged, second));

  requireIndicesInRange(first);

  // Reproducible, compared bitwise because a NaN never compares equal to itself.
  REQUIRE(first.Positions.size() == second.Positions.size());
  for (std::size_t i = 0; i < first.Positions.size(); ++i) {
    REQUIRE(ma::positionKey(first.Positions[i]) == ma::positionKey(second.Positions[i]));
  }
  CHECK(first.Indices == second.Indices);

  // Three damaged voxels touch at most 3 * 6 grid edges, so no more than that many vertices can
  // come out non finite. A larger count would mean the damage spread.
  std::size_t nonFinite = 0;
  for (auto const& position : first.Positions) {
    if (!std::isfinite(position[0]) || !std::isfinite(position[1]) || !std::isfinite(position[2])) {
      ++nonFinite;
    }
  }
  INFO("non finite vertices: ", nonFinite);
  CHECK(nonFinite > 0);       // the damage is real, so the bound below is actually being tested
  CHECK(nonFinite <= 3 * 6);

  // The ball itself sits far from the damaged voxels and has to come out untouched.
  auto const cleanTriangles   = ma::canonicalTriangles(cleanMesh);
  auto const damagedTriangles = ma::canonicalTriangles(first);
  std::size_t surviving = 0;
  for (auto const& triangle : cleanTriangles) {
    if (std::binary_search(damagedTriangles.begin(), damagedTriangles.end(), triangle)) ++surviving;
  }
  INFO(surviving, " of ", cleanTriangles.size(), " undamaged triangles survived");
  CHECK(surviving == cleanTriangles.size());
}

TEST_CASE("A failing CUDA call is reported as an exception") {
  if (cudaMissing()) return;

  // The extraction reports by throwing rather than by aborting, so that a caller can recover and
  // so that the surrounding objects still destruct cleanly. An allocation far past what any device
  // has is a failure the runtime reports without poisoning the context.
  void* impossible = nullptr;
  CHECK_THROWS_AS(CUDA_CHECK(cudaMalloc(&impossible, ~std::size_t(0))), pme::CudaError);

  // Clear the recorded error so that it is not misattributed to a later call.
  cudaGetLastError();

  // The extractor must still be usable afterwards.
  auto const chunk = makeChunk({0, 0, 0}, ballField({31.5, 31.5, 31.5}, 18.0));
  pme::CudaChunkMeshExtractor extractor(0.0f);
  pme::ChunkMesh mesh;
  REQUIRE_NOTHROW(extractor.Extract(*chunk, mesh));
  CHECK_FALSE(mesh.IsEmpty());
}

TEST_SUITE_END();
