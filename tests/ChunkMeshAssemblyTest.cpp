
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "doctest.h"
#include "MeshAssertions.h"
#include "ReferenceMarchingCubes.h"
#include "../MeshExtraction/Chunkifier.h"
#include "../MeshExtraction/CudaChunkMeshExtractor.h"

namespace pme = parallel_mesh_extractor;
namespace ma  = mesh_assertions;

namespace {

  constexpr std::int64_t CoreSize = 62;  // the chunk to chunk stride, pinned as a literal

  using Dims = std::array<std::uint32_t, 3>;

  bool cudaMissing() {
    if (pme::CudaChunkMeshExtractor::IsCudaAvailable()) return false;
    MESSAGE("no CUDA device available, skipping the GPU assembly case");
    return true;
  }

  struct Volume {
    std::vector<float> Data;
    Dims Dimensions{};
    float Background = 0.0f;

    /// Reads a voxel, or the background where the coordinate leaves the volume. This is the same
    /// padding the chunkifier writes into the parts of a chunk that reach outside, so it is what
    /// the reference implementation has to see in order to be comparable.
    float Sample(std::int64_t x, std::int64_t y, std::int64_t z) const {
      if (x < 0 || y < 0 || z < 0) return this->Background;
      if (x >= this->Dimensions[0] || y >= this->Dimensions[1] || z >= this->Dimensions[2]) {
        return this->Background;
      }
      auto const index = (z * this->Dimensions[1] + y) * this->Dimensions[0] + x;
      return this->Data[static_cast<std::size_t>(index)];
    }
  };

  template<class Field>
  Volume makeVolume(Dims const& dimensions, float background, Field const& field) {
    Volume volume;
    volume.Dimensions = dimensions;
    volume.Background = background;
    volume.Data.resize(static_cast<std::size_t>(dimensions[0]) * dimensions[1] * dimensions[2]);

    std::size_t index = 0;
    for (std::int64_t z = 0; z < dimensions[2]; ++z) {
      for (std::int64_t y = 0; y < dimensions[1]; ++y) {
        for (std::int64_t x = 0; x < dimensions[0]; ++x) volume.Data[index++] = field(x, y, z);
      }
    }
    return volume;
  }

  /// A solid ball as a density, dense inside and negative outside.
  auto ballField(std::array<double, 3> const& center, double radius) {
    return [center, radius](std::int64_t x, std::int64_t y, std::int64_t z) {
      double const dx = static_cast<double>(x) - center[0];
      double const dy = static_cast<double>(y) - center[1];
      double const dz = static_cast<double>(z) - center[2];
      return static_cast<float>(radius - std::sqrt(dx * dx + dy * dy + dz * dz));
    };
  }

  /// A smooth field with a good deal of structure, so that the surface runs through many chunks in
  /// many orientations rather than presenting the extraction with one tidy blob. The offset keeps
  /// the values away from the isovalue, which keeps the geometry free of degenerate triangles.
  auto rippleField() {
    return [](std::int64_t x, std::int64_t y, std::int64_t z) {
      return std::sin(0.21f * x) + std::sin(0.17f * y) + std::sin(0.13f * z) + 0.11f;
    };
  }

  /// The same ripples, but confined to a blob well away from the volume walls, so that the field is
  /// uniformly empty everywhere near the boundary. That matters whenever a test wants a *closed*
  /// surface: a surface that reaches the lower volume wall is left open there, which the dedicated
  /// test below pins down.
  auto wrinkledBlobField(std::array<double, 3> const& center, double radius) {
    auto const ripple = rippleField();
    return [center, radius, ripple](std::int64_t x, std::int64_t y, std::int64_t z) {
      double const dx = static_cast<double>(x) - center[0];
      double const dy = static_cast<double>(y) - center[1];
      double const dz = static_cast<double>(z) - center[2];
      return static_cast<float>(radius - std::sqrt(dx * dx + dy * dy + dz * dz))
           + 3.0f * ripple(x, y, z);
    };
  }

  /// The cell domain the chunks cover between them: the chunkifier lays out ceil(dim / CoreSize)
  /// chunks per axis and each owns CoreSize cells, so the union reaches past the volume and into
  /// the background padding.
  std::array<std::int64_t, 3> paddedCellEnd(Dims const& dimensions) {
    std::array<std::int64_t, 3> end{};
    for (int axis = 0; axis < 3; ++axis) {
      auto const tiles = (dimensions[axis] + CoreSize - 1) / CoreSize;
      end[axis] = tiles * CoreSize;
    }
    return end;
  }

  /// Chunk the volume, extract every surviving chunk on the GPU and concatenate the results. No
  /// welding: this is the raw union, exactly as the pipeline produces it.
  pme::ChunkMesh extractByChunks(Volume const& volume, float isoThreshold, std::size_t* chunkCount = nullptr) {
    pme::Chunkifier chunkifier(volume.Data.data(), volume.Dimensions, isoThreshold, volume.Background);
    chunkifier.ComputeChunking(volume.Dimensions);

    pme::CudaChunkMeshExtractor extractor(isoThreshold);
    pme::ChunkMesh assembled, single;
    std::size_t count = 0;

    for (auto const& chunk : chunkifier) {
      extractor.Extract(chunk, single);
      ma::append(assembled, single);
      ++count;
    }

    if (chunkCount) *chunkCount = count;
    return assembled;
  }

  pme::ChunkMesh extractWhole(Volume const& volume, float isoThreshold) {
    return reference_marching_cubes::extract(
      [&volume](std::int64_t x, std::int64_t y, std::int64_t z) { return volume.Sample(x, y, z); },
      {0, 0, 0}, paddedCellEnd(volume.Dimensions), isoThreshold);
  }

  std::size_t degenerateTriangleCount(pme::ChunkMesh const& mesh) {
    std::size_t count = 0;
    for (std::size_t triangle = 0; triangle < mesh.TriangleCount(); ++triangle) {
      auto const a = mesh.Indices[3 * triangle + 0];
      auto const b = mesh.Indices[3 * triangle + 1];
      auto const c = mesh.Indices[3 * triangle + 2];
      if (a == b || b == c || c == a) ++count;
    }
    return count;
  }

} // namespace

// Everything below needs a CUDA device. The cases skip themselves when none is present, and the
// suite name lets a machine without a card exclude them outright.
TEST_SUITE_BEGIN("gpu");

// ------------------------------ agreement with a host reference ------------------------------ //

TEST_CASE("A single chunk agrees with a host reference implementation") {
  if (cudaMissing()) return;

  // A volume that fits into one chunk, so this isolates the extraction of a chunk from the
  // question of how chunks fit together. The comparison is on exact bit patterns rather than on a
  // tolerance: the reference deliberately evaluates the interpolation the same way round as the
  // kernel does, so anything but exact agreement is a real difference.
  Dims const dimensions = {50, 50, 50};
  auto const volume = makeVolume(dimensions, -1000.0f, rippleField());

  auto const fromGpu       = extractByChunks(volume, 0.0f);
  auto const fromReference = extractWhole(volume, 0.0f);

  REQUIRE(fromReference.TriangleCount() > 0);
  CHECK(fromGpu.TriangleCount() == fromReference.TriangleCount());
  CHECK(ma::canonicalTriangles(fromGpu) == ma::canonicalTriangles(fromReference));
}

TEST_CASE("A chunked extraction equals the same volume extracted in one piece") {
  if (cudaMissing()) return;

  // Several chunks per axis, so every interface between chunks is exercised, including the ones
  // where the last chunk of an axis reaches into the background padding. Any cell processed twice
  // or not at all shows up as a difference in the triangle set.
  Dims const dimensions = {100, 100, 100};
  auto const volume = makeVolume(dimensions, -1000.0f, rippleField());

  std::size_t chunkCount = 0;
  auto const fromGpu       = extractByChunks(volume, 0.0f, &chunkCount);
  auto const fromReference = extractWhole(volume, 0.0f);

  // Two chunks per axis at this size, minus whatever the value range culling removes.
  INFO("materialized ", chunkCount, " chunks");
  CHECK(chunkCount > 0);
  CHECK(chunkCount <= 8);

  REQUIRE(fromReference.TriangleCount() > 0);
  CHECK(fromGpu.TriangleCount() == fromReference.TriangleCount());
  CHECK(ma::canonicalTriangles(fromGpu) == ma::canonicalTriangles(fromReference));
}

TEST_CASE("A chunk packed with surface stays within the preallocated bounds") {
  if (cudaMissing()) return;

  // Alternating voxel signs put a crossing on essentially every grid edge, which is about as close
  // to the worst case as a real buffer gets. The device buffers are sized for the true worst case,
  // so this must come through without overflowing, and it must still match the reference.
  Dims const dimensions = {62, 62, 62};
  auto const volume = makeVolume(dimensions, -1.0f, [](std::int64_t x, std::int64_t y, std::int64_t z) {
    return ((x + y + z) % 2 == 0) ? +1.0f : -1.0f;
  });

  auto const fromGpu       = extractByChunks(volume, 0.0f);
  auto const fromReference = extractWhole(volume, 0.0f);

  INFO("triangles: ", fromGpu.TriangleCount(), ", vertices: ", fromGpu.Positions.size());
  CHECK(fromGpu.TriangleCount() == fromReference.TriangleCount());
  CHECK(ma::canonicalTriangles(fromGpu) == ma::canonicalTriangles(fromReference));

  // Well past the density of any realistic dataset, which is the point of running it.
  CHECK(fromGpu.TriangleCount() > 500000);
  CHECK(fromGpu.TriangleCount() <= pme::CudaChunkMeshExtractor::MaxTriangles);
  CHECK(fromGpu.Positions.size() <= pme::CudaChunkMeshExtractor::MaxVertices);
}

// ------------------------------- watertightness across chunks -------------------------------- //

TEST_CASE("The chunk meshes weld into a watertight surface") {
  if (cudaMissing()) return;

  // This is what the whole ghost shell design exists for. Each chunk is extracted in complete
  // isolation, yet two chunks meeting at a grid edge read the same two voxel values and run the
  // same interpolation, so their vertices agree to the last bit. Welding on exact equality and
  // finding a surface without a single open edge is the proof that they do; an epsilon would have
  // hidden it.
  //
  // The field is deliberately wrinkly, so that the chunk interfaces are crossed by surface in
  // every orientation rather than by one tidy blob, and deliberately confined well away from the
  // volume walls, so that nothing here depends on how the boundary of the volume is handled.
  Dims const dimensions = {190, 190, 190};
  auto const volume = makeVolume(dimensions, -1000.0f, wrinkledBlobField({ 95.5, 95.5, 95.5}, 70.25));

  std::size_t chunkCount = 0;
  auto const assembled = extractByChunks(volume, 0.0f, &chunkCount);

  INFO("materialized ", chunkCount, " chunks");
  REQUIRE(chunkCount > 1);
  REQUIRE_FALSE(assembled.IsEmpty());

  // The precondition for exact welding: no corner value landed on the isovalue, so no two distinct
  // vertices of a chunk share a position and the collapse below can only merge across chunks.
  REQUIRE(degenerateTriangleCount(assembled) == 0);

  auto const welded = ma::weldByExactPosition(assembled);

  // Welding has to actually do something, otherwise the test would pass on a mesh whose chunks
  // never touched.
  INFO(assembled.Positions.size(), " vertices welded down to ", welded.Positions.size());
  CHECK(welded.Positions.size() < assembled.Positions.size());

  auto const report = ma::analyzeManifold(welded);
  INFO("boundary edges: ", report.BoundaryEdges, ", non manifold edges: ", report.NonManifoldEdges);

  // Watertight: not one open edge anywhere in the assembled surface.
  CHECK(report.BoundaryEdges == 0);
  CHECK(report.IsWatertight);
  CHECK(report.IsConsistentlyOriented);

  // Manifoldness is a different and weaker claim. The classic Marching Cubes tables can pinch the
  // surface against itself where a configuration is ambiguous, leaving an edge shared by four
  // triangles instead of two. The result still encloses a well defined volume, and resolving it
  // would take MC33 style subcase tables. What must not happen is for that to become common, so
  // the count is bounded rather than ignored.
  INFO("pinched edges: ", report.NonManifoldEdges, " of ", report.UniqueEdges);
  CHECK(report.NonManifoldEdges * 1000 < report.UniqueEdges);
}

TEST_CASE("A surface is closed at the upper volume walls but left open at the lower ones") {
  if (cudaMissing()) return;

  // The chunks own the cells based at [0, chunks * CoreSize) in every axis. At the upper end that
  // reaches past the volume and into the background padding, which caps the surface. At the lower
  // end it stops at coordinate 0: the cell between the coordinates -1 and 0 belongs to no chunk,
  // so the background below the volume never takes part and a surface reaching that wall is simply
  // cut off.
  //
  // The asymmetry is a consequence of where the chunk grid starts, not of the extraction, and this
  // test exists to state it rather than to endorse it. A caller needing a mesh closed on all six
  // sides has to pad the volume itself for now.
  Dims const dimensions = {100, 100, 100};

  SUBCASE("clipped by an upper wall") {
    auto const volume = makeVolume(dimensions, -1000.0f, ballField({98.5, 50.5, 50.5}, 30.25));
    auto const welded = ma::weldByExactPosition(extractByChunks(volume, 0.0f));
    auto const report = ma::analyzeManifold(welded);

    INFO("boundary edges: ", report.BoundaryEdges);
    CHECK(report.BoundaryEdges == 0);
    CHECK(report.IsWatertight);
  }

  SUBCASE("clipped by a lower wall") {
    auto const volume = makeVolume(dimensions, -1000.0f, ballField({1.5, 50.5, 50.5}, 30.25));
    auto const welded = ma::weldByExactPosition(extractByChunks(volume, 0.0f));
    auto const report = ma::analyzeManifold(welded);

    INFO("boundary edges: ", report.BoundaryEdges);
    CHECK(report.BoundaryEdges > 0);

    // The opening is confined to the wall it runs into: every open edge lies in the x = 0 plane.
    auto const directed = ma::directedEdgeCounts(welded);
    std::map<ma::Edge, int> undirectedCounts;
    for (auto const& [edge, count] : directed) {
      undirectedCounts[ma::undirected(edge.first, edge.second)] += count;
    }
    for (auto const& [edge, count] : undirectedCounts) {
      if (count == 2) continue;
      INFO("open edge between (", welded.Positions[edge.first][0], ", ",
           welded.Positions[edge.first][1], ", ", welded.Positions[edge.first][2], ") and (",
           welded.Positions[edge.second][0], ", ", welded.Positions[edge.second][1], ", ",
           welded.Positions[edge.second][2], ")");
      CHECK(welded.Positions[edge.first][0] == 0.0f);
      CHECK(welded.Positions[edge.second][0] == 0.0f);
    }
  }
}

TEST_CASE("A ball spanning several chunks welds into a single closed sphere") {
  if (cudaMissing()) return;

  // The same watertightness statement, but on a surface whose topology is known, so that the
  // Euler characteristic can pin the connectivity rather than only the absence of holes. The
  // centre and radius are deliberately off the voxel grid to keep corner values off the isovalue.
  Dims const dimensions = {130, 130, 130};
  auto const volume = makeVolume(dimensions, -1000.0f, ballField({65.5, 65.5, 65.5}, 40.25));

  auto const assembled = extractByChunks(volume, 0.0f);
  REQUIRE_FALSE(assembled.IsEmpty());
  REQUIRE(degenerateTriangleCount(assembled) == 0);

  auto const welded = ma::weldByExactPosition(assembled);
  auto const report = ma::analyzeManifold(welded);

  CHECK(report.BoundaryEdges == 0);
  CHECK(report.NonManifoldEdges == 0);
  CHECK(report.IsConsistentlyOriented);
  CHECK(ma::eulerCharacteristic(welded) == 2);

  constexpr double pi = 3.14159265358979323846;
  CHECK(ma::surfaceArea(welded) == doctest::Approx(4.0 * pi * 40.25 * 40.25).epsilon(0.02));
  CHECK(ma::enclosedVolume(welded)
        == doctest::Approx(4.0 / 3.0 * pi * 40.25 * 40.25 * 40.25).epsilon(0.02));
}

TEST_CASE("A surface running into the volume wall is closed off by the background padding") {
  if (cudaMissing()) return;

  // The ball is centred so that it is cut open by the volume boundary. What closes it again is the
  // background the chunkifier pads with, which the extraction sees as ordinary data. Without that,
  // the mesh would end in mid air along the wall.
  Dims const dimensions = {100, 100, 100};
  auto const volume = makeVolume(dimensions, -1000.0f, ballField({98.5, 50.5, 50.5}, 30.25));

  auto const assembled = extractByChunks(volume, 0.0f);
  REQUIRE_FALSE(assembled.IsEmpty());
  REQUIRE(degenerateTriangleCount(assembled) == 0);

  auto const welded = ma::weldByExactPosition(assembled);
  auto const report = ma::analyzeManifold(welded);

  INFO("boundary edges: ", report.BoundaryEdges);
  CHECK(report.BoundaryEdges == 0);
  CHECK(report.NonManifoldEdges == 0);
  CHECK(report.IsConsistentlyOriented);

  // A ball with a cap cut off it is still topologically a sphere.
  CHECK(ma::eulerCharacteristic(welded) == 2);

  // And it really is clipped, i.e. the test is not quietly extracting a whole ball.
  constexpr double pi = 3.14159265358979323846;
  CHECK(ma::enclosedVolume(welded) < 4.0 / 3.0 * pi * 30.25 * 30.25 * 30.25 * 0.95);
}

TEST_SUITE_END();
