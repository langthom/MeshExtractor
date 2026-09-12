
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "doctest.h"
#include "MeshAssertions.h"
#include "PLYReader.h"
#include "../app/ExtractionPipeline.h"

namespace pme = parallel_mesh_extractor;
namespace ma  = mesh_assertions;
namespace pr  = ply_reader;

namespace {

  constexpr double Pi = 3.14159265358979323846;

  bool cudaMissing() {
    if (pme::CudaChunkMeshExtractor::IsCudaAvailable()) return false;
    MESSAGE("no CUDA device available, skipping the end to end case");
    return true;
  }

  /// An MHD volume on disk, written from a field, removing itself afterwards. The header is
  /// spelled out here rather than produced by our own writer, so that the reader is tested against
  /// a file it did not author.
  class TemporaryVolume {
  public:
    template<class Field>
    TemporaryVolume(std::string const& name, std::array<std::uint32_t, 3> const& dim,
                    std::array<float, 3> const& spacing, std::array<float, 3> const& origin,
                    Field const& field, std::size_t headerBytes = 0)
      : MHDPath(std::filesystem::temp_directory_path() / (name + ".mhd"))
      , RawPath(std::filesystem::temp_directory_path() / (name + ".raw"))
    {
      std::ofstream raw(this->RawPath, std::ios::binary);
      REQUIRE(raw.good());

      std::vector<char> const filler(headerBytes, '\0');
      if (headerBytes > 0) raw.write(filler.data(), static_cast<std::streamsize>(headerBytes));

      std::vector<std::uint16_t> slice(static_cast<std::size_t>(dim[0]) * dim[1]);
      for (std::int64_t z = 0; z < dim[2]; ++z) {
        std::size_t index = 0;
        for (std::int64_t y = 0; y < dim[1]; ++y) {
          for (std::int64_t x = 0; x < dim[0]; ++x) slice[index++] = field(x, y, z);
        }
        raw.write(reinterpret_cast<char const*>(slice.data()),
                  static_cast<std::streamsize>(slice.size() * sizeof(std::uint16_t)));
      }
      raw.close();

      std::ofstream mhd(this->MHDPath);
      REQUIRE(mhd.good());
      mhd << "ObjectType      = Image\n"
          << "NDims           = 3\n"
          << "Offset          = " << origin[0] << " " << origin[1] << " " << origin[2] << "\n";
      if (headerBytes > 0) mhd << "HeaderSize      = " << headerBytes << "\n";
      mhd << "ElementSpacing  = " << spacing[0] << " " << spacing[1] << " " << spacing[2] << "\n"
          << "DimSize         = " << dim[0] << " " << dim[1] << " " << dim[2] << "\n"
          << "ElementType     = MET_USHORT\n"
          << "ElementDataFile = " << this->RawPath.string() << "\n";
    }

    ~TemporaryVolume() {
      std::error_code ignored;
      std::filesystem::remove(this->MHDPath, ignored);
      std::filesystem::remove(this->RawPath, ignored);
    }

    std::filesystem::path const& Path() const { return this->MHDPath; }

  private:
    std::filesystem::path MHDPath, RawPath;
  };

  class TemporaryOutput {
  public:
    explicit TemporaryOutput(std::string const& name)
      : OutputPath(std::filesystem::temp_directory_path() / name) {}

    ~TemporaryOutput() {
      std::error_code ignored;
      std::filesystem::remove(this->OutputPath, ignored);
      std::filesystem::remove(std::filesystem::path(this->OutputPath.string() + ".faces.tmp"), ignored);
    }

    std::filesystem::path const& Path() const { return this->OutputPath; }

  private:
    std::filesystem::path OutputPath;
  };

  double distanceTo(std::array<double, 3> const& center,
                    std::int64_t x, std::int64_t y, std::int64_t z) {
    double const dx = static_cast<double>(x) - center[0];
    double const dy = static_cast<double>(y) - center[1];
    double const dz = static_cast<double>(z) - center[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  /// A dense ball: 3000 inside, 0 outside, with the isovalue halfway up the step.
  ///
  /// Good enough wherever only the structure of the result matters. It is deliberately *not* used
  /// where the geometry is measured: a field that jumps between two values gives the interpolation
  /// nothing to work with, so every vertex lands on a voxel midpoint and the surface comes out as
  /// a staircase. Such a surface encloses about the right volume but has noticeably more area than
  /// the sphere it approximates.
  auto ballField(std::array<double, 3> const& center, double radius) {
    return [center, radius](std::int64_t x, std::int64_t y, std::int64_t z) -> std::uint16_t {
      return (distanceTo(center, x, y, z) < radius) ? std::uint16_t{3000} : std::uint16_t{0};
    };
  }

  /// The same ball, but with the density falling off across the surface instead of jumping, which
  /// is both what real scan data looks like and what lets the interpolation place a vertex to well
  /// under a voxel. Use this wherever an analytic area or volume is being checked.
  ///
  /// Values change by 300 per voxel near the surface, so quantizing them to integers costs about a
  /// three-hundredth of a voxel. The isovalue sits on a half integer, which no sample can equal,
  /// keeping the result free of the degenerate triangles an exact hit would produce.
  constexpr float SmoothBallIso = 1500.5f;

  auto smoothBallField(std::array<double, 3> const& center, double radius) {
    return [center, radius](std::int64_t x, std::int64_t y, std::int64_t z) -> std::uint16_t {
      double const value = 1500.0 + 300.0 * (radius - distanceTo(center, x, y, z));
      return static_cast<std::uint16_t>(std::clamp(value, 0.0, 65535.0));
    };
  }

  /// Read a written PLY back as a mesh, ready for the geometric assertions.
  pme::ChunkMesh readMesh(std::filesystem::path const& path) {
    auto const parsed = pr::parsePLY(path);
    pme::ChunkMesh mesh;
    mesh.Positions = parsed.Positions;
    mesh.Indices = parsed.Indices;
    return mesh;
  }

  std::string readAllBytes(std::filesystem::path const& path) {
    std::ifstream file(path, std::ios::binary);
    return std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  }

  std::array<std::array<float, 2>, 3> boundingBox(pme::ChunkMesh const& mesh) {
    std::array<std::array<float, 2>, 3> box{};
    for (int axis = 0; axis < 3; ++axis) box[axis] = {mesh.Positions.at(0)[axis], mesh.Positions.at(0)[axis]};

    for (auto const& position : mesh.Positions) {
      for (int axis = 0; axis < 3; ++axis) {
        box[axis][0] = std::min(box[axis][0], position[axis]);
        box[axis][1] = std::max(box[axis][1], position[axis]);
      }
    }
    return box;
  }

} // namespace

TEST_SUITE_BEGIN("gpu");

TEST_CASE("An MHD volume becomes a closed PLY mesh of the right size") {
  if (cudaMissing()) return;

  // The ball is comfortably inside the volume on every side, so nothing here depends on how the
  // volume boundary is handled, and the surface is a plain sphere.
  constexpr double radius = 40.0;
  std::array<std::uint32_t, 3> const dim = {130, 130, 130};

  TemporaryVolume const volume("pme_e2e_ball", dim, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                               smoothBallField({64.5, 64.5, 64.5}, radius));
  TemporaryOutput const output("pme_e2e_ball.ply");

  pme::ExtractionSettings settings;
  settings.IsoThreshold = SmoothBallIso;
  settings.BackgroundValue = 0.0f;

  auto const report = pme::ExtractVolumeToPLY(volume.Path(), output.Path(), settings);

  CHECK(report.Dimensions == dim);
  CHECK(report.Triangles > 0);
  CHECK(report.Slabs == 3);

  auto const mesh = readMesh(output.Path());
  REQUIRE(mesh.Positions.size() == report.Vertices);
  REQUIRE(mesh.TriangleCount() == report.Triangles);

  // Welded across chunks by the assembler, so the file already carries the connectivity: reading
  // it back and finding a closed surface means the streaming never dropped or duplicated a vertex.
  auto const manifold = ma::analyzeManifold(mesh);
  INFO("boundary edges: ", manifold.BoundaryEdges, ", non manifold edges: ", manifold.NonManifoldEdges);
  CHECK(manifold.BoundaryEdges == 0);
  CHECK(manifold.IsWatertight);
  CHECK(manifold.IsConsistentlyOriented);
  CHECK(ma::eulerCharacteristic(mesh) == 2);

  // With the density falling off smoothly across the surface, the interpolation places each vertex
  // well within a voxel of the true sphere, so both quantities come out to well under a percent.
  // Every vertex here has been through the reader, the chunkifier, the GPU, the welding, the file
  // and back, so these two numbers stand in for the whole chain being right.
  CHECK(ma::surfaceArea(mesh) == doctest::Approx(4.0 * Pi * radius * radius).epsilon(0.01));
  CHECK(ma::enclosedVolume(mesh)
        == doctest::Approx(4.0 / 3.0 * Pi * radius * radius * radius).epsilon(0.01));

  // Every vertex sits on the sphere, not merely in the right neighbourhood.
  double worstRadialError = 0.0;
  for (auto const& position : mesh.Positions) {
    ma::Vertex const offset = {position[0] - 64.5f, position[1] - 64.5f, position[2] - 64.5f};
    worstRadialError = std::max(worstRadialError, std::abs(ma::length(offset) - radius));
  }
  INFO("largest radial deviation: ", worstRadialError, " voxels");
  CHECK(worstRadialError < 0.1);
}

TEST_CASE("The slab depth changes nothing but memory") {
  if (cudaMissing()) return;

  // The point of the whole streaming arrangement. Reading one chunk layer at a time and reading
  // the entire volume at once have to produce the very same bytes; if welding or pruning were off
  // by anything at all, the files would differ.
  std::array<std::uint32_t, 3> const dim = {100, 100, 200};

  TemporaryVolume const volume("pme_e2e_slabs", dim, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                               ballField({49.5, 49.5, 99.5}, 40.0));

  pme::ExtractionSettings settings;
  settings.IsoThreshold = 1500.0f;

  std::vector<std::string> written;
  std::vector<std::uint64_t> slabCounts;

  for (std::uint32_t layersPerSlab : {1u, 2u, 3u, 1000u}) {
    TemporaryOutput const output("pme_e2e_slabs_" + std::to_string(layersPerSlab) + ".ply");

    settings.LayersPerSlab = layersPerSlab;
    auto const report = pme::ExtractVolumeToPLY(volume.Path(), output.Path(), settings);

    INFO(layersPerSlab, " layers per slab");
    CHECK(report.Triangles > 0);

    written.push_back(readAllBytes(output.Path()));
    slabCounts.push_back(report.Slabs);
  }

  // The plans really are different, so the comparison below is not comparing a thing with itself.
  CHECK(slabCounts.front() > slabCounts.back());
  CHECK(slabCounts.back() == 1);

  for (std::size_t i = 1; i < written.size(); ++i) {
    INFO("plan ", i, " against plan 0");
    CHECK(written[i].size() == written[0].size());
    CHECK(written[i] == written[0]);
  }
}

TEST_CASE("The spacing and origin place the mesh in world coordinates") {
  if (cudaMissing()) return;

  // "Offset" is the world position of the first voxel, in the same unit as the spacing, so a
  // vertex ends up at origin + voxel * spacing. Extracting the same volume both ways and comparing
  // states exactly that, without having to know where the surface actually runs.
  std::array<std::uint32_t, 3> const dim = {80, 80, 80};
  std::array<float, 3> const spacing = {0.5f, 0.25f, 2.0f};
  std::array<float, 3> const origin  = {-10.0f, 5.0f, 2.5f};

  TemporaryVolume const volume("pme_e2e_world", dim, spacing, origin,
                               ballField({39.5, 39.5, 39.5}, 25.0));

  TemporaryOutput const inVoxels("pme_e2e_voxels.ply");
  TemporaryOutput const inWorld("pme_e2e_world.ply");

  pme::ExtractionSettings settings;
  settings.IsoThreshold = 1500.0f;

  settings.VoxelCoordinates = true;
  auto const voxelReport = pme::ExtractVolumeToPLY(volume.Path(), inVoxels.Path(), settings);

  settings.VoxelCoordinates = false;
  auto const worldReport = pme::ExtractVolumeToPLY(volume.Path(), inWorld.Path(), settings);

  CHECK(voxelReport.Spacing == spacing);
  CHECK(voxelReport.Origin == origin);

  // The transform happens after the welding, so it may not change the mesh's structure at all.
  REQUIRE(worldReport.Vertices == voxelReport.Vertices);
  REQUIRE(worldReport.Triangles == voxelReport.Triangles);

  auto const voxelMesh = readMesh(inVoxels.Path());
  auto const worldMesh = readMesh(inWorld.Path());
  REQUIRE(voxelMesh.Indices == worldMesh.Indices);
  REQUIRE(voxelMesh.Positions.size() == worldMesh.Positions.size());

  for (std::size_t i = 0; i < voxelMesh.Positions.size(); ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      float const expected = origin[axis] + voxelMesh.Positions[i][axis] * spacing[axis];
      INFO("vertex ", i, ", axis ", axis);
      REQUIRE(worldMesh.Positions[i][axis] == doctest::Approx(expected).epsilon(1e-6));
    }
  }

  // And the whole thing really did move and stretch, rather than the two runs agreeing trivially.
  auto const voxelBox = boundingBox(voxelMesh);
  auto const worldBox = boundingBox(worldMesh);
  for (int axis = 0; axis < 3; ++axis) {
    INFO("axis ", axis);
    CHECK(worldBox[axis][0] == doctest::Approx(origin[axis] + voxelBox[axis][0] * spacing[axis]).epsilon(1e-5));
    CHECK(worldBox[axis][1] == doctest::Approx(origin[axis] + voxelBox[axis][1] * spacing[axis]).epsilon(1e-5));
  }
  CHECK(worldBox[0][0] < 0.0f);   // the negative origin really did move it off the grid corner
}

TEST_CASE("A raw file with a header of its own is read past correctly") {
  if (cudaMissing()) return;

  // "HeaderSize" bytes of the raw file are not voxels. Extracting with and without such a header
  // in front of identical data has to give identical meshes.
  std::array<std::uint32_t, 3> const dim = {70, 70, 70};

  TemporaryVolume const plain("pme_e2e_plain", dim, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                              ballField({34.5, 34.5, 34.5}, 22.0));
  TemporaryVolume const headed("pme_e2e_headed", dim, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                               ballField({34.5, 34.5, 34.5}, 22.0), /*headerBytes=*/2048);

  TemporaryOutput const plainOut("pme_e2e_plain.ply");
  TemporaryOutput const headedOut("pme_e2e_headed.ply");

  pme::ExtractionSettings settings;
  settings.IsoThreshold = 1500.0f;

  auto const plainReport  = pme::ExtractVolumeToPLY(plain.Path(),  plainOut.Path(),  settings);
  auto const headedReport = pme::ExtractVolumeToPLY(headed.Path(), headedOut.Path(), settings);

  CHECK(plainReport.Triangles > 0);
  CHECK(headedReport.Triangles == plainReport.Triangles);
  CHECK(readAllBytes(headedOut.Path()) == readAllBytes(plainOut.Path()));
}

TEST_CASE("A volume with no surface produces a valid, empty mesh") {
  if (cudaMissing()) return;

  // The chunkifier can cull every chunk away. That has to come out as an empty but well formed
  // PLY rather than as a failure or a truncated file.
  std::array<std::uint32_t, 3> const dim = {70, 70, 70};

  TemporaryVolume const volume("pme_e2e_empty", dim, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                               [](std::int64_t, std::int64_t, std::int64_t) { return std::uint16_t{500}; });
  TemporaryOutput const output("pme_e2e_empty.ply");

  pme::ExtractionSettings settings;
  settings.IsoThreshold = 1500.0f;
  settings.BackgroundValue = 500.0f;   // the same as the data, so the wall carries no surface either

  auto const report = pme::ExtractVolumeToPLY(volume.Path(), output.Path(), settings);

  CHECK(report.Vertices == 0);
  CHECK(report.Triangles == 0);

  auto const parsed = pr::parsePLY(output.Path());
  CHECK(parsed.DeclaredVertices == 0);
  CHECK(parsed.DeclaredFaces == 0);
}

TEST_CASE("The ASCII form describes the same mesh as the binary one") {
  if (cudaMissing()) return;

  std::array<std::uint32_t, 3> const dim = {70, 70, 70};

  TemporaryVolume const volume("pme_e2e_ascii", dim, {0.5f, 0.5f, 0.5f}, {1.5f, -2.5f, 0.25f},
                               ballField({34.5, 34.5, 34.5}, 22.0));

  TemporaryOutput const binaryOut("pme_e2e_binary.ply");
  TemporaryOutput const asciiOut("pme_e2e_ascii.ply");

  pme::ExtractionSettings settings;
  settings.IsoThreshold = 1500.0f;

  settings.Format = pme::PLYFormat::BinaryLittleEndian;
  pme::ExtractVolumeToPLY(volume.Path(), binaryOut.Path(), settings);

  settings.Format = pme::PLYFormat::ASCII;
  pme::ExtractVolumeToPLY(volume.Path(), asciiOut.Path(), settings);

  auto const binaryMesh = readMesh(binaryOut.Path());
  auto const asciiMesh  = readMesh(asciiOut.Path());

  REQUIRE(binaryMesh.Positions.size() == asciiMesh.Positions.size());
  CHECK(binaryMesh.Indices == asciiMesh.Indices);
  for (std::size_t i = 0; i < binaryMesh.Positions.size(); ++i) {
    INFO("vertex ", i);
    CHECK(ma::positionKey(binaryMesh.Positions[i]) == ma::positionKey(asciiMesh.Positions[i]));
  }
}

TEST_SUITE_END();
