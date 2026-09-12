
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "doctest.h"
#include "MeshAssertions.h"
#include "PLYReader.h"
#include "../MeshIO/PLYStreamWriter.h"

namespace pme = parallel_mesh_extractor;
namespace ma  = mesh_assertions;
namespace pr  = ply_reader;

namespace {

  /// A scratch file that removes itself, so a failing assertion cannot leave litter behind.
  class TemporaryPath {
  public:
    explicit TemporaryPath(std::string const& name)
      : Path(std::filesystem::temp_directory_path() / name) {
      std::error_code ignored;
      std::filesystem::remove(this->Path, ignored);
    }

    ~TemporaryPath() {
      std::error_code ignored;
      std::filesystem::remove(this->Path, ignored);
      std::filesystem::remove(std::filesystem::path(this->Path.string() + ".faces.tmp"), ignored);
    }

    std::filesystem::path const& Get() const { return this->Path; }
    std::filesystem::path FaceTemporary() const { return this->Path.string() + ".faces.tmp"; }

  private:
    std::filesystem::path Path;
  };

  /// A mesh with values that are awkward on purpose: negatives, fractions, and a coordinate whose
  /// decimal form is long enough to catch a writer that rounds.
  pme::ChunkMesh awkwardMesh() {
    pme::ChunkMesh mesh;
    mesh.Positions = {
      { 0.0f,  0.0f,  0.0f},
      { 1.0f,  0.0f,  0.0f},
      { 0.0f,  1.0f,  0.0f},
      {-12.3456789f, 0.1234567f, 1234.5678f},
      {-0.0f, 3.4028235e+38f, 1.1754944e-38f},
    };
    mesh.Indices = {0, 1, 2, 1, 3, 2, 2, 3, 4};
    return mesh;
  }

} // namespace

// ------------------------------------- writing and reading ----------------------------------- //

TEST_CASE("A mesh survives a round trip through the writer") {
  auto const mesh = awkwardMesh();

  for (auto const format : {pme::PLYFormat::BinaryLittleEndian, pme::PLYFormat::ASCII}) {
    INFO("format: ", (format == pme::PLYFormat::BinaryLittleEndian ? "binary" : "ascii"));
    TemporaryPath const output("pme_roundtrip.ply");

    {
      pme::PLYStreamWriter writer(output.Get(), format);
      writer.WriteMesh(mesh);
      CHECK(writer.VertexCount() == mesh.Positions.size());
      CHECK(writer.TriangleCount() == mesh.TriangleCount());
      writer.Finish();
    }

    auto const parsed = pr::parsePLY(output.Get());

    CHECK(parsed.DeclaredVertices == mesh.Positions.size());
    CHECK(parsed.DeclaredFaces == mesh.TriangleCount());
    CHECK(parsed.Indices == mesh.Indices);

    // Bit for bit, including the negative zero: nine significant digits is exactly what a float
    // needs to survive the trip through decimal.
    REQUIRE(parsed.Positions.size() == mesh.Positions.size());
    for (std::size_t i = 0; i < mesh.Positions.size(); ++i) {
      INFO("vertex ", i);
      CHECK(ma::positionKey(parsed.Positions[i]) == ma::positionKey(mesh.Positions[i]));
    }
  }
}

TEST_CASE("Streaming in many small batches gives the same file as writing at once") {
  // The streaming pipeline hands over one chunk at a time, so the batching must not be able to
  // show up in the result.
  auto const mesh = awkwardMesh();

  TemporaryPath const atOnce("pme_at_once.ply");
  TemporaryPath const batched("pme_batched.ply");

  {
    pme::PLYStreamWriter writer(atOnce.Get(), pme::PLYFormat::BinaryLittleEndian);
    writer.WriteMesh(mesh);
    writer.Finish();
  }

  {
    pme::PLYStreamWriter writer(batched.Get(), pme::PLYFormat::BinaryLittleEndian);
    // One vertex and, once enough vertices exist, one triangle at a time.
    for (std::size_t i = 0; i < mesh.Positions.size(); ++i) {
      writer.WriteVertices({mesh.Positions[i]});
    }
    for (std::size_t i = 0; i < mesh.Indices.size(); i += 3) {
      writer.WriteFaces({mesh.Indices[i], mesh.Indices[i + 1], mesh.Indices[i + 2]});
    }
    writer.Finish();
  }

  std::ifstream a(atOnce.Get(), std::ios::binary), b(batched.Get(), std::ios::binary);
  std::string const contentA{std::istreambuf_iterator<char>(a), std::istreambuf_iterator<char>()};
  std::string const contentB{std::istreambuf_iterator<char>(b), std::istreambuf_iterator<char>()};
  CHECK(contentA == contentB);
}

// ---------------------------------------- the header ----------------------------------------- //

TEST_CASE("Patching the counts leaves the header the same length") {
  // The counts are patched in place, so the header written up front and the header left behind
  // have to occupy exactly the same bytes. If they did not, the patch would overwrite the data.
  TemporaryPath const empty("pme_header_empty.ply");
  TemporaryPath const full("pme_header_full.ply");

  {
    pme::PLYStreamWriter writer(empty.Get(), pme::PLYFormat::BinaryLittleEndian);
    writer.Finish();
  }
  {
    pme::PLYStreamWriter writer(full.Get(), pme::PLYFormat::BinaryLittleEndian);
    writer.WriteMesh(awkwardMesh());
    writer.Finish();
  }

  auto const parsedEmpty = pr::parsePLY(empty.Get());
  auto const parsedFull  = pr::parsePLY(full.Get());

  CHECK(parsedEmpty.HeaderBytes == parsedFull.HeaderBytes);
  CHECK(parsedEmpty.DeclaredVertices == 0);
  CHECK(parsedEmpty.DeclaredFaces == 0);
}

TEST_CASE("The binary body is exactly as long as the element declarations say") {
  // Twelve bytes a vertex, thirteen a face, and no padding anywhere: this is what a reader
  // stepping through the file by size relies on.
  auto const mesh = awkwardMesh();
  TemporaryPath const output("pme_sizes.ply");

  {
    pme::PLYStreamWriter writer(output.Get(), pme::PLYFormat::BinaryLittleEndian);
    writer.WriteMesh(mesh);
    writer.Finish();
  }

  auto const parsed = pr::parsePLY(output.Get());
  auto const total = std::filesystem::file_size(output.Get());

  auto const expected = parsed.HeaderBytes
                      + mesh.Positions.size() * 12
                      + mesh.TriangleCount() * 13;
  CHECK(total == expected);
}

TEST_CASE("An empty mesh produces a valid, empty PLY") {
  // The chunkifier can cull an entire volume away, so producing nothing is a normal outcome and
  // has to result in a file a reader accepts rather than in a truncated one.
  TemporaryPath const output("pme_empty.ply");

  {
    pme::PLYStreamWriter writer(output.Get(), pme::PLYFormat::BinaryLittleEndian);
    writer.Finish();
  }

  auto const parsed = pr::parsePLY(output.Get());
  CHECK(parsed.DeclaredVertices == 0);
  CHECK(parsed.DeclaredFaces == 0);
  CHECK(parsed.Positions.empty());
  CHECK(parsed.Indices.empty());
  CHECK(std::filesystem::file_size(output.Get()) == parsed.HeaderBytes);
}

// ---------------------------------- the temporary face file ---------------------------------- //

TEST_CASE("The temporary face file lives next to the output and does not outlive the writer") {
  // Next to the output rather than in the system temporary directory, so that a spill and its
  // final append never cross a filesystem and cannot run into a small /tmp.
  //
  // A one byte face buffer forces the spill, which is the only way the file comes into existence
  // at all now that faces are held in memory by default.
  TemporaryPath const output("pme_temporary.ply");
  constexpr std::size_t forceSpill = 1;

  SUBCASE("created on spilling, removed after finishing") {
    {
      pme::PLYStreamWriter writer(output.Get(), pme::PLYFormat::BinaryLittleEndian, forceSpill);
      CHECK_FALSE(std::filesystem::exists(output.FaceTemporary()));

      writer.WriteMesh(awkwardMesh());
      CHECK(std::filesystem::exists(output.FaceTemporary()));

      writer.Finish();
    }
    CHECK_FALSE(std::filesystem::exists(output.FaceTemporary()));
  }

  SUBCASE("removed even when the writer is abandoned") {
    {
      pme::PLYStreamWriter writer(output.Get(), pme::PLYFormat::BinaryLittleEndian, forceSpill);
      writer.WriteMesh(awkwardMesh());
      REQUIRE(std::filesystem::exists(output.FaceTemporary()));
      // No Finish: the output is left incomplete on purpose, but the scratch file is ours.
    }
    CHECK_FALSE(std::filesystem::exists(output.FaceTemporary()));
  }
}

// ------------------------------------------ failures ----------------------------------------- //

TEST_CASE("A face referencing a vertex that was never written is rejected") {
  // All vertices precede all faces in the file, so a forward reference cannot be honoured. Failing
  // here beats writing a file that only falls over when something reads it.
  TemporaryPath const output("pme_forward_reference.ply");
  pme::PLYStreamWriter writer(output.Get(), pme::PLYFormat::BinaryLittleEndian);

  writer.WriteVertices({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
  CHECK_NOTHROW(writer.WriteFaces({0, 1, 2}));
  CHECK_THROWS_AS(writer.WriteFaces({0, 1, 3}), std::runtime_error);
}

TEST_CASE("Face indices have to come in triples") {
  TemporaryPath const output("pme_partial_triangle.ply");
  pme::PLYStreamWriter writer(output.Get(), pme::PLYFormat::BinaryLittleEndian);

  writer.WriteVertices({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
  CHECK_THROWS_AS(writer.WriteFaces({0, 1}), std::runtime_error);
  CHECK_THROWS_AS(writer.WriteFaces({0, 1, 2, 0}), std::runtime_error);
}

TEST_CASE("An unwritable output path is reported rather than ignored") {
  CHECK_THROWS_AS(
    pme::PLYStreamWriter(std::filesystem::path("/definitely/not/a/directory/mesh.ply"),
                         pme::PLYFormat::BinaryLittleEndian),
    std::runtime_error);
}

TEST_CASE("Writing after finishing is refused") {
  TemporaryPath const output("pme_after_finish.ply");
  pme::PLYStreamWriter writer(output.Get(), pme::PLYFormat::BinaryLittleEndian);

  writer.WriteVertices({{0, 0, 0}});
  writer.Finish();

  CHECK_THROWS_AS(writer.WriteVertices({{1, 1, 1}}), std::runtime_error);
  CHECK_THROWS_AS(writer.WriteFaces({0, 0, 0}), std::runtime_error);

  // Finishing twice is harmless, so that a caller need not track whether it already did.
  CHECK_NOTHROW(writer.Finish());
}

// ----------------------------------- the in memory face buffer ------------------------------- //

TEST_CASE("Buffering faces in memory and spilling them produce the same file") {
  // Spilling exists only so that a mesh larger than memory still works. Which path was taken must
  // not be visible in the result, so the two are compared byte for byte.
  auto const mesh = awkwardMesh();

  TemporaryPath const buffered("pme_buffered.ply");
  TemporaryPath const spilled("pme_spilled.ply");

  {
    pme::PLYStreamWriter writer(buffered.Get(), pme::PLYFormat::BinaryLittleEndian,
                                /*faceBufferBytes=*/1 << 20);
    writer.WriteMesh(mesh);
    writer.Finish();
    CHECK_FALSE(writer.SpilledToDisk());
  }
  {
    // One byte of buffer, so every batch of faces goes straight to the temporary file.
    pme::PLYStreamWriter writer(spilled.Get(), pme::PLYFormat::BinaryLittleEndian,
                                /*faceBufferBytes=*/1);
    writer.WriteMesh(mesh);
    writer.Finish();
    CHECK(writer.SpilledToDisk());
  }

  std::ifstream a(buffered.Get(), std::ios::binary), b(spilled.Get(), std::ios::binary);
  std::string const contentA{std::istreambuf_iterator<char>(a), std::istreambuf_iterator<char>()};
  std::string const contentB{std::istreambuf_iterator<char>(b), std::istreambuf_iterator<char>()};
  CHECK(contentA == contentB);
}

TEST_CASE("No temporary file is created when the faces fit in memory") {
  // The common case, and the one worth protecting: a run that never spills should leave nothing
  // behind and should never touch the disk twice for the same bytes.
  TemporaryPath const output("pme_nospill.ply");

  pme::PLYStreamWriter writer(output.Get(), pme::PLYFormat::BinaryLittleEndian);
  CHECK_FALSE(std::filesystem::exists(output.FaceTemporary()));

  writer.WriteMesh(awkwardMesh());
  CHECK_FALSE(std::filesystem::exists(output.FaceTemporary()));

  writer.Finish();
  CHECK_FALSE(writer.SpilledToDisk());
  CHECK_FALSE(std::filesystem::exists(output.FaceTemporary()));
}

TEST_CASE("Spilling across many small batches keeps the faces in order") {
  // The spill path appends batch by batch, so an ordering slip would only show up once more than
  // one spill has happened.
  auto const mesh = awkwardMesh();

  TemporaryPath const atOnce("pme_spill_once.ply");
  TemporaryPath const batched("pme_spill_batched.ply");

  {
    pme::PLYStreamWriter writer(atOnce.Get(), pme::PLYFormat::BinaryLittleEndian);
    writer.WriteMesh(mesh);
    writer.Finish();
  }
  {
    pme::PLYStreamWriter writer(batched.Get(), pme::PLYFormat::BinaryLittleEndian, 1);
    writer.WriteVertices(mesh.Positions);
    for (std::size_t i = 0; i < mesh.Indices.size(); i += 3) {
      writer.WriteFaces({mesh.Indices[i], mesh.Indices[i + 1], mesh.Indices[i + 2]});
    }
    writer.Finish();
    CHECK(writer.SpilledToDisk());
  }

  CHECK(pr::parsePLY(batched.Get()).Indices == pr::parsePLY(atOnce.Get()).Indices);
}
