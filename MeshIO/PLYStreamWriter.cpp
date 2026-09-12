#include "PLYStreamWriter.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pme = parallel_mesh_extractor;

namespace {

  // The binary form writes the in-memory bytes of the coordinates straight out, which is only the
  // little endian layout the header promises on a little endian host.
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                "the binary PLY path writes native bytes and assumes a little endian host");
  static_assert(sizeof(std::array<float, 3>) == 3 * sizeof(float),
                "a vertex has to be three tightly packed floats for the bulk write below");
  static_assert(sizeof(float) == 4 && sizeof(std::uint32_t) == 4,
                "the PLY element sizes the header declares assume 32 bit floats and indices");

  /// A count rendered into a field of fixed width, padded with trailing spaces. Keeping the width
  /// constant is what lets the header be patched in place once the totals are known.
  std::string countField(std::uint64_t value) {
    std::string field = std::to_string(value);
    if (field.size() > pme::PLYStreamWriter::CountFieldWidth) {
      throw std::runtime_error("PLY element count does not fit the reserved header field");
    }
    field.resize(pme::PLYStreamWriter::CountFieldWidth, ' ');
    return field;
  }

  void appendBytes(std::vector<char>& target, void const* data, std::size_t size) {
    auto const* bytes = static_cast<char const*>(data);
    target.insert(target.end(), bytes, bytes + size);
  }

  void appendText(std::vector<char>& target, char const* text, std::size_t size) {
    target.insert(target.end(), text, text + size);
  }

} // namespace

pme::PLYStreamWriter::PLYStreamWriter(std::filesystem::path const& output, PLYFormat format)
  : OutputPath(output)
  , FaceTemporaryPath(output.string() + ".faces.tmp")
  , Format(format)
{
  this->Output.open(this->OutputPath, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!this->Output) {
    throw std::runtime_error("cannot open the PLY output file: " + this->OutputPath.string());
  }

  this->FaceTemporary.open(this->FaceTemporaryPath, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!this->FaceTemporary) {
    throw std::runtime_error("cannot open the temporary face file: " + this->FaceTemporaryPath.string());
  }

  this->WriteHeader();
}

pme::PLYStreamWriter::~PLYStreamWriter() {
  // A writer destroyed without finishing leaves the output incomplete, which is the caller's
  // business, but the temporary file is ours and must not be left behind. Nothing here may throw.
  if (this->Finished) return;

  std::error_code ignored;
  this->FaceTemporary.close();
  std::filesystem::remove(this->FaceTemporaryPath, ignored);
}

void pme::PLYStreamWriter::WriteHeader() {
  std::string header;
  header += "ply\n";
  header += (this->Format == PLYFormat::BinaryLittleEndian) ? "format binary_little_endian 1.0\n"
                                                            : "format ascii 1.0\n";
  header += "comment written by ParallelMeshExtractor\n";

  header += "element vertex ";
  this->VertexCountOffset = static_cast<std::streamoff>(header.size());
  header += countField(0);
  header += "\n";
  header += "property float x\n";
  header += "property float y\n";
  header += "property float z\n";

  header += "element face ";
  this->FaceCountOffset = static_cast<std::streamoff>(header.size());
  header += countField(0);
  header += "\n";
  header += "property list uchar uint vertex_indices\n";
  header += "end_header\n";

  this->Output.write(header.data(), static_cast<std::streamsize>(header.size()));
  if (!this->Output) throw std::runtime_error("writing the PLY header failed");
}

void pme::PLYStreamWriter::FlushToOutput(std::vector<char> const& bytes) {
  if (bytes.empty()) return;
  this->Output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!this->Output) throw std::runtime_error("writing to the PLY output file failed");
}

void pme::PLYStreamWriter::WriteVertices(std::vector<std::array<float, 3>> const& vertices) {
  if (this->Finished) throw std::runtime_error("the PLY writer has already been finished");
  if (vertices.empty()) return;

  if (this->Format == PLYFormat::BinaryLittleEndian) {
    // Three tightly packed floats per vertex is exactly the file layout, so the whole batch goes
    // out in one write with no per-vertex work at all.
    this->Output.write(reinterpret_cast<char const*>(vertices.data()),
                       static_cast<std::streamsize>(vertices.size() * sizeof(std::array<float, 3>)));
    if (!this->Output) throw std::runtime_error("writing PLY vertices failed");
  } else {
    this->Scratch.clear();
    char line[128];
    for (auto const& vertex : vertices) {
      // Nine significant digits is what it takes for a float to survive the round trip through
      // decimal, which keeps the ASCII form as exact as the binary one.
      int const length = std::snprintf(line, sizeof(line), "%.9g %.9g %.9g\n",
                                       static_cast<double>(vertex[0]),
                                       static_cast<double>(vertex[1]),
                                       static_cast<double>(vertex[2]));
      if (length <= 0) throw std::runtime_error("formatting a PLY vertex failed");
      appendText(this->Scratch, line, static_cast<std::size_t>(length));
    }
    this->FlushToOutput(this->Scratch);
  }

  this->Vertices += vertices.size();
}

void pme::PLYStreamWriter::WriteFaces(std::vector<std::uint32_t> const& indices) {
  if (this->Finished) throw std::runtime_error("the PLY writer has already been finished");
  if (indices.empty()) return;

  if (indices.size() % 3 != 0) {
    throw std::runtime_error("PLY face indices have to come in triples");
  }

  // All vertices precede all faces in the file, so an index may only address a vertex already
  // written. Catching it here turns a silently corrupt file into an immediate failure.
  for (auto const index : indices) {
    if (index >= this->Vertices) {
      throw std::runtime_error("a PLY face references a vertex that has not been written");
    }
  }

  this->Scratch.clear();

  if (this->Format == PLYFormat::BinaryLittleEndian) {
    this->Scratch.reserve(indices.size() / 3 * (1 + 3 * sizeof(std::uint32_t)));
    std::uint8_t const cornerCount = 3;
    for (std::size_t i = 0; i < indices.size(); i += 3) {
      appendBytes(this->Scratch, &cornerCount, sizeof(cornerCount));
      appendBytes(this->Scratch, &indices[i], 3 * sizeof(std::uint32_t));
    }
  } else {
    char line[64];
    for (std::size_t i = 0; i < indices.size(); i += 3) {
      int const length = std::snprintf(line, sizeof(line), "3 %u %u %u\n",
                                       indices[i], indices[i + 1], indices[i + 2]);
      if (length <= 0) throw std::runtime_error("formatting a PLY face failed");
      appendText(this->Scratch, line, static_cast<std::size_t>(length));
    }
  }

  this->FaceTemporary.write(this->Scratch.data(), static_cast<std::streamsize>(this->Scratch.size()));
  if (!this->FaceTemporary) throw std::runtime_error("writing to the temporary face file failed");

  this->Triangles += indices.size() / 3;
}

void pme::PLYStreamWriter::WriteMesh(ChunkMesh const& mesh) {
  this->WriteVertices(mesh.Positions);
  this->WriteFaces(mesh.Indices);
}

void pme::PLYStreamWriter::AppendFaceTemporary() {
  this->FaceTemporary.flush();
  if (!this->FaceTemporary) throw std::runtime_error("flushing the temporary face file failed");
  this->FaceTemporary.close();

  std::ifstream faces(this->FaceTemporaryPath, std::ios::binary);
  if (!faces) throw std::runtime_error("cannot reopen the temporary face file");

  // Block-wise rather than character-wise: the face data is the larger half of a big mesh.
  std::vector<char> block(1 << 20);
  while (faces) {
    faces.read(block.data(), static_cast<std::streamsize>(block.size()));
    auto const got = faces.gcount();
    if (got <= 0) break;

    this->Output.write(block.data(), got);
    if (!this->Output) throw std::runtime_error("appending the PLY faces failed");
  }

  if (faces.bad()) throw std::runtime_error("reading the temporary face file failed");
}

void pme::PLYStreamWriter::PatchCounts() {
  auto const patch = [this](std::streamoff offset, std::uint64_t value) {
    auto const field = countField(value);
    this->Output.seekp(offset, std::ios::beg);
    if (!this->Output) throw std::runtime_error("seeking back to the PLY header failed");

    this->Output.write(field.data(), static_cast<std::streamsize>(field.size()));
    if (!this->Output) throw std::runtime_error("patching the PLY header counts failed");
  };

  patch(this->VertexCountOffset, this->Vertices);
  patch(this->FaceCountOffset, this->Triangles);
}

void pme::PLYStreamWriter::Finish() {
  if (this->Finished) return;

  this->AppendFaceTemporary();
  this->PatchCounts();

  this->Output.flush();
  if (!this->Output) throw std::runtime_error("flushing the PLY output file failed");
  this->Output.close();

  std::error_code ignored;
  std::filesystem::remove(this->FaceTemporaryPath, ignored);

  this->Finished = true;
}
