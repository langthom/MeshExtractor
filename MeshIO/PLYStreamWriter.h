#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "../MeshExtraction/ChunkMesh.h"

namespace parallel_mesh_extractor {

  enum class PLYFormat {
    /// Compact and exact: the vertex coordinates keep their bit patterns. The natural choice for
    /// anything beyond a few thousand triangles.
    BinaryLittleEndian,

    /// Human readable, and considerably larger. Useful for looking at small results by eye.
    ASCII,
  };

  /// Writes a PLY file without ever holding the mesh.
  ///
  /// PLY puts the vertex and face counts in its header, which a streaming writer does not know
  /// until it is done. The way out taken here is to write the header up front with both counts as
  /// fixed width fields padded with spaces, and to seek back and overwrite exactly those fields at
  /// the end. PLY readers split the header on whitespace, so the padding is invisible to them and
  /// the header's byte length never changes.
  ///
  /// That still leaves the ordering problem, since all vertices precede all faces in the file and
  /// their total size is unknown while writing. Vertices therefore go straight into the output,
  /// while faces are held back until the vertices are done.
  ///
  /// Faces are held in memory up to "faceBufferBytes" and only spill to a temporary file beyond
  /// it. That matters more than it sounds: spilling turns every face byte into three -- written to
  /// the temporary, read back, written again -- and interleaves those writes with the vertex
  /// writes, which on a rotating disk costs more than the extra bytes do. Measured on a 7200 rpm
  /// drive, two files written alternately run at 85 MB/s against 190 MB/s for one sequential
  /// stream. A few hundred megabytes of buffer avoids all of it; the spill remains for meshes that
  /// genuinely do not fit.
  ///
  /// The temporary file is placed *next to* the output, so that a spill and its final append never
  /// cross a filesystem.
  ///
  /// Everything is reported by throwing, including a failed write, so that a truncated file cannot
  /// pass for a complete one.
  class PLYStreamWriter {
  public:
    /// Faces held in memory before spilling to disk. Large enough that ordinary meshes never
    /// spill, small enough to be an unremarkable allocation.
    static constexpr std::size_t DefaultFaceBufferBytes = 1024ull * 1024ull * 1024ull;

    PLYStreamWriter(std::filesystem::path const& output, PLYFormat format,
                    std::size_t faceBufferBytes = DefaultFaceBufferBytes);
    ~PLYStreamWriter();

    PLYStreamWriter(PLYStreamWriter const&) = delete;
    PLYStreamWriter& operator=(PLYStreamWriter const&) = delete;

    /// Appends vertices. Their indices run in the order written, starting at zero.
    void WriteVertices(std::vector<std::array<float, 3>> const& vertices);

    /// Appends triangles, three vertex indices each.
    ///
    /// Every index must address a vertex already written. A forward reference would produce a file
    /// that only fails when something tries to read it, so it is rejected here instead.
    void WriteFaces(std::vector<std::uint32_t> const& indices);

    /// Convenience for a mesh that is already complete in memory.
    void WriteMesh(ChunkMesh const& mesh);

    /// Appends the buffered faces, patches the counts into the header and removes the temporary
    /// file. Doing nothing at all and then finishing produces a valid, empty PLY.
    void Finish();

    std::uint64_t VertexCount() const { return this->Vertices; }
    std::uint64_t TriangleCount() const { return this->Triangles; }

    /// Whether the face buffer overflowed and a temporary file had to be opened. Of no interest to
    /// a caller, but it is what a test checks to know which path it exercised.
    bool SpilledToDisk() const { return this->Spilled; }

    /// Width of each count field in the header. Exposed so that a test can state what it is
    /// checking rather than rediscovering it.
    static constexpr std::size_t CountFieldWidth = 20;

  private:
    void WriteHeader();
    void PatchCounts();
    void AppendFaces();
    void SpillFaceBuffer();
    void FlushToOutput(std::vector<char> const& bytes);

    std::filesystem::path OutputPath;
    std::filesystem::path FaceTemporaryPath;
    PLYFormat Format;

    std::ofstream Output;

    /// Faces accumulate here, and only reach FaceTemporary once they no longer fit.
    std::vector<char> FaceBuffer;
    std::size_t FaceBufferLimit;
    bool Spilled = false;
    std::ofstream FaceTemporary;

    std::uint64_t Vertices = 0;
    std::uint64_t Triangles = 0;

    std::streamoff VertexCountOffset = 0;
    std::streamoff FaceCountOffset = 0;

    bool Finished = false;

    /// Reused across calls, so that streaming a few thousand chunks does not reallocate per chunk.
    std::vector<char> Scratch;
  };

} // namespace parallel_mesh_extractor
