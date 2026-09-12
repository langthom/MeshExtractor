#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "../MeshExtraction/ChunkMesh.h"
#include "../MeshExtraction/Chunkifier.h"
#include "../MeshExtraction/CudaChunkMeshExtractor.h"
#include "../MeshExtraction/MeshAssembler.h"
#include "../MeshExtraction/SlabSchedule.h"
#include "../MeshIO/PLYStreamWriter.h"
#include "../VolumeIO/SliceChunkedMHDIO.h"

namespace parallel_mesh_extractor {

  struct ExtractionSettings {
    float IsoThreshold = 0.0f;

    /// Value sampled outside the volume. It is what closes the surface along the volume wall, so
    /// it belongs on the empty side of the isovalue for the mesh to come out capped there.
    float BackgroundValue = 0.0f;

    /// How many chunk layers to hold at once. One keeps the footprint smallest; more trades memory
    /// for fewer, larger reads.
    std::uint32_t LayersPerSlab = 1;

    /// Write voxel indices rather than world coordinates, i.e. apply neither the spacing nor the
    /// origin from the volume's meta data.
    bool VoxelCoordinates = false;

    PLYFormat Format = PLYFormat::BinaryLittleEndian;
  };

  struct ExtractionReport {
    std::array<std::uint32_t, 3> Dimensions{};
    std::array<float, 3> Spacing{};
    std::array<float, 3> Origin{};

    std::uint64_t Slabs = 0;
    std::uint64_t Chunks = 0;
    std::uint64_t Vertices = 0;
    std::uint64_t Triangles = 0;

    /// Slices held at once, and the bytes that costs. The point of the whole slab arrangement is
    /// that this does not grow with the depth of the volume.
    std::uint32_t SlabSlices = 0;
    std::uint64_t SlabBytes = 0;
  };

  /// Reads an MHD volume, extracts its isosurface and writes it as a PLY, holding only one slab of
  /// slices and one chunk layer's worth of shared vertices at a time.
  ///
  /// The volume is never assembled in memory: each slab is read, chunked, extracted, welded
  /// against the layer below it and handed straight to the writer, after which it is dropped. What
  /// peak memory depends on is the area of a slice and the slab depth, not the depth of the volume
  /// or the size of the mesh.
  inline ExtractionReport ExtractVolumeToPLY(std::filesystem::path const& input,
                                             std::filesystem::path const& output,
                                             ExtractionSettings const& settings,
                                             std::ostream* progress = nullptr) {
    SliceChunkedMHDIO volumeIO;
    volumeIO.SetFilePath(input);
    auto const metaData = volumeIO.ReadMetaData();

    ExtractionReport report;
    report.Dimensions = metaData.dim;
    report.Spacing    = metaData.spacing;
    report.Origin     = metaData.origin;

    auto const plan = PlanSlabs(metaData.dim, settings.LayersPerSlab);
    report.Slabs = plan.size();

    auto const sliceSize = static_cast<std::size_t>(metaData.dim[0]) * metaData.dim[1];
    report.SlabSlices = MaxSlabSliceCount(metaData.dim, settings.LayersPerSlab);
    report.SlabBytes  = static_cast<std::uint64_t>(sliceSize) * report.SlabSlices * sizeof(float);

    // Allocated once for the whole run, sized for the deepest slab of the plan.
    std::vector<float> slab(sliceSize * report.SlabSlices);

    PLYStreamWriter writer(output, settings.Format);
    MeshAssembler assembler;
    CudaChunkMeshExtractor extractor(settings.IsoThreshold);

    ChunkMesh chunkMesh;
    std::vector<std::array<float, 3>> newVertices;
    std::vector<std::uint32_t> faceIndices;

    for (auto const& window : plan) {
      // Read the slab in pieces. ReadSlices allocates a raw and a float buffer for every call, so
      // asking for the whole slab at once would briefly double its footprint for no benefit.
      constexpr std::uint32_t readChunkSlices = 16;
      for (std::uint32_t done = 0; done < window.ZCount; done += readChunkSlices) {
        std::uint32_t const count = std::min(readChunkSlices, window.ZCount - done);

        auto const piece = volumeIO.ReadSlices(static_cast<unsigned int>(window.ZBegin + done), count);
        if (!piece) {
          throw std::runtime_error("reading slices from the volume failed");
        }
        std::memcpy(slab.data() + static_cast<std::size_t>(done) * sliceSize, piece.get(),
                    static_cast<std::size_t>(count) * sliceSize * sizeof(float));
      }

      Chunkifier chunkifier(slab.data(), metaData.dim, window,
                            settings.IsoThreshold, settings.BackgroundValue);
      chunkifier.ComputeChunking(metaData.dim);

      for (auto const& chunk : chunkifier) {
        extractor.Extract(chunk, chunkMesh);
        ++report.Chunks;
        if (chunkMesh.IsEmpty()) continue;

        // Weld in voxel coordinates, which is the space the extraction produces and the only one
        // in which two chunks are guaranteed to agree bit for bit, then convert on the way out.
        assembler.Add(chunkMesh, newVertices, faceIndices);

        if (!settings.VoxelCoordinates) {
          for (auto& vertex : newVertices) {
            for (int axis = 0; axis < 3; ++axis) {
              vertex[axis] = metaData.origin[axis] + vertex[axis] * metaData.spacing[axis];
            }
          }
        }

        writer.WriteVertices(newVertices);
        writer.WriteFaces(faceIndices);
      }

      // Everything below the plane this slab shares with the next one is out of reach now.
      assembler.PruneBelowZ(window.PruneBelowZ);

      if (progress) {
        *progress << "  slab " << (&window - plan.data() + 1) << "/" << plan.size()
                  << ": layers [" << window.TileZBegin << ", "
                  << window.TileZBegin + window.TileZCount << "), slices ["
                  << window.ZBegin << ", " << window.ZBegin + window.ZCount << ")"
                  << ", " << assembler.VertexCount() << " vertices so far\n";
      }
    }

    writer.Finish();

    report.Vertices  = writer.VertexCount();
    report.Triangles = writer.TriangleCount();
    return report;
  }

} // namespace parallel_mesh_extractor
