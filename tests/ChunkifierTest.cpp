
#include <array>
#include <cstdint>
#include <memory>
#include "doctest.h"
#include "../MeshExtraction/Chunkifier.h"

namespace pme = parallel_mesh_extractor;

namespace {

  // The chunk layout is a hard contract of the extraction pipeline: a 64^3 allocation owning a
  // 62^3 core, wrapped in a 1 voxel ghost shell replicating the neighbors. The tests pin it with
  // their own literals instead of deriving from the chunkifier's constants, so that changing those
  // constants shows up as a test failure rather than silently moving the expectations along.
  constexpr std::int32_t ChunkSize  = 64;
  constexpr std::int32_t GhostWidth = 1;
  constexpr std::int32_t CoreSize   = ChunkSize - 2 * GhostWidth;

  /// Create a volume in which every voxel carries a value that uniquely identifies its global
  /// position, namely its flat index plus one. The offset keeps every real voxel distinguishable
  /// from a 0.0f background value.
  std::unique_ptr<float[]> makeVolume(std::array<std::uint32_t, 3> const& dims) {
    auto const numVoxels = static_cast<std::int64_t>(dims[0]) * dims[1] * dims[2];

    // A float only represents integers exactly up to 2^24, so the encoding above stays lossless
    // only for volumes below that size. All test volumes are deliberately kept small.
    REQUIRE(numVoxels < (1 << 24));

    auto volume = std::make_unique<float[]>(numVoxels);
    for (std::int64_t i = 0; i < numVoxels; ++i) {
      volume[i] = static_cast<float>(i + 1);
    }
    return volume;
  }

  /// Derive the value a chunk must carry at one of its local indices.
  /// This deliberately re-derives the expectation from the chunk layout contract rather than from
  /// the chunkifier: local index l of chunk c samples the global coordinate
  /// (c * CoreSize + l - GhostWidth), and anything outside the volume is background.
  float expectedAt(std::array<std::uint32_t, 3> const& chunkCoord,
                   std::array<std::int32_t, 3> const& local,
                   std::array<std::uint32_t, 3> const& dims,
                   float background) {
    std::array<std::int64_t, 3> global;

    for (int axis = 0; axis < 3; ++axis) {
      global[axis] = static_cast<std::int64_t>(chunkCoord[axis]) * CoreSize
                   + local[axis] - GhostWidth;

      if (global[axis] < 0 || global[axis] >= dims[axis]) {
        return background;
      }
    }

    auto const flatIndex = (global[2] * dims[1] + global[1]) * dims[0] + global[0];
    return static_cast<float>(flatIndex + 1);
  }

  void checkChunking(std::array<std::uint32_t, 3> const& dims, float background = 0.0f) {
    auto const volume = makeVolume(dims);

    std::array<std::uint32_t, 3> numChunks;
    for (int axis = 0; axis < 3; ++axis) {
      numChunks[axis] = (dims[axis] + CoreSize - 1) / CoreSize;
    }

    pme::Chunkifier chunkifier(volume.get(), dims, background);

    // We expect the traversal order Z->Y->X, i.e., X advancing fastest.
    std::uint32_t chunkCount = 0;

    for (auto const& chunk : chunkifier) {
      std::array<std::uint32_t, 3> const chunkCoord = {
        chunkCount % numChunks[0],
        (chunkCount / numChunks[0]) % numChunks[1],
        chunkCount / (numChunks[0] * numChunks[1]),
      };

      INFO("chunk ", chunkCount);

      // The chunk has to report the global origin of the region it owns, which the extraction
      // needs later on to place its vertices in world space.
      std::array<std::uint32_t, 3> const expectedOrigin = {
        chunkCoord[0] * CoreSize,
        chunkCoord[1] * CoreSize,
        chunkCoord[2] * CoreSize,
      };
      REQUIRE(chunk.CoreOrigin == expectedOrigin);

      for (std::int32_t z = 0; z < ChunkSize; ++z) {
        for (std::int32_t y = 0; y < ChunkSize; ++y) {
          for (std::int32_t x = 0; x < ChunkSize; ++x) {
            float const expected = expectedAt(chunkCoord, {x, y, z}, dims, background);

            // Only report on an actual mismatch, so that a failure names the offending voxel
            // instead of drowning in a quarter million successful assertions per chunk.
            if (chunk.data[z][y][x] != expected) {
              INFO("local voxel (", x, ", ", y, ", ", z, ")");
              REQUIRE(chunk.data[z][y][x] == expected);
            }
          }
        }
      }

      ++chunkCount;
    }

    CHECK(chunkCount == numChunks[0] * numChunks[1] * numChunks[2]);
  }

} // namespace

TEST_CASE("The chunk layout matches the extraction contract") {
  // The extraction kernels rely on the allocation being a power of two, on a single ghost layer
  // being available for the boundary cells, and on the core size being the chunk-to-chunk stride.
  CHECK(pme::Chunkifier::ChunkSize  == ChunkSize);
  CHECK(pme::Chunkifier::GhostWidth == GhostWidth);
  CHECK(pme::Chunkifier::CoreSize   == CoreSize);
  CHECK(pme::Chunkifier::DataChunk::ChunkDimSize == ChunkSize);
}

TEST_CASE("Chunkifying a data buffer smaller than a single chunk core") {
  // The whole volume fits into the core of one chunk, so the chunk is surrounded by background
  // on all 6 sides -- both where the ghost shell reaches below the volume and where the core
  // itself already runs past its end.
  checkChunking({32, 32, 32});
}

TEST_CASE("Chunkifying a data buffer matching the chunk core exactly") {
  // The core of a single chunk is filled completely, and the entire ghost shell around it falls
  // outside the volume.
  checkChunking({CoreSize, CoreSize, CoreSize});
}

TEST_CASE("Chunkifying a data buffer exceeding the chunk core by a single voxel") {
  // Guards the rounding of the chunk count: a single voxel past the core forces a second chunk
  // per axis, whose own core holds just that one voxel.
  constexpr std::uint32_t size = CoreSize + 1;
  checkChunking({size, size, size});
}

TEST_CASE("Chunkifying a data buffer matching the chunk allocation size") {
  // The allocation size is not the stride: 64 voxels per axis still spill into a second chunk,
  // since a chunk only owns CoreSize of them.
  constexpr std::uint32_t size = ChunkSize;
  checkChunking({size, size, size});
}

TEST_CASE("Chunkifying a larger data buffer") {
  // Here there is actually something to do, i.e., there is more than a single chunk, and the
  // number of chunks differs per axis.
  checkChunking({32, 130, 64});
}

TEST_CASE("Chunkifying a data buffer that is an exact multiple of the chunk core") {
  // Without a remainder every ghost voxel between two chunks is backed by real data, so this
  // covers the case in which the chunks in the middle are filled completely.
  constexpr std::uint32_t size = 2 * CoreSize;
  checkChunking({size, size, size});
}

TEST_CASE("Chunkifying pads outside of the volume with the background value") {
  // Padding must honor the configured background, e.g. the air value of a CT scan, rather than
  // defaulting to zero.
  checkChunking({32, 32, 32}, -1000.0f);
}
