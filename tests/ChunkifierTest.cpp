
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include "doctest.h"
#include "../MeshExtraction/Chunkifier.h"

namespace pme = parallel_mesh_extractor;

namespace {

  // The chunk layout is a hard contract of the extraction pipeline: a 64^3 allocation owning a
  // 62^3 core, wrapped in a 1 voxel ghost shell replicating the neighbors. The tests pin it with
  // their own literals instead of deriving from the chunkifier's constants, so that changing those
  // constants shows up as a test failure rather than silently moving the expectations along.
  constexpr std::int64_t ChunkSize  = 64;
  constexpr std::int64_t GhostWidth = 1;
  constexpr std::int64_t CoreSize   = ChunkSize - 2 * GhostWidth;

  using Volume = std::unique_ptr<float[]>;
  using Dims   = std::array<std::uint32_t, 3>;

  std::int64_t numVoxelsOf(Dims const& dims) {
    return static_cast<std::int64_t>(dims[0]) * dims[1] * dims[2];
  }

  /// Volume in which every voxel carries a value that uniquely identifies its global position,
  /// while the alternating sign makes almost every tile span a surface crossing at an isovalue of
  /// zero. That way one volume serves both the content and the culling expectations.
  Volume makeVolume(Dims const& dims) {
    auto const numVoxels = numVoxelsOf(dims);

    // A float only represents integers exactly up to 2^24, so the encoding stays lossless only for
    // volumes below that size. All test volumes are deliberately kept small.
    REQUIRE(numVoxels + 10000 < (1 << 24));

    Volume volume = std::make_unique<float[]>(numVoxels);
    for (std::int64_t i = 0; i < numVoxels; ++i) {
      volume[i] = static_cast<float>((i % 2 == 0 ? -1 : 1) * (10000 + i));
    }
    return volume;
  }

  /// Volume whose value only depends on the Z coordinate, so that an isovalue selects a
  /// contiguous band of chunks along Z and culls everything above it.
  Volume makeRampVolume(Dims const& dims) {
    auto const numVoxels = numVoxelsOf(dims);
    Volume volume = std::make_unique<float[]>(numVoxels);
    for (std::int64_t i = 0; i < numVoxels; ++i) {
      volume[i] = static_cast<float>(i / (static_cast<std::int64_t>(dims[0]) * dims[1]));
    }
    return volume;
  }

  Volume makeConstantVolume(Dims const& dims, float value) {
    Volume volume = std::make_unique<float[]>(numVoxelsOf(dims));
    for (std::int64_t i = 0; i < numVoxelsOf(dims); ++i) volume[i] = value;
    return volume;
  }

  std::array<std::int64_t, 3> numTilesOf(Dims const& dims) {
    std::array<std::int64_t, 3> numTiles;
    for (int axis = 0; axis < 3; ++axis) {
      numTiles[axis] = (dims[axis] + CoreSize - 1) / CoreSize;
    }
    return numTiles;
  }

  /// Read a global voxel, or the background value where the coordinate leaves the volume.
  float sampleOrBackground(Volume const& volume, Dims const& dims,
                           std::array<std::int64_t, 3> const& global, float background) {
    for (int axis = 0; axis < 3; ++axis) {
      if (global[axis] < 0 || global[axis] >= dims[axis]) return background;
    }
    auto const flatIndex = (global[2] * dims[1] + global[1]) * dims[0] + global[0];
    return volume[flatIndex];
  }

  /// Derive the value a chunk must carry at one of its local indices, straight from the chunk
  /// layout contract: local index l of tile c samples the global coordinate
  /// (c * CoreSize + l - GhostWidth), and anything outside the volume is background.
  float expectedAt(Volume const& volume, Dims const& dims, float background,
                   std::array<std::int64_t, 3> const& tileCoord,
                   std::array<std::int64_t, 3> const& local) {
    std::array<std::int64_t, 3> global;
    for (int axis = 0; axis < 3; ++axis) {
      global[axis] = tileCoord[axis] * CoreSize + local[axis] - GhostWidth;
    }
    return sampleOrBackground(volume, dims, global, background);
  }

  /// Derive the value range a tile must report, by materializing the chunk the way the extraction
  /// sees it and reducing over the voxels its owned cells actually touch. A tile owns the cells
  /// starting at each of its CoreSize core voxels, and a cell spans one voxel further, so those
  /// cells cover the local indices [GhostWidth, ChunkSize).
  std::array<float, 2> expectedRange(Volume const& volume, Dims const& dims, float background,
                                     std::array<std::int64_t, 3> const& tileCoord) {
    float lo = +std::numeric_limits<float>::infinity();
    float hi = -std::numeric_limits<float>::infinity();

    for (std::int64_t z = GhostWidth; z < ChunkSize; ++z) {
      for (std::int64_t y = GhostWidth; y < ChunkSize; ++y) {
        for (std::int64_t x = GhostWidth; x < ChunkSize; ++x) {
          float const v = expectedAt(volume, dims, background, tileCoord, {x, y, z});
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
      }
    }
    return {lo, hi};
  }

  /// The tiles the chunkifier is expected to hand out, in traversal order Z->Y->X, i.e., with X
  /// advancing fastest. A tile is materialized exactly when the isovalue lies inside its range.
  std::vector<std::array<std::int64_t, 3>> expectedTileCoords(Volume const& volume, Dims const& dims,
                                                              float isoThreshold, float background) {
    auto const numTiles = numTilesOf(dims);
    std::vector<std::array<std::int64_t, 3>> expected;

    for (std::int64_t tileZ = 0; tileZ < numTiles[2]; ++tileZ) {
      for (std::int64_t tileY = 0; tileY < numTiles[1]; ++tileY) {
        for (std::int64_t tileX = 0; tileX < numTiles[0]; ++tileX) {
          std::array<std::int64_t, 3> const tileCoord = {tileX, tileY, tileZ};
          auto const [lo, hi] = expectedRange(volume, dims, background, tileCoord);
          if (lo <= isoThreshold && isoThreshold <= hi) expected.push_back(tileCoord);
        }
      }
    }
    return expected;
  }

  /// Run the chunkifier and compare both which chunks it materializes and what they contain.
  void checkChunking(Volume const& volume, Dims const& dims,
                     float isoThreshold, float background, bool checkContent = true) {
    auto const expected = expectedTileCoords(volume, dims, isoThreshold, background);

    pme::Chunkifier chunkifier(volume.get(), dims, isoThreshold, background);
    chunkifier.ComputeChunking(dims);

    std::size_t emitted = 0;

    for (auto it = chunkifier.begin(); it != chunkifier.end(); ++it) {
      // The culling makes the emitted count data dependent, so bail out before dereferencing once
      // more chunks arrive than are expected, rather than reading past the expectations.
      INFO("emitted chunk ", emitted, ", expected ", expected.size(), " chunks in total");
      REQUIRE(emitted < expected.size());

      auto const chunk = *it;

      auto const& tileCoord = expected[emitted];
      std::array<std::int64_t, 3> const expectedOrigin = {
        tileCoord[0] * CoreSize, tileCoord[1] * CoreSize, tileCoord[2] * CoreSize,
      };
      REQUIRE(chunk.CoreOrigin == expectedOrigin);

      if (checkContent) {
        for (std::int64_t z = 0; z < ChunkSize; ++z) {
          for (std::int64_t y = 0; y < ChunkSize; ++y) {
            for (std::int64_t x = 0; x < ChunkSize; ++x) {
              float const want = expectedAt(volume, dims, background, tileCoord, {x, y, z});

              // Only report on an actual mismatch, so that a failure names the offending voxel
              // instead of drowning in a quarter million successful assertions per chunk.
              if (chunk.data[z][y][x] != want) {
                INFO("local voxel (", x, ", ", y, ", ", z, ")");
                REQUIRE(chunk.data[z][y][x] == want);
              }
            }
          }
        }
      }

      ++emitted;
    }

    CHECK(emitted == expected.size());
  }

  /// Whether the chunkifier hands out no chunks at all, i.e. its begin() already compares equal to
  /// its end(). Wrapped in a function so that the comparison is not fed through doctest's
  /// expression decomposition, which cannot take the iterators apart.
  bool iterationIsEmpty(pme::Chunkifier const& chunkifier) {
    return !(chunkifier.begin() != chunkifier.end());
  }

  /// Count how many chunks the chunkifier hands out, without inspecting them.
  std::size_t countChunks(Volume const& volume, Dims const& dims,
                          float isoThreshold, float background) {
    pme::Chunkifier chunkifier(volume.get(), dims, isoThreshold, background);
    chunkifier.ComputeChunking(dims);

    std::size_t count = 0;
    auto const cap = static_cast<std::size_t>(numTilesOf(dims)[0] * numTilesOf(dims)[1] * numTilesOf(dims)[2]);

    for (auto it = chunkifier.begin(); it != chunkifier.end(); ++it) {
      // Check before dereferencing: an iteration that runs past its last chunk must surface as a
      // failed expectation, not as a crash inside the chunkifier.
      REQUIRE(count < cap);
      auto const chunk = *it;
      (void)chunk;
      ++count;
    }
    return count;
  }

} // namespace

// ------------------------------------- the chunk layout -------------------------------------- //

TEST_CASE("The chunk layout matches the extraction contract") {
  // The extraction kernels rely on the allocation being a power of two, on a single ghost layer
  // being available for the boundary cells, and on the core size being the chunk-to-chunk stride.
  CHECK(pme::Chunkifier::ChunkSize  == ChunkSize);
  CHECK(pme::Chunkifier::GhostWidth == GhostWidth);
  CHECK(pme::Chunkifier::CoreSize   == CoreSize);
  CHECK(pme::Chunkifier::DataChunk::ChunkDimSize == ChunkSize);
}

// ------------------------------ chunk placement and content ---------------------------------- //

TEST_CASE("Chunkifying a data buffer smaller than a single chunk core") {
  // The whole volume fits into the core of one chunk, so the chunk is surrounded by background
  // on all 6 sides -- both where the ghost shell reaches below the volume and where the core
  // itself already runs past its end.
  Dims const dims = {32, 32, 32};
  checkChunking(makeVolume(dims), dims, 0.0f, 0.0f);
}

TEST_CASE("Chunkifying a data buffer matching the chunk core exactly") {
  // The core of a single chunk is filled completely, and the entire ghost shell around it falls
  // outside the volume.
  Dims const dims = {CoreSize, CoreSize, CoreSize};
  checkChunking(makeVolume(dims), dims, 0.0f, 0.0f);
}

TEST_CASE("Chunkifying a data buffer exceeding the chunk core by a single voxel") {
  // Guards the rounding of the chunk count: a single voxel past the core forces a second chunk
  // per axis, whose own core holds just that one voxel.
  Dims const dims = {CoreSize + 1, CoreSize + 1, CoreSize + 1};
  checkChunking(makeVolume(dims), dims, 0.0f, 0.0f);
}

TEST_CASE("Chunkifying a data buffer matching the chunk allocation size") {
  // The allocation size is not the stride: 64 voxels per axis still spill into a second chunk,
  // since a chunk only owns CoreSize of them.
  Dims const dims = {ChunkSize, ChunkSize, ChunkSize};
  checkChunking(makeVolume(dims), dims, 0.0f, 0.0f);
}

TEST_CASE("Chunkifying a larger data buffer") {
  // Here there is actually something to do, i.e., there is more than a single chunk, and the
  // number of chunks differs per axis.
  Dims const dims = {32, 130, 64};
  checkChunking(makeVolume(dims), dims, 0.0f, 0.0f);
}

TEST_CASE("Chunkifying a data buffer that is an exact multiple of the chunk core") {
  // Without a remainder every ghost voxel between two chunks is backed by real data, so this
  // covers the case in which the chunks in the middle are filled completely.
  Dims const dims = {2 * CoreSize, 2 * CoreSize, 2 * CoreSize};
  checkChunking(makeVolume(dims), dims, 0.0f, 0.0f);
}

TEST_CASE("Chunkifying pads outside of the volume with the background value") {
  // Padding must honor the configured background, e.g. the air value of a CT scan, rather than
  // defaulting to zero. The isovalue is placed between the data and the background so that no
  // chunk is culled away before its padding can be inspected.
  Dims const dims = {32, 32, 32};
  checkChunking(makeVolume(dims), dims, -5000.0f, -1000.0f);
}

// -------------------------------- culling by the ISO threshold ------------------------------- //

TEST_CASE("Chunks whose value range excludes the isovalue are never materialized") {
  // A homogeneous volume carries no surface at all, so with the isovalue outside of its range
  // the chunkifier must hand out nothing and never touch a chunk buffer.
  Dims const dims = {130, 130, 130};
  auto const volume = makeConstantVolume(dims, 500.0f);
  CHECK(countChunks(volume, dims, 0.0f, 500.0f) == 0);
}

TEST_CASE("An isovalue on the boundary of the value range still materializes the chunk") {
  // The range test has to be inclusive: a surface exactly at the isovalue is still a surface.
  Dims const dims = {130, 130, 130};
  auto const volume = makeConstantVolume(dims, 500.0f);
  auto const numTiles = numTilesOf(dims);
  CHECK(countChunks(volume, dims, 500.0f, 500.0f)
        == static_cast<std::size_t>(numTiles[0] * numTiles[1] * numTiles[2]));
}

TEST_CASE("The background value participates in the culling decision") {
  // The volume itself is homogeneous, so the only surface in the whole dataset is the one the
  // padding creates along the volume wall. Exactly those chunks reaching outside of the volume
  // must survive the culling -- otherwise the mesh would not be closed at the dataset boundary.
  Dims const dims = {2 * CoreSize, 2 * CoreSize, 2 * CoreSize};
  auto const volume = makeConstantVolume(dims, 1000.0f);
  checkChunking(volume, dims, 0.0f, -1000.0f);

  // Only the tile at index 0 of every axis stays clear of the volume wall.
  CHECK(countChunks(volume, dims, 0.0f, -1000.0f) == 7);
}

TEST_CASE("Culling keeps the traversal order of the surviving chunks") {
  // Whatever is skipped, the chunks that do come out must still arrive in the order Z->Y->X.
  Dims const dims = {130, 70, 64};
  checkChunking(makeVolume(dims), dims, 0.0f, 0.0f, /*checkContent=*/false);
}

// ------------------------------- iterating over nothing at all ------------------------------- //

TEST_CASE("Iterating a chunkifier that culls every chunk yields an empty but well behaved loop") {
  // With the isovalue outside of the data range there is no surface anywhere in the volume, so
  // every single chunk is culled. The iteration then has to degenerate cleanly: begin() must
  // already compare equal to end(), and neither loop form may enter its body or walk past the end
  // of the precomputed tile list.
  Dims const dims = {130, 130, 130};
  auto const volume = makeConstantVolume(dims, 500.0f);

  // Without the culling this volume would be handed out as 27 chunks, so the iterator really does
  // have a non-empty tile list to skip over.
  auto const numTiles = numTilesOf(dims);
  auto const totalTiles = static_cast<std::size_t>(numTiles[0] * numTiles[1] * numTiles[2]);
  REQUIRE(totalTiles == 27);

  SUBCASE("isovalue below the data range") {
    pme::Chunkifier chunkifier(volume.get(), dims, -1000.0f, 500.0f);
    chunkifier.ComputeChunking(dims);

    // The past-the-end iterator has to be reachable without taking a single step.
    CHECK(iterationIsEmpty(chunkifier));

    // An explicit loop must terminate right away...
    std::size_t explicitIterations = 0;
    for (auto it = chunkifier.begin(); it != chunkifier.end(); ++it) {
      // Never let a non-terminating iteration run away; report it as a failure instead.
      REQUIRE(++explicitIterations <= totalTiles);
    }
    CHECK(explicitIterations == 0);

    // ...and so must a range based one, which is how the extraction is going to consume this.
    std::size_t rangeIterations = 0;
    for (auto const& chunk : chunkifier) {
      (void)chunk;
      REQUIRE(++rangeIterations <= totalTiles);
    }
    CHECK(rangeIterations == 0);
  }

  SUBCASE("isovalue above the data range") {
    pme::Chunkifier chunkifier(volume.get(), dims, 5000.0f, 500.0f);
    chunkifier.ComputeChunking(dims);

    CHECK(iterationIsEmpty(chunkifier));

    std::size_t iterations = 0;
    for (auto const& chunk : chunkifier) {
      (void)chunk;
      REQUIRE(++iterations <= totalTiles);
    }
    CHECK(iterations == 0);
  }
}

TEST_CASE("Iterating a chunkifier whose chunking was never computed yields an empty loop") {
  // Nothing has been chunked yet, so there is not even a tile list to cull. Iterating anyway must
  // still be well defined rather than comparing two iterators that point nowhere.
  Dims const dims = {130, 130, 130};
  auto const volume = makeVolume(dims);

  pme::Chunkifier chunkifier(volume.get(), dims, 0.0f, 0.0f);

  CHECK(iterationIsEmpty(chunkifier));

  std::size_t iterations = 0;
  for (auto const& chunk : chunkifier) {
    (void)chunk;
    REQUIRE(++iterations <= 1);
  }
  CHECK(iterations == 0);
}

TEST_CASE("Iteration terminates when the trailing chunks are all culled") {
  // The ramp carries its surface only in the lowest chunks along Z, and the background sits above
  // the whole data range, so the entire tail of the tile list is culled. Reaching the end then
  // happens inside operator++, which has to stop at the end rather than step past it.
  Dims const dims = {130, 130, 130};
  auto const volume = makeRampVolume(dims);

  checkChunking(volume, dims, 10.0f, 200.0f, /*checkContent=*/false);

  // Only the lowest of the three chunk layers along Z spans the isovalue.
  CHECK(countChunks(volume, dims, 10.0f, 200.0f) == 9);
}
