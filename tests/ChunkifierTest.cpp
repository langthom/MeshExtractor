
#include <memory>
#include <string>
#include <vector>
#include "doctest.h"
#include "../MeshExtraction/Chunkifier.h"

namespace pme = parallel_mesh_extractor;

bool checkChunking(std::array<std::uint32_t, 3> const& dims) {
  auto const chunkSize = pme::Chunkifier::ChunkSize;

  // Create the input data, in which every tile already has its index as constant value.
  auto const numVoxels = static_cast<std::int64_t>(dims[0]) * dims[1] * dims[2];
  auto inputData = std::make_unique<float[]>(numVoxels);

  std::array<std::int64_t, 3> numTiles;
  for (int i = 0; i < 3; ++i) {
    numTiles[i] = (dims[i] + chunkSize - 1) / chunkSize;
  }

  #pragma omp parallel for
  for (std::int64_t i = 0; i < numVoxels; ++i) {
    // Conver the 1D index "i" into a 3D index.
    std::int64_t const z = i / (dims[0] * dims[1]);
    std::int64_t const k = i - z * dims[0] * dims[1];
    std::int64_t const y = k / dims[0];
    std::int64_t const x = k % dims[0];

    // Detect which tile this voxel belongs to.
    std::int64_t const tileX  = x / chunkSize;
    std::int64_t const tileY  = y / chunkSize;
    std::int64_t const tileZ  = z / chunkSize;
    std::int64_t const tileIx = (tileZ * numTiles[1] + tileY) * numTiles[0] + tileX;

    // Write the tile index for the voxel value.
    inputData[i] = static_cast<float>(tileIx + 1);
  }

  // Create the expected chunk results.
  // For the very first chunks in all axes (tileX = 0 or tileY = 0 or tileZ = 0), these will contain
  // a 1-voxel padding boundary. Likewise, the last chunks in the respective axes are padded to 
  // zeros as well. The intermediate chunks will respect the 1-voxel ghost layer overlay.
  std::vector<pme::Chunkifier::DataChunk> expectedChunks(numTiles[0] * numTiles[1] * numTiles[2]);

  auto tileCoordinateRange = [dims](std::int64_t tileIx, int axis) {
    auto tx = static_cast<std::uint32_t>(tileIx);
    // For the end, this is either the end of the chunk or the difference to the full volume.
    std::uint32_t const coordinateEnd = std::min(pme::Chunkifier::ChunkSize, dims[axis] - tx * chunkSize) - 1;
    return coordinateEnd;
  };

  for (std::int64_t tileZ = 0; tileZ < numTiles[2]; ++tileZ) {
    for (std::int64_t tileY = 0; tileY < numTiles[1]; ++tileY) {
      for (std::int64_t tileX = 0; tileX < numTiles[0]; ++tileX) {
        auto const tileIx1D = static_cast<float>((tileZ * numTiles[1] + tileY) * numTiles[0] + tileX);

        // Compute which portion of the current chunk receives the chunk index value.
        auto const xEnd = tileCoordinateRange(tileX, 0);
        auto const yEnd = tileCoordinateRange(tileY, 1);
        auto const zEnd = tileCoordinateRange(tileZ, 2);

        // Write the expected value (the tileIx1D value) to every affected voxel in the chunk.
        pme::Chunkifier::DataChunk& chunk = expectedChunks.at(tileIx1D);
        for (int z = 0; z <= zEnd; ++z) {
          for (int y = 0; y <= yEnd; ++y) {
            for (int x = 0; x <= xEnd; ++x) {
              chunk.data[z][y][x] = tileIx1D + 1;
            }
          }
        }
      }
    }
  }

  // Now, run the chunkifier and compare the results. We expect the traversal order Z->Y->X.
  pme::Chunkifier chunkifier(inputData.get(), dims);
  auto chunkIterator         = chunkifier.begin();
  auto chunkifierEnd         = chunkifier.end();
  auto expectedChunkIterator = expectedChunks.begin();

  int ci = 0;

  for (; chunkIterator != chunkifierEnd; ++chunkIterator, ++expectedChunkIterator, ++ci) {
    auto const& currentChunk  = *chunkIterator;
    auto const& expectedChunk = *expectedChunkIterator;

    for (int z = 0; z < chunkSize; ++z) {
      for (int y = 0; y < chunkSize; ++y) {
        for (int x = 0; x < chunkSize; ++x) {
          if (expectedChunk.data[z][y][x] != currentChunk.data[z][y][x]) {
            return false;
          }
        }
      }
    }
  }

  return true;
}

TEST_CASE("Chunkifying a too small data buffer") {
  // In this test case, the original data buffer is smaller than a single chunk is.
  // The expected result is the original data buffer, but padded with zeros.
  CHECK(checkChunking({32, 32, 32}));
}

TEST_CASE("Chunkifying a perfectly-sized data buffer") {
  // Here, the chunk including the data and a 1 voxel padding on all sides in total
  // takes the chunk size.
  CHECK(checkChunking({pme::Chunkifier::ChunkSize, pme::Chunkifier::ChunkSize, pme::Chunkifier::ChunkSize}));
}

TEST_CASE("Chunkifying larger data buffer") {
  // Here there is actually something to do, i.e., there is more than a single chunk.
  CHECK(checkChunking({32, 130, 64}));
}


