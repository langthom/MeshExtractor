#include "Chunkifier.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <omp.h>

namespace pme = parallel_mesh_extractor;

// --------------------------------------- Chunkifier ------------------------------------------ //

pme::Chunkifier::Chunkifier(float const* data, std::array<std::uint32_t, 3> const& dim,
                            float isoThreshold, float backgroundValue) noexcept
{
  ChunkingDataCollection cdc;
  cdc.Data = data;
  cdc.Dimensions = dim;
  cdc.ISOThreshold = isoThreshold;
  cdc.BackgroundValue = backgroundValue;

  this->ChunkingData = cdc;
}

void pme::Chunkifier::ComputeChunking(std::array<std::uint32_t, 3> const& dim) {
  // Compute the number of chunks.
  // Since the chunks only own their inner core and the ghost shell is pure overlap, consecutive
  // chunks advance by CoreSize rather than by the allocated ChunkSize.
  std::uint32_t const numTilesZ = (dim[2] + Chunkifier::CoreSize - 1) / Chunkifier::CoreSize;
  std::uint32_t const numTilesY = (dim[1] + Chunkifier::CoreSize - 1) / Chunkifier::CoreSize;
  std::uint32_t const numTilesX = (dim[0] + Chunkifier::CoreSize - 1) / Chunkifier::CoreSize;

  // Construct the list of core origins, i.e., the global coordinate of the first voxel each chunk
  // owns. The ghost shell of a chunk then covers the coordinates [origin-GhostWidth, origin) and
  // [origin+CoreSize, origin+CoreSize+GhostWidth).
  std::int64_t const numTiles = static_cast<std::int64_t>(numTilesX) * numTilesY * numTilesZ;
  this->ChunkingData.CoreOrigins.resize(numTiles);
  this->ChunkingData.ValueRanges.resize(numTiles);

  // Compute how many threads to use.
  // In case of a fixed defined OMP_NUM_THREADS, use that. Otherwise, use the available number of
  // threads which might be reduced depending on the actual workload (number of Z axis slices).
  int numThreads = 1;
  #ifdef OMP_NUM_THREADS
    numThreads = OMP_NUM_THREADS;
  #else
    numThreads = static_cast<int>(std::min<std::int64_t>(omp_get_max_threads(), numTiles));
  #endif
  numThreads = std::max(numThreads, 1);

  #pragma omp parallel for schedule(static) num_threads(numThreads)
  for (std::int64_t tileIx1D = 0; tileIx1D < numTiles; ++tileIx1D) {
    // Compute the origin of the tile.
    std::int64_t const tileZ = tileIx1D / (numTilesY * numTilesX);
    std::int64_t const tyz   = tileIx1D - tileZ * numTilesY * numTilesX;
    std::int64_t const tileY = tyz / numTilesX;
    std::int64_t const tileX = tyz % numTilesX;

    std::array<std::int64_t, 3> const origin = {
      tileX * Chunkifier::CoreSize,
      tileY * Chunkifier::CoreSize,
      tileZ * Chunkifier::CoreSize,
    };
    this->ChunkingData.CoreOrigins[tileIx1D] = origin;

    // ----------------------------------------
    // Get the value range in the "core" region of this current tile.
    // We compute the range at this point, so when the value range does
    // not fit the ISO threshold later on, we never copy the data to 
    // a chunk buffer.
    float tileMin = +std::numeric_limits<float>::infinity();
    float tileMax = -std::numeric_limits<float>::infinity();

    // The cells a tile owns start at each of its CoreSize core voxels and span one voxel further,
    // so the sampled box has to be CoreSize+1 voxels wide: the core itself plus the first voxel of
    // the upper ghost layer. Sampling only CoreSize would miss a surface crossing in the outermost
    // owned cell and cull a tile that does carry geometry, which would leave a hole in the mesh.
    std::array<std::int64_t, 3> rangeBegin, rangeEnd;
    bool reachesOutsideVolume = false;

    for (int axis = 0; axis < 3; ++axis) {
      std::int64_t const wantedEnd = origin[axis] + Chunkifier::CoreSize + 1;
      rangeBegin[axis] = origin[axis];
      rangeEnd[axis]   = std::min<std::int64_t>(wantedEnd, dim[axis]);
      reachesOutsideVolume |= (wantedEnd > static_cast<std::int64_t>(dim[axis]));
    }

    // Where that box reaches past the volume, the chunk is padded with the background value. That
    // padding is part of the data the extraction sees, and it is what closes the mesh along the
    // volume wall, so it has to participate in the range as well.
    if (reachesOutsideVolume) {
      tileMin = std::min(tileMin, this->ChunkingData.BackgroundValue);
      tileMax = std::max(tileMax, this->ChunkingData.BackgroundValue);
    }

    std::int64_t const rowLength = rangeEnd[0] - rangeBegin[0];

    for (std::int64_t z = rangeBegin[2]; z < rangeEnd[2]; ++z) {
      for (std::int64_t y = rangeBegin[1]; y < rangeEnd[1]; ++y) {
        // Index arithmetic in std::size_t throughout: a volume of a few thousand voxels per axis
        // already exceeds what a 32 bit index can address.
        std::size_t rowIx1D = static_cast<std::size_t>(z) * dim[1];
        rowIx1D += static_cast<std::size_t>(y);
        rowIx1D *= dim[0];
        rowIx1D += static_cast<std::size_t>(rangeBegin[0]);

        float const* const row = this->ChunkingData.Data + rowIx1D;

        // The reduction is requested explicitly because the compiler will not vectorize a float
        // min/max reduction on its own: the NaN semantics of std::min do not match what vminps
        // does, so without this pragma the loop stays scalar and costs several times as much.
        #pragma omp simd reduction(min:tileMin) reduction(max:tileMax)
        for (std::int64_t x = 0; x < rowLength; ++x) {
          tileMin = std::min(tileMin, row[x]);
          tileMax = std::max(tileMax, row[x]);
        }
      }
    }

    this->ChunkingData.ValueRanges[tileIx1D] = {tileMin, tileMax};
  }
}

pme::Chunkifier::ChunkIterator pme::Chunkifier::begin() const {
  return ChunkIterator(&this->ChunkingData);
}

pme::Chunkifier::ChunkIterator pme::Chunkifier::end() const {
  return ChunkIterator(nullptr);
}

// -------------------------------------- ChunkIterator ---------------------------------------- //

pme::Chunkifier::ChunkIterator::ChunkIterator(ChunkingDataCollection const* chunkingDataCollectionPtr) noexcept
  : ChunkingDataPtr(chunkingDataCollectionPtr)
{
  // A null collection constructs the past-the-end sentinel. It deliberately carries no position of
  // its own -- there is no collection to take one from -- and Equals() below compares against it by
  // asking whether the other side has any chunks left instead.
  if (ChunkingDataPtr) {
    this->CoreOriginsIterator = ChunkingDataPtr->CoreOrigins.cbegin();
    this->CoreOriginsEnd      = ChunkingDataPtr->CoreOrigins.cend();
    this->ValueRangesIterator = ChunkingDataPtr->ValueRanges.cbegin();

    // Settle the culling immediately, so that even the very first chunk handed out is one that
    // actually carries the surface.
    this->SkipCulledChunks();
  }
}

void pme::Chunkifier::ChunkIterator::SkipCulledChunks() {
  // The culling has to be resolved here rather than on dereference: a range based for loop compares
  // against end() *before* it dereferences, so by that point the iterator must already have skipped
  // over every chunk whose value range does not contain the ISO threshold. Resolving it in
  // operator*() instead would make the loop run one chunk past its last surviving one.
  while (this->CoreOriginsIterator != this->CoreOriginsEnd) {
    auto const [lo, hi] = *this->ValueRangesIterator;
    if (lo <= this->ChunkingDataPtr->ISOThreshold && this->ChunkingDataPtr->ISOThreshold <= hi) {
      return;
    }

    ++this->CoreOriginsIterator;
    ++this->ValueRangesIterator;
  }
}

bool pme::Chunkifier::ChunkIterator::AtEnd() const {
  return this->ChunkingDataPtr == nullptr || this->CoreOriginsIterator == this->CoreOriginsEnd;
}

pme::Chunkifier::DataChunk pme::Chunkifier::ChunkIterator::operator*() const {
  // The constructor and operator++ have already skipped the culled chunks, so the iterator always
  // rests on one that is meant to be materialized.
  assert(!this->AtEnd());

  // Get the core origin of the current chunk.
  auto const coreOrigin = *this->CoreOriginsIterator;

  // Construct the target chunk.
  DataChunk chunk;
  chunk.CoreOrigin = coreOrigin;

  // Determine which local indices of the chunk actually map onto a voxel of the volume.
  // The local index l samples the global coordinate (coreOrigin + l - GhostWidth), so it is valid
  // exactly as long as that coordinate stays within [0, dim). Resolving this per axis once keeps
  // the copy loop below free of any per-voxel bounds check.
  std::array<std::uint32_t, 3> localBegin, localEnd;
  bool coversFullChunk = true;

  for (int axis = 0; axis < 3; ++axis) {
    std::uint32_t const origin = coreOrigin[axis];
    localBegin[axis] = (origin < Chunkifier::GhostWidth) ? Chunkifier::GhostWidth - origin : 0;
    localEnd[axis]   = std::min<std::uint32_t>(Chunkifier::ChunkSize, this->ChunkingDataPtr->Dimensions[axis] - origin + Chunkifier::GhostWidth);
    coversFullChunk &= (localBegin[axis] == 0 && localEnd[axis] == Chunkifier::ChunkSize);
  }

  // Everything outside of that region reaches across the volume boundary and receives the
  // background value instead. Chunks lying fully inside the volume are overwritten completely by
  // the copy below and can skip this.
  if (!coversFullChunk) {
    chunk.Fill(this->ChunkingDataPtr->BackgroundValue);
  }

  // Global coordinate the local index 0 of this chunk maps to. This is -GhostWidth for the first
  // chunk of an axis, hence the signed type.
  std::array<std::int64_t, 3> globalBase;
  for (int axis = 0; axis < 3; ++axis) {
    globalBase[axis] = static_cast<std::int64_t>(coreOrigin[axis]) - Chunkifier::GhostWidth;
  }

  auto const zBegin = static_cast<std::int32_t>(localBegin[2]);
  auto const zEnd   = static_cast<std::int32_t>(localEnd[2]);
  auto const yBegin = static_cast<std::int32_t>(localBegin[1]);
  auto const yEnd   = static_cast<std::int32_t>(localEnd[1]);

  // Along X the valid range is contiguous in both the volume and the chunk, so a whole row can be
  // copied in one go instead of voxel by voxel.
  std::size_t const rowLength = localEnd[0] - localBegin[0];

  // Compute how many threads to use.
  // In case of a fixed defined OMP_NUM_THREADS, use that. Otherwise, use the available number of
  // threads which might be reduced depending on the actual workload (number of Z axis slices).
  int numThreads = 1;
  #ifdef OMP_NUM_THREADS
    numThreads = OMP_NUM_THREADS;
  #else
    numThreads = std::min(omp_get_max_threads(), zEnd - zBegin);
  #endif
  numThreads = std::max(numThreads, 1);

  // Copy the data to the allocated chunk. Since both memory regions live in main memory, a simple
  // copy loop suffices. The outermost loop is parallelized, yet including the innermost loops would
  // likely destroy cache coherence and introduce too much thread creation overhead.
  #pragma omp parallel for schedule(static) num_threads(numThreads)
  for (std::int32_t z = zBegin; z < zEnd; ++z) {
    auto const gz = static_cast<std::size_t>(globalBase[2] + z);

    for (std::int32_t y = yBegin; y < yEnd; ++y) {
      auto const gy = static_cast<std::size_t>(globalBase[1] + y);
      auto const gx = static_cast<std::size_t>(globalBase[0] + localBegin[0]);

      // Index arithmetic in std::size_t throughout: a volume of a few thousand voxels per axis
      // already exceeds what a 32 bit index can address.
      std::size_t dataIx1D = gz * this->ChunkingDataPtr->Dimensions[1];
      dataIx1D += gy;
      dataIx1D *= this->ChunkingDataPtr->Dimensions[0];
      dataIx1D += gx;

      auto localBeginPtr = &chunk.data[z][y][localBegin[0]];
      std::memcpy(localBeginPtr, this->ChunkingDataPtr->Data + dataIx1D, rowLength * sizeof(float));
    }
  }

  return chunk;
}

pme::Chunkifier::ChunkIterator& pme::Chunkifier::ChunkIterator::operator++() {
  ++this->CoreOriginsIterator;
  ++this->ValueRangesIterator;
  this->SkipCulledChunks();
  return *this;
}

bool pme::Chunkifier::ChunkIterator::Equals(ChunkIterator const& other) const {
  // The past-the-end sentinel carries no position, so any comparison involving it reduces to the
  // question of whether the other side still has chunks left. Comparing the underlying vector
  // iterators is only meaningful while both sides actually point into the collection.
  if (this->AtEnd() || other.AtEnd()) {
    return this->AtEnd() == other.AtEnd();
  }
  return this->CoreOriginsIterator == other.CoreOriginsIterator;
}

bool operator!=(pme::Chunkifier::ChunkIterator const& lhs, pme::Chunkifier::ChunkIterator const& rhs) {
  return !lhs.Equals(rhs);
}
