#include "Chunkifier.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <omp.h>

namespace pme = parallel_mesh_extractor;

// --------------------------------------- Chunkifier ------------------------------------------ //

pme::Chunkifier::Chunkifier(float const* data, std::array<std::uint32_t, 3> const& dim,
                            float backgroundValue) noexcept
  : Data(data)
  , Dimensions(dim)
  , BackgroundValue(backgroundValue)
{
}

pme::Chunkifier::ChunkIterator pme::Chunkifier::begin() const {
  ChunkIterator it(this->Data, this->BackgroundValue);
  it.ComputeChunkingOffsets(this->Dimensions);
  return it;
}

pme::Chunkifier::ChunkIterator pme::Chunkifier::end() const {
  return ChunkIterator(nullptr, this->BackgroundValue);
}

// -------------------------------------- ChunkIterator ---------------------------------------- //

pme::Chunkifier::ChunkIterator::ChunkIterator(float const* data, float backgroundValue) noexcept
  : ChunkIndex(data == nullptr ? -1 : 0)
  , Data(data)
  , BackgroundValue(backgroundValue)
{
}

void pme::Chunkifier::ChunkIterator::ComputeChunkingOffsets(std::array<std::uint32_t, 3> const& dim) {
  this->DataDims = dim;

  // Compute the number of chunks.
  // Since the chunks only own their inner core and the ghost shell is pure overlap, consecutive
  // chunks advance by CoreSize rather than by the allocated ChunkSize.
  std::uint32_t const numTilesZ = (dim[2] + Chunkifier::CoreSize - 1) / Chunkifier::CoreSize;
  std::uint32_t const numTilesY = (dim[1] + Chunkifier::CoreSize - 1) / Chunkifier::CoreSize;
  std::uint32_t const numTilesX = (dim[0] + Chunkifier::CoreSize - 1) / Chunkifier::CoreSize;

  // Construct the list of core origins, i.e., the global coordinate of the first voxel each chunk
  // owns. The ghost shell of a chunk then covers the coordinates [origin-GhostWidth, origin) and
  // [origin+CoreSize, origin+CoreSize+GhostWidth).
  this->CoreOrigins.resize(static_cast<std::size_t>(numTilesX) * numTilesY * numTilesZ);
  std::size_t tileIx1D = 0;

  for (std::uint32_t tileZ = 0; tileZ < numTilesZ; ++tileZ) {
    for (std::uint32_t tileY = 0; tileY < numTilesY; ++tileY) {
      for (std::uint32_t tileX = 0; tileX < numTilesX; ++tileX) {
        this->CoreOrigins[tileIx1D++] = {
          tileX * Chunkifier::CoreSize,
          tileY * Chunkifier::CoreSize,
          tileZ * Chunkifier::CoreSize,
        };
      }
    }
  }

  // An empty volume yields no chunks at all, in which case this iterator has to compare equal to
  // the past-the-end iterator right away.
  this->ChunkIndex = this->CoreOrigins.empty()
                   ? -1
                   : static_cast<std::int32_t>(this->CoreOrigins.size()) - 1;
}

pme::Chunkifier::DataChunk pme::Chunkifier::ChunkIterator::operator*() const {
  // Get the core origin of the current chunk.
  assert(this->ChunkIndex >= 0);
  auto const coreOrigin = this->CoreOrigins[this->CoreOrigins.size() - 1 - this->ChunkIndex];

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
    localEnd[axis]   = std::min<std::uint32_t>(Chunkifier::ChunkSize, this->DataDims[axis] - origin + Chunkifier::GhostWidth);
    coversFullChunk &= (localBegin[axis] == 0 && localEnd[axis] == Chunkifier::ChunkSize);
  }

  // Everything outside of that region reaches across the volume boundary and receives the
  // background value instead. Chunks lying fully inside the volume are overwritten completely by
  // the copy below and can skip this.
  if (!coversFullChunk) {
    chunk.Fill(this->BackgroundValue);
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
      std::size_t dataIx1D = gz * this->DataDims[1];
      dataIx1D += gy;
      dataIx1D *= this->DataDims[0];
      dataIx1D += gx;

      std::memcpy(&chunk.data[z][y][localBegin[0]], this->Data + dataIx1D, rowLength * sizeof(float));
    }
  }

  return chunk;
}

pme::Chunkifier::ChunkIterator& pme::Chunkifier::ChunkIterator::operator++() {
  // In our logic, we do only know the number of chunks from the ComputeChunkingOffsets function.
  // Consequently, the past-the-end iterator constructed from null data does not know this.
  // Therefore, we instead internally decrement the chunk index pointer upon incrementation, and
  // we instead iterate until this chunk index (beginning from number of chunks minus one), reaches 0.
  --this->ChunkIndex;
  return *this;
}

std::int32_t pme::Chunkifier::ChunkIterator::GetChunkIndex(void) const {
  return this->ChunkIndex;
}

bool operator!=(pme::Chunkifier::ChunkIterator const& lhs, pme::Chunkifier::ChunkIterator const& rhs) {
  return lhs.GetChunkIndex() != rhs.GetChunkIndex();
}
