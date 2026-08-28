#include "Chunkifier.h"
#include <cassert>
#include <cstdint>
#include <omp.h>

namespace pme = parallel_mesh_extractor;

// --------------------------------------- Chunkifier ------------------------------------------ //

pme::Chunkifier::Chunkifier(float const* data, std::array<std::uint32_t, 3> const& dim) noexcept
  : Data(data)
  , Dimensions(dim)
{
}

pme::Chunkifier::ChunkIterator pme::Chunkifier::begin() const {
  ChunkIterator it(this->Data);
  it.ComputeChunkingOffsets(this->Dimensions);
  return it;
}

pme::Chunkifier::ChunkIterator pme::Chunkifier::end() const {
  return ChunkIterator(nullptr);
}

// -------------------------------------- ChunkIterator ---------------------------------------- //

pme::Chunkifier::ChunkIterator::ChunkIterator(float const* data) noexcept
  : Data(data)
  , ChunkIndex(data == nullptr ? -1 : 0)
{
}

void pme::Chunkifier::ChunkIterator::ComputeChunkingOffsets(std::array<std::uint32_t, 3> const& dim) {
  this->DataDims = dim;

  // Compute the number of Rois.
  std::uint32_t const numTilesZ = (dim[2] + Chunkifier::ChunkSize - 1) / Chunkifier::ChunkSize;
  std::uint32_t const numTilesY = (dim[1] + Chunkifier::ChunkSize - 1) / Chunkifier::ChunkSize;
  std::uint32_t const numTilesX = (dim[0] + Chunkifier::ChunkSize - 1) / Chunkifier::ChunkSize;

  // Construct the list of origins.
  // The origins of the chunks are always aligned on our chunk size, so the computation is 
  // particularly simple :)
  this->ChunkOrigins.resize(numTilesX * numTilesY * numTilesZ);
  int tileIx1D = 0;

  for (std::uint32_t tileZ = 0; tileZ < numTilesZ; ++tileZ) {
    for (std::uint32_t tileY = 0; tileY < numTilesY; ++tileY) {
      for (std::uint32_t tileX = 0; tileX < numTilesX; ++tileX) {
        this->ChunkOrigins[tileIx1D++] = {
          tileX * Chunkifier::ChunkSize,
          tileY * Chunkifier::ChunkSize,
          tileZ * Chunkifier::ChunkSize,
        };
      }
    }
  }

  this->ChunkIndex = this->ChunkOrigins.size() - 1;
}

pme::Chunkifier::DataChunk pme::Chunkifier::ChunkIterator::operator*() const {
  // Get the origin of the current chunk.
  assert(this->ChunkIndex >= 0);
  auto const [ox, oy, oz] = this->ChunkOrigins[this->ChunkOrigins.size()-1-this->ChunkIndex];
  
  // Construct the target chunk.
  DataChunk chunk;

  // Iterate over the data and extract the chunk by copying.
  // Since the source data is also assumed to be in-memory, we simply do the 3D loop to copy the
  // data to its correct location in the chunk. As the chunk size is typically rather small, we 
  // parallelize only the outermost dimension to avoid a huge thread overhead.

  // Calculate the number of slices for the current dimension. This is either the chunk size for
  // fully filled tiles, or the remaining slices (from the computed origin to the end of the domain).
  auto numSlicesForDim = [&](std::uint32_t origin, int axis) -> std::uint32_t {
    return std::min<std::uint32_t>(DataChunk::ChunkDimSize, this->DataDims[axis] - origin);
  };

  std::int32_t const numZSlices = numSlicesForDim(oz, 2);
  std::int32_t const numYSlices = numSlicesForDim(oy, 1);
  std::int32_t const numXSlices = numSlicesForDim(ox, 0);

  // Compute how many threads to use.
  // In case of a fixed defined OMP_NUM_THREADS, use that. Otherwise, use the available number of
  // threads which might be reduced depending on the actual workload (number of Z axis slices).
  int numThreads = 1;
  #ifdef OMP_NUM_THREADS
    numThreads = OMP_NUM_THREADS;
  #else
    #pragma omp parallel
    {
      numThreads = std::min(omp_get_num_threads(), numZSlices);
    }
  #endif

  // Copy the data to the allocated chunk. Since both memory regions live in main memory, a simple
  // copy loop suffices. The outermost loop is parallelized, yet including the innermost loops would
  // likely destroy cache coherence and introduce too much thread creation overhead.
  #pragma omp parallel for schedule(static, 8) num_threads(numThreads)
  for (std::int32_t z = 0; z < numZSlices; ++z) {
    for (std::int32_t y = 0; y < numYSlices; ++y) {
      for (std::int32_t x = 0; x < numXSlices; ++x) {
        std::size_t dataIx1D = (z + oz) * this->DataDims[1];
        dataIx1D += y + oy;
        dataIx1D *= this->DataDims[0];
        dataIx1D += x + ox;

        chunk.data[z][y][x] = this->Data[dataIx1D];
      }
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



