#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

#include "CudaChunkMeshExtractor.h"
#include "CudaUtils.h"
#include "MarchingCubesTables.h"

namespace pme = parallel_mesh_extractor;
namespace mc  = parallel_mesh_extractor::marching_cubes;

namespace {

  // ------------------------------------ the chunk layout --------------------------------------
  //
  // Everything below is expressed in "corner coordinates": the corner coordinate c addresses the
  // chunk voxel (c + GhostWidth), and maps to the global voxel coordinate (CoreOrigin + c). The
  // shift by the ghost width is absorbed once, here, so that no kernel has to carry it around.
  //
  // Corner coordinates run over [0, CornerDim), which is exactly the set of corners the cells a
  // chunk owns touch. The chunk's lowest voxel layer -- local index 0, the lower ghost shell -- is
  // consequently never addressed: with no gradients to evaluate, nothing needs it.

  constexpr int   ChunkDim      = static_cast<int>(pme::Chunkifier::ChunkSize);
  constexpr int   GhostWidth    = static_cast<int>(pme::Chunkifier::GhostWidth);
  constexpr int   CellDim       = static_cast<int>(pme::CudaChunkMeshExtractor::CellDim);
  constexpr int   CornerDim     = static_cast<int>(pme::CudaChunkMeshExtractor::CornerDim);

  constexpr int   NumCorners    = CornerDim * CornerDim * CornerDim;
  constexpr int   NumEdgeSlots  = 3 * NumCorners;
  constexpr int   NumCells      = CellDim * CellDim * CellDim;

  /// The chunk is a power of two per axis, so the voxel index is a shift and an or rather than a
  /// pair of multiplications.
  constexpr int ChunkShiftY = 6;   // log2(64)
  constexpr int ChunkShiftZ = 12;  // log2(64 * 64)

  __host__ __device__ inline int voxelIndex(int cx, int cy, int cz) {
    return ((cz + GhostWidth) << ChunkShiftZ) | ((cy + GhostWidth) << ChunkShiftY) | (cx + GhostWidth);
  }

  __host__ __device__ inline int cornerFlat(int cx, int cy, int cz) {
    return (cz * CornerDim + cy) * CornerDim + cx;
  }

  __host__ __device__ inline int cellFlat(int dx, int dy, int dz) {
    return (dz * CellDim + dy) * CellDim + dx;
  }

  // --------------------------------------- device tables --------------------------------------

  __constant__ std::int8_t  cTriTable[256][16];

  /// How far the edge slot of a cell edge sits from the edge slot of the cell's own minimum corner.
  ///
  /// A cell edge is owned by the grid edge starting at (cell base + offset) and running along a
  /// given axis. Because the offsets are 0 or 1 and the corner coordinates stay well inside
  /// CornerDim, that resolves to a *constant* displacement in the flat edge slot numbering, so the
  /// triangle kernel needs a single addition instead of reconstructing a 3D coordinate.
  __constant__ std::int32_t cEdgeSlotDelta[12];

  constexpr std::array<std::int32_t, 12> makeEdgeSlotDelta() {
    std::array<std::int32_t, 12> delta{};
    for (int edge = 0; edge < 12; ++edge) {
      auto const& owner = mc::EdgeToOwner[edge];
      delta[edge] = 3 * ((owner.Offset[2] * CornerDim + owner.Offset[1]) * CornerDim + owner.Offset[0])
                  + owner.Axis;
    }
    return delta;
  }

  /// How far each of the 8 cell corners sits from the cell's minimum corner in the flat voxel
  /// index. Constant for the same reason as the edge slot displacement above: the offsets are 0 or
  /// 1, and a cell's minimum corner is never so close to the upper end of the chunk that adding one
  /// could carry out of a coordinate's bit field. Derived from the corner table rather than written
  /// out again, so the two cannot drift apart.
  __constant__ std::int32_t cCornerVoxelDelta[8];

  constexpr std::array<std::int32_t, 8> makeCornerVoxelDelta() {
    std::array<std::int32_t, 8> delta{};
    for (int corner = 0; corner < 8; ++corner) {
      delta[corner] = (mc::CornerOffset[corner][2] << ChunkShiftZ)
                    + (mc::CornerOffset[corner][1] << ChunkShiftY)
                    +  mc::CornerOffset[corner][0];
    }
    return delta;
  }


  // ------------------------------------------ kernels -----------------------------------------

  /// Classifies, in one pass, both the grid edges and the cells of a chunk.
  ///
  /// One thread per grid corner. Each corner owns the three edges leaving it towards +x, +y and +z,
  /// and -- unless it sits on the upper boundary -- the cell it is the minimum corner of. Fusing
  /// the two is what lets the eight corner values be loaded once and used for both: the three edge
  /// neighbours are three of the eight cell corners.
  __global__ void kClassify(float const* __restrict__ chunk, float isoThreshold,
                            std::uint32_t* __restrict__ edgeActive,
                            std::uint8_t*  __restrict__ cellCase,
                            std::uint32_t* __restrict__ cellTriangleCount)
  {
    int const cx = blockIdx.x * blockDim.x + threadIdx.x;
    int const cy = blockIdx.y * blockDim.y + threadIdx.y;
    int const cz = blockIdx.z * blockDim.z + threadIdx.z;

    if (cx >= CornerDim || cy >= CornerDim || cz >= CornerDim) return;

    // A corner is the minimum corner of an owned cell unless it lies on the upper boundary of the
    // corner grid, where the cell would reach outside of what this chunk owns.
    bool const isCellOrigin = (cx < CellDim) && (cy < CellDim) && (cz < CellDim);

    // Corners 0, 1, 3 and 4 of the Marching Cubes numbering are the corner itself and its three
    // neighbours along +x, +y and +z, which is why the edge classification can reuse the values
    // loaded for the cell instead of fetching them a second time. Kept function local because a
    // namespace scope constexpr array has no address in device code.
    constexpr int AxisNeighbourCorner[3] = {1, 3, 4};

    int const limit[3] = {cx, cy, cz};
    int const voxelBase = voxelIndex(cx, cy, cz);

    float value[8];
    if (isCellOrigin) {
      #pragma unroll
      for (int corner = 0; corner < 8; ++corner) {
        value[corner] = chunk[voxelBase + cCornerVoxelDelta[corner]];
      }
    } else {
      // On the upper boundary of the corner grid the cell does not exist, and some of its corners
      // would lie outside. Only the three edge neighbours are needed here, each guarded on its own
      // axis.
      value[0] = chunk[voxelBase];
      #pragma unroll
      for (int axis = 0; axis < 3; ++axis) {
        int const corner = AxisNeighbourCorner[axis];
        value[corner] = ((limit[axis] + 1) < CornerDim)
                      ? chunk[voxelBase + cCornerVoxelDelta[corner]]
                      : value[0];
      }
    }

    // The sign convention of the whole extraction: a corner is inside when its value compares less
    // than the isovalue. Note that this makes a NaN count as outside, since every comparison
    // against a NaN is false, which keeps the classification total and deterministic.
    bool const inside0 = value[0] < isoThreshold;

    int const slotBase = 3 * cornerFlat(cx, cy, cz);

    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      // An edge leaving the last corner of an axis would leave the corner grid. It can never be
      // referenced by an owned cell, so flagging it inactive is enough -- and it keeps the slot
      // numbering a plain dense 3 * CornerDim^3, with no holes to special case elsewhere.
      bool const usable = (limit[axis] + 1) < CornerDim;
      bool const inside1 = value[AxisNeighbourCorner[axis]] < isoThreshold;

      edgeActive[slotBase + axis] = (usable && (inside0 != inside1)) ? 1u : 0u;
    }

    if (!isCellOrigin) return;

    std::uint32_t caseIndex = 0;
    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
      if (value[corner] < isoThreshold) caseIndex |= (1u << corner);
    }

    int const cell = cellFlat(cx, cy, cz);
    cellCase[cell] = static_cast<std::uint8_t>(caseIndex);

    int triangles = 0;
    for (int i = 0; i < 16 && cTriTable[caseIndex][i] >= 0; i += 3) ++triangles;
    cellTriangleCount[cell] = static_cast<std::uint32_t>(triangles);
  }

  /// Places the vertex of every active grid edge.
  ///
  /// One thread per edge slot, including the inactive ones: a thread that reads a zero flag and
  /// returns costs less than compacting the list would. The write offset comes from the prefix sum
  /// rather than from an atomic, which is what makes the output order -- and therefore the whole
  /// extraction -- reproducible.
  __global__ void kGenerateVertices(float const* __restrict__ chunk, float isoThreshold,
                                    std::uint32_t const* __restrict__ edgeActive,
                                    std::uint32_t const* __restrict__ edgeVertexIndex,
                                    long long coreOriginX, long long coreOriginY, long long coreOriginZ,
                                    float3* __restrict__ positions)
  {
    int const slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= NumEdgeSlots || edgeActive[slot] == 0u) return;

    int const axis    = slot % 3;
    int       corner  = slot / 3;
    int const cx      = corner % CornerDim; corner /= CornerDim;
    int const cy      = corner % CornerDim;
    int const cz      = corner / CornerDim;

    int const neighbour[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    float const value0 = chunk[voxelIndex(cx, cy, cz)];
    float const value1 = chunk[voxelIndex(cx + neighbour[axis][0],
                                          cy + neighbour[axis][1],
                                          cz + neighbour[axis][2])];

    // The edge was flagged active because the two values straddle the isovalue, so they differ and
    // the division is safe.
    float const t = (isoThreshold - value0) / (value1 - value0);

    // Global voxel coordinates. Both chunks that see a shared grid edge read the same two volume
    // voxels and run this same expression, so they produce bit identical positions and the
    // assembled mesh closes exactly.
    float position[3] = {
      static_cast<float>(coreOriginX + cx),
      static_cast<float>(coreOriginY + cy),
      static_cast<float>(coreOriginZ + cz),
    };
    position[axis] += t;

    positions[edgeVertexIndex[slot]] = make_float3(position[0], position[1], position[2]);
  }

  /// Emits the triangles of every active cell, as indices into the vertices placed above.
  ///
  /// One thread per *compacted* active cell, so that the threads of a warp all have table work to
  /// do instead of one of them carrying the block while the rest idle.
  __global__ void kGenerateTriangles(std::uint32_t const* __restrict__ activeCells,
                                     std::uint32_t const* __restrict__ numActiveCells,
                                     std::uint8_t  const* __restrict__ cellCase,
                                     std::uint32_t const* __restrict__ cellTriangleOffset,
                                     std::uint32_t const* __restrict__ edgeVertexIndex,
                                     std::uint32_t* __restrict__ indices)
  {
    // Bail out before touching shared memory: the grid is sized for the worst case, so on a
    // typical chunk the vast majority of blocks have nothing to do and should not pay for the
    // table load.
    std::uint32_t const activeCount = *numActiveCells;
    if (blockIdx.x * blockDim.x >= activeCount) return;

    // The case index differs from thread to thread, so the table is read divergently. Constant
    // memory would serialize that across the warp; shared memory does not.
    __shared__ std::int8_t sTriTable[256][16];
    for (int i = threadIdx.x; i < 256 * 16; i += blockDim.x) {
      sTriTable[i / 16][i % 16] = cTriTable[i / 16][i % 16];
    }
    __syncthreads();

    std::uint32_t const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= activeCount) return;

    std::uint32_t const cell = activeCells[index];

    int       remaining = static_cast<int>(cell);
    int const dx = remaining % CellDim; remaining /= CellDim;
    int const dy = remaining % CellDim;
    int const dz = remaining / CellDim;

    int const caseIndex = cellCase[cell];
    int const slotBase  = 3 * cornerFlat(dx, dy, dz);

    std::uint32_t out = 3u * cellTriangleOffset[cell];

    for (int i = 0; i < 16 && sTriTable[caseIndex][i] >= 0; ++i) {
      indices[out + i] = edgeVertexIndex[slotBase + cEdgeSlotDelta[sTriTable[caseIndex][i]]];
    }
  }

} // namespace

// ----------------------------------------- the extractor --------------------------------------

struct pme::CudaChunkMeshExtractor::Impl {
  float IsoThreshold = 0.0f;
  cudaStream_t Stream = nullptr;

  DeviceBuffer<float>         Chunk;
  DeviceBuffer<std::uint32_t> EdgeActive;
  DeviceBuffer<std::uint32_t> EdgeVertexIndex;
  DeviceBuffer<std::uint8_t>  CellCase;
  DeviceBuffer<std::uint32_t> CellTriangleCount;
  DeviceBuffer<std::uint32_t> CellTriangleOffset;
  DeviceBuffer<std::uint32_t> ActiveCells;
  DeviceBuffer<std::uint32_t> NumActiveCells;
  DeviceBuffer<float3>        Positions;
  DeviceBuffer<std::uint32_t> Indices;
  DeviceBuffer<unsigned char> Temporary;

  /// Vertex and triangle totals, read back before the variable sized payload can be sized.
  PinnedBuffer<std::uint32_t> Counts;

  /// Staging buffer for the upload, so that the copy can be asynchronous.
  PinnedBuffer<float> Staging;

  std::size_t TemporaryBytes = 0;
};

pme::CudaChunkMeshExtractor::CudaChunkMeshExtractor(float isoThreshold)
  : P(std::make_unique<Impl>())
{
  this->P->IsoThreshold = isoThreshold;

  CUDA_CHECK(cudaStreamCreate(&this->P->Stream));

  // The scans run over one element more than there is data, with that extra element held at zero,
  // so that the exclusive prefix sum deposits the grand total in it. That saves a separate
  // reduction just to learn how much was produced.
  this->P->Chunk              = DeviceBuffer<float>(ChunkDim * ChunkDim * ChunkDim);
  this->P->EdgeActive         = DeviceBuffer<std::uint32_t>(NumEdgeSlots + 1);
  this->P->EdgeVertexIndex    = DeviceBuffer<std::uint32_t>(NumEdgeSlots + 1);
  this->P->CellCase           = DeviceBuffer<std::uint8_t>(NumCells);
  this->P->CellTriangleCount  = DeviceBuffer<std::uint32_t>(NumCells + 1);
  this->P->CellTriangleOffset = DeviceBuffer<std::uint32_t>(NumCells + 1);
  this->P->ActiveCells        = DeviceBuffer<std::uint32_t>(NumCells);
  this->P->NumActiveCells     = DeviceBuffer<std::uint32_t>(1);
  this->P->Positions          = DeviceBuffer<float3>(MaxVertices);
  this->P->Indices            = DeviceBuffer<std::uint32_t>(3 * static_cast<std::size_t>(MaxTriangles));

  this->P->Counts  = PinnedBuffer<std::uint32_t>(2);
  this->P->Staging = PinnedBuffer<float>(ChunkDim * ChunkDim * ChunkDim);

  // The sentinel element of each scan input is written once and never touched again.
  CUDA_CHECK(cudaMemset(this->P->EdgeActive.Get() + NumEdgeSlots, 0, sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMemset(this->P->CellTriangleCount.Get() + NumCells, 0, sizeof(std::uint32_t)));

  // The upload skips the chunk's lowest voxel layer, which no kernel reads. Filling it with a NaN
  // pattern once means that a future indexing mistake reading it produces obvious garbage rather
  // than plausible looking noise.
  CUDA_CHECK(cudaMemset(this->P->Chunk.Get(), 0xFF, ChunkDim * ChunkDim * sizeof(float)));

  auto const edgeSlotDelta   = makeEdgeSlotDelta();
  auto const cornerVoxelDelta = makeCornerVoxelDelta();
  CUDA_CHECK(cudaMemcpyToSymbol(cTriTable, mc::TriTable.data(), sizeof(std::int8_t) * 256 * 16));
  CUDA_CHECK(cudaMemcpyToSymbol(cEdgeSlotDelta, edgeSlotDelta.data(), sizeof(std::int32_t) * 12));
  CUDA_CHECK(cudaMemcpyToSymbol(cCornerVoxelDelta, cornerVoxelDelta.data(), sizeof(std::int32_t) * 8));

  // Size the CUB scratch space for the largest of the three reductions and allocate it once, so
  // that no extraction ever has to allocate.
  std::size_t scanEdges = 0, scanCells = 0, select = 0;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanEdges, this->P->EdgeActive.Get(),
                                           this->P->EdgeVertexIndex.Get(), NumEdgeSlots + 1));
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanCells, this->P->CellTriangleCount.Get(),
                                           this->P->CellTriangleOffset.Get(), NumCells + 1));
  CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, select, cub::CountingInputIterator<std::uint32_t>(0),
                                        this->P->CellTriangleCount.Get(), this->P->ActiveCells.Get(),
                                        this->P->NumActiveCells.Get(), NumCells));

  this->P->TemporaryBytes = std::max(scanEdges, std::max(scanCells, select));
  this->P->Temporary = DeviceBuffer<unsigned char>(this->P->TemporaryBytes);
}

pme::CudaChunkMeshExtractor::~CudaChunkMeshExtractor() {
  if (this->P && this->P->Stream) cudaStreamDestroy(this->P->Stream);
}

pme::CudaChunkMeshExtractor::CudaChunkMeshExtractor(CudaChunkMeshExtractor&&) noexcept = default;
pme::CudaChunkMeshExtractor& pme::CudaChunkMeshExtractor::operator=(CudaChunkMeshExtractor&&) noexcept = default;

bool pme::CudaChunkMeshExtractor::IsCudaAvailable() {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
    // Clear the sticky error so that a later, legitimate call is not reported as having failed.
    cudaGetLastError();
    return false;
  }
  return deviceCount > 0;
}

void pme::CudaChunkMeshExtractor::Extract(Chunkifier::DataChunk const& chunk, ChunkMesh& out) {
  out.Clear();

  auto& impl = *this->P;
  auto const stream = impl.Stream;

  // The chunk's lowest voxel layer is the lower ghost shell, which nothing reads now that no
  // gradients are evaluated. It is contiguous, so skipping it is a pointer offset and a shorter
  // copy -- roughly 1.6 % off every upload in a pipeline that is bound by exactly this transfer.
  constexpr std::size_t skippedVoxels = static_cast<std::size_t>(ChunkDim) * ChunkDim;
  constexpr std::size_t uploadVoxels  = static_cast<std::size_t>(ChunkDim) * ChunkDim * ChunkDim - skippedVoxels;

  std::memcpy(impl.Staging.Get(), &chunk.data[1][0][0], uploadVoxels * sizeof(float));
  CUDA_CHECK(cudaMemcpyAsync(impl.Chunk.Get() + skippedVoxels, impl.Staging.Get(),
                             uploadVoxels * sizeof(float), cudaMemcpyHostToDevice, stream));

  {
    dim3 const block(8, 8, 4);
    dim3 const grid((CornerDim + block.x - 1) / block.x,
                    (CornerDim + block.y - 1) / block.y,
                    (CornerDim + block.z - 1) / block.z);
    kClassify<<<grid, block, 0, stream>>>(impl.Chunk.Get(), impl.IsoThreshold,
                                          impl.EdgeActive.Get(), impl.CellCase.Get(),
                                          impl.CellTriangleCount.Get());
    CUDA_CHECK_LAST_LAUNCH();
  }

  std::size_t temporaryBytes = impl.TemporaryBytes;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(impl.Temporary.Get(), temporaryBytes,
                                           impl.EdgeActive.Get(), impl.EdgeVertexIndex.Get(),
                                           NumEdgeSlots + 1, stream));

  temporaryBytes = impl.TemporaryBytes;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(impl.Temporary.Get(), temporaryBytes,
                                           impl.CellTriangleCount.Get(), impl.CellTriangleOffset.Get(),
                                           NumCells + 1, stream));

  temporaryBytes = impl.TemporaryBytes;
  CUDA_CHECK(cub::DeviceSelect::Flagged(impl.Temporary.Get(), temporaryBytes,
                                        cub::CountingInputIterator<std::uint32_t>(0),
                                        impl.CellTriangleCount.Get(), impl.ActiveCells.Get(),
                                        impl.NumActiveCells.Get(), NumCells, stream));

  {
    int const block = 256;
    int const grid  = (NumEdgeSlots + block - 1) / block;
    kGenerateVertices<<<grid, block, 0, stream>>>(impl.Chunk.Get(), impl.IsoThreshold,
                                                  impl.EdgeActive.Get(), impl.EdgeVertexIndex.Get(),
                                                  chunk.CoreOrigin[0], chunk.CoreOrigin[1],
                                                  chunk.CoreOrigin[2], impl.Positions.Get());
    CUDA_CHECK_LAST_LAUNCH();
  }

  {
    // Sized for the worst case rather than for the actual number of active cells, which would need
    // a device to host round trip first. The surplus blocks exit on their first instruction.
    int const block = 128;
    int const grid  = (NumCells + block - 1) / block;
    kGenerateTriangles<<<grid, block, 0, stream>>>(impl.ActiveCells.Get(), impl.NumActiveCells.Get(),
                                                   impl.CellCase.Get(), impl.CellTriangleOffset.Get(),
                                                   impl.EdgeVertexIndex.Get(), impl.Indices.Get());
    CUDA_CHECK_LAST_LAUNCH();
  }

  // The sentinel element of each scan now holds the grand total.
  CUDA_CHECK(cudaMemcpyAsync(impl.Counts.Get() + 0, impl.EdgeVertexIndex.Get() + NumEdgeSlots,
                             sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(impl.Counts.Get() + 1, impl.CellTriangleOffset.Get() + NumCells,
                             sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::uint32_t const numVertices  = impl.Counts.Get()[0];
  std::uint32_t const numTriangles = impl.Counts.Get()[1];

  // The buffers are sized for the worst case a chunk can possibly emit, so exceeding them would
  // mean the layout assumptions themselves are wrong rather than that the data is unusual.
  if (numVertices > MaxVertices || numTriangles > MaxTriangles) {
    throw CudaError("chunk extraction produced more geometry than the layout permits");
  }

  if (numTriangles == 0) return;

  out.Positions.resize(numVertices);
  out.Indices.resize(3u * static_cast<std::size_t>(numTriangles));

  CUDA_CHECK(cudaMemcpyAsync(out.Positions.data(), impl.Positions.Get(),
                             numVertices * sizeof(float3), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(out.Indices.data(), impl.Indices.Get(),
                             3u * static_cast<std::size_t>(numTriangles) * sizeof(std::uint32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}
