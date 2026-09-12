#include "CpuChunkMeshExtractor.h"

#include <algorithm>
#include <array>

#include "MarchingCubesTables.h"

namespace pme = parallel_mesh_extractor;
namespace mc  = parallel_mesh_extractor::marching_cubes;

namespace {

  constexpr int CellDim   = static_cast<int>(pme::CpuChunkMeshExtractor::CellDim);
  constexpr int CornerDim = static_cast<int>(pme::CpuChunkMeshExtractor::CornerDim);
  constexpr int GhostWidth = static_cast<int>(pme::Chunkifier::GhostWidth);

  // Same addressing as the device side: a corner coordinate addresses the chunk voxel one ghost
  // layer further in, and maps to the global voxel (CoreOrigin + corner).
  constexpr int ChunkShiftY = 6;
  constexpr int ChunkShiftZ = 12;

  inline int voxelIndex(int cx, int cy, int cz) {
    return ((cz + GhostWidth) << ChunkShiftZ) | ((cy + GhostWidth) << ChunkShiftY) | (cx + GhostWidth);
  }

  inline int cornerFlat(int cx, int cy, int cz) {
    return (cz * CornerDim + cy) * CornerDim + cx;
  }

  /// Step along each axis, indexed by the axis a grid edge runs along.
  constexpr int AxisStep[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

} // namespace

pme::CpuChunkMeshExtractor::CpuChunkMeshExtractor(float isoThreshold)
  : IsoThreshold(isoThreshold)
  , EdgeVertexIndex(static_cast<std::size_t>(3) * CornerDim * CornerDim * CornerDim, -1)
{}

void pme::CpuChunkMeshExtractor::Extract(Chunkifier::DataChunk const& chunk, ChunkMesh& out) {
  out.Clear();

  // Every edge starts out unassigned. Clearing the whole table is a 3 MB fill, which costs less
  // than tracking which entries were touched last time would.
  std::fill(this->EdgeVertexIndex.begin(), this->EdgeVertexIndex.end(), -1);

  float const* const data = &chunk.data[0][0][0];
  float const isoThreshold = this->IsoThreshold;

  for (int dz = 0; dz < CellDim; ++dz) {
    for (int dy = 0; dy < CellDim; ++dy) {
      for (int dx = 0; dx < CellDim; ++dx) {

        // A corner counts as inside when its value compares less than the isovalue, which also
        // makes a NaN count as outside, exactly as on the device.
        float value[8];
        std::uint32_t caseIndex = 0;
        for (int corner = 0; corner < 8; ++corner) {
          value[corner] = data[voxelIndex(dx + mc::CornerOffset[corner][0],
                                          dy + mc::CornerOffset[corner][1],
                                          dz + mc::CornerOffset[corner][2])];
          if (value[corner] < isoThreshold) caseIndex |= (1u << corner);
        }

        // The overwhelming majority of cells are entirely on one side of the surface, so this is
        // the branch that decides the runtime.
        if (caseIndex == 0 || caseIndex == 255) continue;

        auto const& row = mc::TriTable[caseIndex];
        for (int i = 0; i < 16 && row[i] >= 0; ++i) {
          auto const& owner = mc::EdgeToOwner[row[i]];

          int const ox = dx + owner.Offset[0];
          int const oy = dy + owner.Offset[1];
          int const oz = dz + owner.Offset[2];
          int const slot = 3 * cornerFlat(ox, oy, oz) + owner.Axis;

          std::int32_t index = this->EdgeVertexIndex[slot];
          if (index < 0) {
            // First triangle to want this edge places its vertex. Interpolating from the edge's
            // minimum corner towards its maximum is what makes the result agree bit for bit with
            // the device, and with whatever the neighbouring chunk computes for the same edge.
            float const value0 = data[voxelIndex(ox, oy, oz)];
            float const value1 = data[voxelIndex(ox + AxisStep[owner.Axis][0],
                                                 oy + AxisStep[owner.Axis][1],
                                                 oz + AxisStep[owner.Axis][2])];
            float const t = (isoThreshold - value0) / (value1 - value0);

            std::array<float, 3> position = {
              static_cast<float>(chunk.CoreOrigin[0] + ox),
              static_cast<float>(chunk.CoreOrigin[1] + oy),
              static_cast<float>(chunk.CoreOrigin[2] + oz),
            };
            position[owner.Axis] += t;

            index = static_cast<std::int32_t>(out.Positions.size());
            out.Positions.push_back(position);
            this->EdgeVertexIndex[slot] = index;
          }

          out.Indices.push_back(static_cast<std::uint32_t>(index));
        }
      }
    }
  }
}
