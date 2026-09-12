#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "../MeshExtraction/ChunkMesh.h"
#include "../MeshExtraction/MarchingCubesTables.h"

namespace reference_marching_cubes {

  /// A deliberately plain, single threaded Marching Cubes on the host, used as the oracle the GPU
  /// extraction is compared against.
  ///
  /// It shares the lookup tables with the extraction -- there is no point in transcribing those a
  /// second time, and the table tests check them on their own terms -- but nothing else. In
  /// particular it emits a triangle soup with no vertex sharing at all, so agreeing with it says
  /// something about the geometry and the case selection without the edge ownership scheme being
  /// able to hide a mistake inside a matching one.
  ///
  /// The interpolation is written to match the kernel's arithmetic exactly: the parameter is always
  /// computed from the edge's *minimum* corner towards its maximum, so that the two produce bit
  /// identical positions and can be compared without a tolerance.
  ///
  /// "sample" is called with global voxel coordinates and may be asked for coordinates outside the
  /// volume, which is how the background padding is modelled. Cells are visited with their minimum
  /// corner running over [cellBegin, cellEnd) on each axis.
  template<class Sampler>
  parallel_mesh_extractor::ChunkMesh extract(Sampler const& sample,
                                             std::array<std::int64_t, 3> const& cellBegin,
                                             std::array<std::int64_t, 3> const& cellEnd,
                                             float isoThreshold) {
    namespace mc = parallel_mesh_extractor::marching_cubes;
    parallel_mesh_extractor::ChunkMesh mesh;

    for (std::int64_t z = cellBegin[2]; z < cellEnd[2]; ++z) {
      for (std::int64_t y = cellBegin[1]; y < cellEnd[1]; ++y) {
        for (std::int64_t x = cellBegin[0]; x < cellEnd[0]; ++x) {

          float value[8];
          std::uint32_t caseIndex = 0;
          for (int corner = 0; corner < 8; ++corner) {
            value[corner] = sample(x + mc::CornerOffset[corner][0],
                                   y + mc::CornerOffset[corner][1],
                                   z + mc::CornerOffset[corner][2]);
            if (value[corner] < isoThreshold) caseIndex |= (1u << corner);
          }

          if (caseIndex == 0 || caseIndex == 255) continue;

          auto const& row = mc::TriTable[caseIndex];
          for (int i = 0; i < 16 && row[i] >= 0; ++i) {
            auto const& owner = mc::EdgeToOwner[row[i]];

            // The edge runs from the corner at "owner.Offset" one step along "owner.Axis". Find the
            // Marching Cubes corner numbers of those two ends, so that the already sampled values
            // can be reused.
            int minimumCorner = -1, maximumCorner = -1;
            for (int corner = 0; corner < 8; ++corner) {
              bool matchesMinimum = true, matchesMaximum = true;
              for (int axis = 0; axis < 3; ++axis) {
                int const expectedMinimum = owner.Offset[axis];
                int const expectedMaximum = owner.Offset[axis] + (axis == owner.Axis ? 1 : 0);
                matchesMinimum &= (mc::CornerOffset[corner][axis] == expectedMinimum);
                matchesMaximum &= (mc::CornerOffset[corner][axis] == expectedMaximum);
              }
              if (matchesMinimum) minimumCorner = corner;
              if (matchesMaximum) maximumCorner = corner;
            }

            float const value0 = value[minimumCorner];
            float const value1 = value[maximumCorner];
            float const t = (isoThreshold - value0) / (value1 - value0);

            std::array<float, 3> position = {
              static_cast<float>(x + owner.Offset[0]),
              static_cast<float>(y + owner.Offset[1]),
              static_cast<float>(z + owner.Offset[2]),
            };
            position[owner.Axis] += t;

            mesh.Indices.push_back(static_cast<std::uint32_t>(mesh.Positions.size()));
            mesh.Positions.push_back(position);
          }
        }
      }
    }

    return mesh;
  }

} // namespace reference_marching_cubes
