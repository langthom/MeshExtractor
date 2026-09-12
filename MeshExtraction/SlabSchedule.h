#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace parallel_mesh_extractor {

  /// One step of a streaming extraction: the slices to have in memory, and the chunk layers that
  /// can be produced from them.
  ///
  /// The slice range is already clamped to the volume. A chunk reaches one voxel below its core
  /// origin, but for the very first layer that voxel lies outside the volume and is background,
  /// which the chunk supplies for itself -- so a slab never has to hold a slice that does not
  /// exist.
  struct SlabWindow {
    /// Global z of the first slice the slab holds, and how many slices follow it.
    std::int64_t  ZBegin = 0;
    std::uint32_t ZCount = 0;

    /// The chunk layers this slab serves. Every layer of the volume belongs to exactly one slab.
    std::int64_t TileZBegin = 0;
    std::int64_t TileZCount = 0;

    /// Voxel z below which a welded vertex can never be referenced again once this slab is done.
    ///
    /// Two chunk layers share exactly one plane of grid corners, the one at the core origin of the
    /// upper layer. Everything this slab emitted below that plane is out of reach of every layer
    /// still to come, so the welding cache can drop it and stay bounded.
    float PruneBelowZ = 0.0f;
  };

  /// Lay out the slabs covering a volume, "layersPerSlab" chunk layers at a time.
  ///
  /// Consecutive slabs overlap by 2 * GhostWidth slices, which is what lets each layer be served
  /// whole by exactly one of them. A larger "layersPerSlab" trades memory for fewer, larger reads;
  /// a value of zero is treated as one. A volume with no slices yields no slabs.
  std::vector<SlabWindow> PlanSlabs(std::array<std::uint32_t, 3> const& dim,
                                    std::uint32_t layersPerSlab);

  /// The largest slice count any slab of that plan asks for, so that the slab buffer can be
  /// allocated once up front rather than grown per slab.
  std::uint32_t MaxSlabSliceCount(std::array<std::uint32_t, 3> const& dim,
                                  std::uint32_t layersPerSlab);

} // namespace parallel_mesh_extractor
