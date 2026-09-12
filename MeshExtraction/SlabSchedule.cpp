#include "SlabSchedule.h"

#include <algorithm>
#include "Chunkifier.h"

namespace pme = parallel_mesh_extractor;

std::vector<pme::SlabWindow> pme::PlanSlabs(std::array<std::uint32_t, 3> const& dim,
                                            std::uint32_t layersPerSlab) {
  std::vector<SlabWindow> plan;
  if (dim[2] == 0) return plan;

  layersPerSlab = std::max<std::uint32_t>(layersPerSlab, 1);

  constexpr std::int64_t coreSize   = Chunkifier::CoreSize;
  constexpr std::int64_t ghostWidth = Chunkifier::GhostWidth;
  constexpr std::int64_t chunkSize  = Chunkifier::ChunkSize;

  std::int64_t const numLayers = (static_cast<std::int64_t>(dim[2]) + coreSize - 1) / coreSize;
  std::int64_t const depth     = dim[2];

  for (std::int64_t first = 0; first < numLayers; first += layersPerSlab) {
    std::int64_t const count = std::min<std::int64_t>(layersPerSlab, numLayers - first);
    std::int64_t const last  = first + count - 1;

    // The first layer of the run reaches one voxel below its core origin, and the last reaches to
    // the far side of its own chunk. Clamping both ends to the volume is what keeps the slab from
    // asking for slices that do not exist; the chunks pad those with the background themselves.
    std::int64_t const spanBegin = std::max<std::int64_t>(first * coreSize - ghostWidth, 0);
    std::int64_t const spanEnd   = std::min<std::int64_t>(last * coreSize - ghostWidth + chunkSize, depth);

    SlabWindow window;
    window.ZBegin     = spanBegin;
    window.ZCount     = static_cast<std::uint32_t>(spanEnd - spanBegin);
    window.TileZBegin = first;
    window.TileZCount = count;

    // The plane of grid corners shared with the next layer up. Vertices sitting exactly on it are
    // still reachable, everything below it is not.
    window.PruneBelowZ = static_cast<float>((last + 1) * coreSize);

    plan.push_back(window);
  }

  return plan;
}

std::uint32_t pme::MaxSlabSliceCount(std::array<std::uint32_t, 3> const& dim,
                                     std::uint32_t layersPerSlab) {
  std::uint32_t largest = 0;
  for (auto const& window : PlanSlabs(dim, layersPerSlab)) {
    largest = std::max(largest, window.ZCount);
  }
  return largest;
}
