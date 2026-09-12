
#include <array>
#include <cstdint>
#include <set>
#include <vector>

#include "doctest.h"
#include "../MeshExtraction/Chunkifier.h"
#include "../MeshExtraction/SlabSchedule.h"

namespace pme = parallel_mesh_extractor;

namespace {

  // Pinned with literals rather than derived from the production constants, as elsewhere in the
  // suite, so that moving the chunk layout surfaces here.
  constexpr std::int64_t ChunkSize  = 64;
  constexpr std::int64_t GhostWidth = 1;
  constexpr std::int64_t CoreSize   = ChunkSize - 2 * GhostWidth;

  using Dims = std::array<std::uint32_t, 3>;

  std::int64_t layerCount(Dims const& dims) {
    return (static_cast<std::int64_t>(dims[2]) + CoreSize - 1) / CoreSize;
  }

  /// The depths worth exercising: around the first chunk, around exact multiples of the stride, and
  /// one either side of each.
  std::vector<std::uint32_t> interestingDepths() {
    std::vector<std::uint32_t> depths = {1, 2, 61, 62, 63, 64, 65, 123, 124, 125, 185, 186, 187, 200, 500};
    return depths;
  }

} // namespace

TEST_CASE("A volume with no slices yields no slabs") {
  CHECK(pme::PlanSlabs({16, 16, 0}, 1).empty());
  CHECK(pme::MaxSlabSliceCount({16, 16, 0}, 1) == 0);
}

TEST_CASE("Every chunk layer is served by exactly one slab") {
  // The whole streaming argument rests on this: a layer served twice would duplicate geometry, and
  // one served by no slab would leave a hole.
  for (auto const depth : interestingDepths()) {
    for (std::uint32_t layersPerSlab : {1u, 2u, 3u, 7u, 1000u}) {
      Dims const dims = {16, 16, depth};
      INFO("depth ", depth, ", ", layersPerSlab, " layers per slab");

      auto const plan = pme::PlanSlabs(dims, layersPerSlab);
      REQUIRE_FALSE(plan.empty());

      std::vector<std::int64_t> served;
      for (auto const& window : plan) {
        for (std::int64_t i = 0; i < window.TileZCount; ++i) served.push_back(window.TileZBegin + i);
      }

      std::vector<std::int64_t> expected;
      for (std::int64_t layer = 0; layer < layerCount(dims); ++layer) expected.push_back(layer);

      CHECK(served == expected);
    }
  }
}

TEST_CASE("A slab holds every slice the layers it claims will need") {
  // Cross checked against the very rule the chunkifier applies, rather than against a second copy
  // of the arithmetic: a slab that claims a layer it cannot serve would hand out chunks filled
  // from slices it never held.
  //
  // Only the claimed layers are asserted. A slab may well also be *capable* of the layer above it,
  // when that layer is the last of the volume and needs only a couple of slices -- which is why
  // ownership is assigned here rather than inferred from capability.
  for (auto const depth : interestingDepths()) {
    for (std::uint32_t layersPerSlab : {1u, 2u, 5u}) {
      Dims const dims = {16, 16, depth};
      auto const plan = pme::PlanSlabs(dims, layersPerSlab);

      for (auto const& window : plan) {
        INFO("depth ", depth, ", ", layersPerSlab, " per slab, slab at z ", window.ZBegin,
             " count ", window.ZCount, " serving layers [", window.TileZBegin, ", ",
             window.TileZBegin + window.TileZCount, ")");

        std::int64_t const slabEnd = window.ZBegin + window.ZCount;

        for (std::int64_t i = 0; i < window.TileZCount; ++i) {
          INFO("claimed layer ", window.TileZBegin + i);
          CHECK(pme::Chunkifier::CoversLayer(window.TileZBegin + i, dims, window.ZBegin, slabEnd));
        }

        // And it must not be able to serve anything below what it claims, which would mean the
        // slab started earlier than the plan says and the layers below were read twice.
        for (std::int64_t layer = 0; layer < window.TileZBegin; ++layer) {
          INFO("layer ", layer, ", below the claim");
          CHECK_FALSE(pme::Chunkifier::CoversLayer(layer, dims, window.ZBegin, slabEnd));
        }
      }
    }
  }
}

TEST_CASE("Slabs stay inside the volume and overlap by the ghost shell") {
  for (auto const depth : interestingDepths()) {
    Dims const dims = {16, 16, depth};
    auto const plan = pme::PlanSlabs(dims, 1);

    for (std::size_t i = 0; i < plan.size(); ++i) {
      INFO("depth ", depth, ", slab ", i);
      CHECK(plan[i].ZBegin >= 0);
      CHECK(plan[i].ZBegin + plan[i].ZCount <= depth);
      CHECK(plan[i].ZCount > 0);

      if (i == 0) continue;

      // Consecutive slabs share the two slices the ghost shells of the layers on either side of
      // the interface both need -- unless the volume ran out first.
      std::int64_t const previousEnd = plan[i - 1].ZBegin + plan[i - 1].ZCount;
      std::int64_t const overlap = previousEnd - plan[i].ZBegin;
      INFO("overlap with the previous slab: ", overlap);
      CHECK(overlap <= 2 * GhostWidth);
      CHECK(overlap >= 0);
    }
  }
}

TEST_CASE("One slab per layer asks for at most a chunk's worth of slices") {
  for (auto const depth : interestingDepths()) {
    Dims const dims = {16, 16, depth};
    for (auto const& window : pme::PlanSlabs(dims, 1)) {
      INFO("depth ", depth, ", slab at z ", window.ZBegin);
      CHECK(window.ZCount <= ChunkSize);
    }
    CHECK(pme::MaxSlabSliceCount(dims, 1) <= ChunkSize);
  }
}

TEST_CASE("Asking for more layers than the volume has gives a single whole volume slab") {
  // The degenerate case the non streaming path is: one slab spanning everything.
  for (auto const depth : interestingDepths()) {
    Dims const dims = {16, 16, depth};
    auto const plan = pme::PlanSlabs(dims, 1000);

    INFO("depth ", depth);
    REQUIRE(plan.size() == 1);
    CHECK(plan[0].ZBegin == 0);
    CHECK(plan[0].ZCount == depth);
    CHECK(plan[0].TileZBegin == 0);
    CHECK(plan[0].TileZCount == layerCount(dims));
  }
}

TEST_CASE("Zero layers per slab is treated as one") {
  // A caller passing zero would otherwise advance the loop by nothing and never terminate. It is a
  // caller mistake, but a silent hang is a bad way to report it.
  Dims const dims = {16, 16, 200};

  auto const fromZero = pme::PlanSlabs(dims, 0);
  auto const fromOne  = pme::PlanSlabs(dims, 1);

  REQUIRE(fromZero.size() == fromOne.size());
  for (std::size_t i = 0; i < fromOne.size(); ++i) {
    INFO("slab ", i);
    CHECK(fromZero[i].ZBegin     == fromOne[i].ZBegin);
    CHECK(fromZero[i].ZCount     == fromOne[i].ZCount);
    CHECK(fromZero[i].TileZBegin == fromOne[i].TileZBegin);
    CHECK(fromZero[i].TileZCount == fromOne[i].TileZCount);
  }
}

TEST_CASE("The prune threshold is the corner plane shared with the next layer up") {
  // Two chunk layers share exactly one plane of grid corners. A vertex sitting on it can still be
  // referenced by the layer above, so the threshold has to be the plane itself and the comparison
  // that uses it has to be strict.
  Dims const dims = {16, 16, 500};

  for (std::uint32_t layersPerSlab : {1u, 3u}) {
    auto const plan = pme::PlanSlabs(dims, layersPerSlab);
    for (auto const& window : plan) {
      std::int64_t const lastLayer = window.TileZBegin + window.TileZCount - 1;
      INFO(layersPerSlab, " layers per slab, last layer ", lastLayer);
      CHECK(window.PruneBelowZ == static_cast<float>((lastLayer + 1) * CoreSize));
    }

    // The corners a layer touches run from its core origin to one core further, so the threshold
    // of a slab must not exceed the lowest corner the next slab will produce.
    for (std::size_t i = 0; i + 1 < plan.size(); ++i) {
      INFO("slab ", i, " prunes below ", plan[i].PruneBelowZ,
           ", next slab starts at layer ", plan[i + 1].TileZBegin);
      CHECK(plan[i].PruneBelowZ <= static_cast<float>(plan[i + 1].TileZBegin * CoreSize));
    }
  }
}

TEST_CASE("More layers per slab means more slices but fewer slabs") {
  Dims const dims = {16, 16, 500};

  auto const one   = pme::PlanSlabs(dims, 1);
  auto const three = pme::PlanSlabs(dims, 3);

  CHECK(three.size() < one.size());
  CHECK(pme::MaxSlabSliceCount(dims, 3) > pme::MaxSlabSliceCount(dims, 1));

  // A run of L layers spans one whole chunk plus the stride of the L-1 layers above it. The very
  // first slab is one slice shorter, because the voxel below the volume is background rather than
  // something it has to hold.
  CHECK(pme::MaxSlabSliceCount(dims, 1) == ChunkSize);
  CHECK(pme::MaxSlabSliceCount(dims, 3) == 2 * CoreSize + ChunkSize);
  CHECK(pme::PlanSlabs(dims, 1).front().ZCount == ChunkSize - GhostWidth);
}
