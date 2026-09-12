
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include "ExtractionPipeline.h"

namespace pme = parallel_mesh_extractor;

namespace {

  void printUsage(char const* progName) {
    std::cerr << "Usage: " << progName << " <input.mhd> <output.ply> <iso>\n"
              << "       " << std::string(std::strlen(progName), ' ')
              << " [--backend cpu|gpu] [--threads <n>] [--background <value>]\n"
              << "       " << std::string(std::strlen(progName), ' ')
              << " [--slab-layers <n>] [--face-buffer <MiB>] [--voxel-coords] [--ascii]\n"
              << "\n"
              << "  input.mhd    Input volume, *.mhd format\n"
              << "  output.ply   Output mesh file, *.ply format\n"
              << "  iso          ISO threshold for surface extraction.\n"
              << "\n"
              << "Options:\n"
              << "  --backend cpu|gpu     Where the surface extraction runs. Both produce the same\n"
              << "                        surface. A chunk is too small to cover the launch and\n"
              << "                        transfer latency of a kernel, so on a machine with many\n"
              << "                        cores the CPU is the faster of the two and needs no CUDA.\n"
              << "                        Default cpu.\n"
              << "  --threads <n>         Threads for the CPU backend. Default: all of them.\n"
              << "  --background <value>  Value sampled outside the volume. It is what closes the\n"
              << "                        surface along the volume wall, so it belongs on the empty\n"
              << "                        side of the isovalue. Default 0.\n"
              << "  --slab-layers <n>     Chunk layers held in memory at once. Larger values trade\n"
              << "                        memory for fewer, larger reads. Default 1.\n"
              << "  --face-buffer <MiB>   Faces held in memory before spilling to a temporary file.\n"
              << "                        Spilling triples the face I/O, so this wants to be larger\n"
              << "                        than the mesh. Default 1024.\n"
              << "  --voxel-coords        Write voxel indices instead of world coordinates, i.e.\n"
              << "                        apply neither the spacing nor the origin.\n"
              << "  --ascii               Write an ASCII PLY instead of a binary one.\n";
  }

  /// Parse a number, reporting the offending text rather than a generic failure.
  template<class T>
  T parseNumber(std::string_view text, char const* what) {
    T value{};
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
      throw std::runtime_error("invalid " + std::string(what) + ": '" + std::string(text) + "'");
    }
    return value;
  }

  template<class T>
  std::string join(std::array<T, 3> const& values) {
    std::ostringstream out;
    out << values[0] << " x " << values[1] << " x " << values[2];
    return out.str();
  }

  std::string humanBytes(std::uint64_t bytes) {
    static char const* const units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double scaled = static_cast<double>(bytes);
    int unit = 0;
    while (scaled >= 1024.0 && unit < 4) { scaled /= 1024.0; ++unit; }

    std::ostringstream out;
    out << std::fixed << std::setprecision(scaled < 10.0 ? 1 : 0) << scaled << " " << units[unit];
    return out.str();
  }

} // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }

  try {
    std::filesystem::path const input = argv[1];
    std::filesystem::path const output = argv[2];

    pme::ExtractionSettings settings;
    settings.IsoThreshold = parseNumber<float>(argv[3], "ISO threshold");

    for (int i = 4; i < argc; ++i) {
      std::string_view const option = argv[i];

      auto const requireValue = [&](char const* what) -> std::string_view {
        if (i + 1 >= argc) throw std::runtime_error(std::string(what) + " needs a value");
        return argv[++i];
      };

      if (option == "--backend") {
        auto const name = requireValue("--backend");
        if (name == "cpu")      settings.Backend = pme::ExtractionBackend::Cpu;
        else if (name == "gpu") settings.Backend = pme::ExtractionBackend::Gpu;
        else throw std::runtime_error("--backend must be 'cpu' or 'gpu', not '" + std::string(name) + "'");
      } else if (option == "--threads") {
        settings.ExtractionThreads = parseNumber<std::uint32_t>(requireValue("--threads"), "thread count");
        if (settings.ExtractionThreads == 0) throw std::runtime_error("--threads must be at least 1");
      } else if (option == "--face-buffer") {
        auto const mib = parseNumber<std::uint64_t>(requireValue("--face-buffer"), "face buffer size");
        settings.FaceBufferBytes = static_cast<std::size_t>(mib) * 1024ull * 1024ull;
      } else if (option == "--background") {
        settings.BackgroundValue = parseNumber<float>(requireValue("--background"), "background value");
      } else if (option == "--slab-layers") {
        settings.LayersPerSlab = parseNumber<std::uint32_t>(requireValue("--slab-layers"), "slab layer count");
        if (settings.LayersPerSlab == 0) throw std::runtime_error("--slab-layers must be at least 1");
      } else if (option == "--voxel-coords") {
        settings.VoxelCoordinates = true;
      } else if (option == "--ascii") {
        settings.Format = pme::PLYFormat::ASCII;
      } else {
        throw std::runtime_error("unknown option: '" + std::string(option) + "'");
      }
    }

    if (settings.Backend == pme::ExtractionBackend::Gpu
        && !pme::CudaChunkMeshExtractor::IsCudaAvailable()) {
      throw std::runtime_error("no CUDA device is available; rerun with --backend cpu");
    }

    auto const started = std::chrono::steady_clock::now();
    std::cout << "Reading " << input << "\n";

    auto const report = pme::ExtractVolumeToPLY(input, output, settings, &std::cout);

    auto const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started);

    std::cout << "\nVolume    " << join(report.Dimensions) << " voxels\n"
              << "Spacing   " << join(report.Spacing) << "\n"
              << "Origin    " << join(report.Origin) << "\n"
              << "Isovalue  " << settings.IsoThreshold
              << " (background " << settings.BackgroundValue << ")\n"
              << "Slabs     " << report.Slabs << ", holding " << report.SlabSlices
              << " slices at a time (" << humanBytes(report.SlabBytes) << ")\n"
              << "Chunks    " << report.Chunks << " survived the culling, "
              << report.ChunksExtracted << " produced geometry\n"
              << "Mesh      " << report.Vertices << " vertices, " << report.Triangles << " triangles"
              << (settings.VoxelCoordinates ? " in voxel coordinates\n" : "\n")
              << "Wrote     " << output << " in " << std::fixed << std::setprecision(2)
              << elapsed.count() << " s\n";

    auto const& timing = report.Timing;
    auto const share = [&](double seconds) {
      std::ostringstream out;
      out << std::fixed << std::setprecision(2) << std::setw(8) << seconds << " s "
          << std::setw(5) << std::setprecision(1)
          << (timing.Total > 0.0 ? 100.0 * seconds / timing.Total : 0.0) << " %";
      return out.str();
    };

    std::cout << "\nWhere the time went   (the three stages run concurrently, so these overlap\n"
              << "                       and sum to more than the total; the largest is the cost)\n"
              << "  read          " << share(timing.Read) << "   [reader thread]\n"
              << "  chunking      " << share(timing.Chunking) << "\n"
              << "  materialize   " << share(timing.Materialize) << "\n"
              << "  extract       " << share(timing.Extract) << "   [" << report.BackendName;
    if (report.ExtractionThreads > 1) std::cout << ", " << report.ExtractionThreads << " threads";
    std::cout << "]\n"
              << "  weld          " << share(timing.Weld) << "\n"
              << "  write         " << share(timing.Write) << "   [writer thread"
              << (report.SpilledToDisk ? ", spilled to disk]" : "]") << "\n"
              << "  wall clock    " << share(timing.Total) << "\n";

    if (timing.Extract > 0.0 && report.Chunks > 0) {
      std::cout << "  " << report.Chunks << " chunks extracted at " << std::setprecision(1)
                << (timing.Extract * 1e6 / report.Chunks) << " us each\n";
    }

    return EXIT_SUCCESS;

  } catch (std::exception const& error) {
    std::cerr << "error: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
