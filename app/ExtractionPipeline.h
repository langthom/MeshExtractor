#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <omp.h>

#include "../MeshExtraction/ChunkMesh.h"
#include "../MeshExtraction/Chunkifier.h"
#include "../MeshExtraction/CpuChunkMeshExtractor.h"
#include "../MeshExtraction/CudaChunkMeshExtractor.h"
#include "../MeshExtraction/MeshAssembler.h"
#include "../MeshExtraction/SlabSchedule.h"
#include "../MeshIO/PLYStreamWriter.h"
#include "../VolumeIO/SliceChunkedMHDIO.h"

namespace parallel_mesh_extractor {

  enum class ExtractionBackend { Gpu, Cpu };

  struct ExtractionSettings {
    float IsoThreshold = 0.0f;

    /// Value sampled outside the volume. It is what closes the surface along the volume wall, so
    /// it belongs on the empty side of the isovalue for the mesh to come out capped there.
    float BackgroundValue = 0.0f;

    /// How many chunk layers to hold at once. One keeps the footprint smallest; more trades memory
    /// for fewer, larger reads.
    std::uint32_t LayersPerSlab = 1;

    /// Write voxel indices rather than world coordinates, i.e. apply neither the spacing nor the
    /// origin from the volume's meta data.
    bool VoxelCoordinates = false;

    PLYFormat Format = PLYFormat::BinaryLittleEndian;

    /// Where the extraction runs. The host is the default because it is simply faster here: a
    /// 62^3 chunk is far too little work to cover the launch and transfer latency of a kernel, so
    /// the device spends most of its time idle, while the same chunks spread across the cores keep
    /// all of them busy. Measured on a 20 thread machine against an RTX 3070, 143.7 us per chunk
    /// against 353.8. Defaulting here also means the tool runs without CUDA.
    ExtractionBackend Backend = ExtractionBackend::Cpu;

    /// Threads the CPU backend uses. Zero means as many as the machine has.
    std::uint32_t ExtractionThreads = 0;

    /// Chunks extracted together. A batch is what gives the CPU backend something to spread across
    /// its threads; the GPU backend walks it one chunk at a time. Larger batches cost a megabyte
    /// of memory each and buy nothing past the thread count.
    std::uint32_t ChunkBatch = 64;

    /// Faces held in memory by the writer before it spills to a temporary file.
    std::size_t FaceBufferBytes = PLYStreamWriter::DefaultFaceBufferBytes;
  };

  struct ExtractionReport {
    std::array<std::uint32_t, 3> Dimensions{};
    std::array<float, 3> Spacing{};
    std::array<float, 3> Origin{};

    std::uint64_t Slabs = 0;
    std::uint64_t Chunks = 0;
    std::uint64_t ChunksExtracted = 0;
    std::uint64_t Vertices = 0;
    std::uint64_t Triangles = 0;

    std::uint32_t SlabSlices = 0;
    std::uint64_t SlabBytes = 0;

    char const* BackendName = "";
    std::uint32_t ExtractionThreads = 1;
    bool SpilledToDisk = false;

    /// Time each stage was busy, in seconds. The reader, the extraction and the writer now run on
    /// separate threads, so these overlap and add up to *more* than Total -- which is the point.
    /// The largest of them is what the run costs.
    struct Timings {
      double Read = 0.0;
      double Chunking = 0.0;
      double Materialize = 0.0;
      double Extract = 0.0;
      double Weld = 0.0;
      double Write = 0.0;
      double Total = 0.0;
    } Timing;
  };

  namespace detail {

    /// A hand-off point between two pipeline stages, bounded so that a fast producer cannot run
    /// ahead of a slow consumer and use all the memory doing it. That bound *is* the backpressure
    /// the pipeline needs: the reader blocks by itself once the extraction is behind.
    template<class T>
    class BoundedQueue {
    public:
      explicit BoundedQueue(std::size_t capacity) : Capacity(capacity) {}

      /// Returns false once the queue has been closed, so a producer can stop on a consumer that
      /// gave up.
      bool Push(T value) {
        std::unique_lock<std::mutex> lock(this->Mutex);
        this->NotFull.wait(lock, [&] { return this->Items.size() < this->Capacity || this->Closed; });
        if (this->Closed) return false;

        this->Items.push_back(std::move(value));
        lock.unlock();
        this->NotEmpty.notify_one();
        return true;
      }

      /// Returns false once the queue is closed *and* drained.
      bool Pop(T& value) {
        std::unique_lock<std::mutex> lock(this->Mutex);
        this->NotEmpty.wait(lock, [&] { return !this->Items.empty() || this->Closed; });
        if (this->Items.empty()) return false;

        value = std::move(this->Items.front());
        this->Items.pop_front();
        lock.unlock();
        this->NotFull.notify_one();
        return true;
      }

      void Close() {
        {
          std::lock_guard<std::mutex> lock(this->Mutex);
          this->Closed = true;
        }
        this->NotEmpty.notify_all();
        this->NotFull.notify_all();
      }

    private:
      std::mutex Mutex;
      std::condition_variable NotEmpty, NotFull;
      std::deque<T> Items;
      std::size_t Capacity;
      bool Closed = false;
    };

    /// How a batch of chunks is turned into meshes. The two backends differ only in where the work
    /// runs, and both produce the same surface.
    class ExtractionBackendBase {
    public:
      virtual ~ExtractionBackendBase() = default;
      virtual void ExtractBatch(std::vector<Chunkifier::DataChunk> const& chunks, std::size_t count,
                                std::vector<ChunkMesh>& out) = 0;
      virtual char const* Name() const = 0;
      virtual std::uint32_t Threads() const { return 1; }
    };

    class GpuBackend : public ExtractionBackendBase {
    public:
      explicit GpuBackend(float isoThreshold) : Extractor(isoThreshold) {}

      void ExtractBatch(std::vector<Chunkifier::DataChunk> const& chunks, std::size_t count,
                        std::vector<ChunkMesh>& out) override {
        // One chunk at a time: the extractor owns a single stream, and a chunk is far too small to
        // fill the device anyway.
        for (std::size_t i = 0; i < count; ++i) this->Extractor.Extract(chunks[i], out[i]);
      }

      char const* Name() const override { return "gpu"; }

    private:
      CudaChunkMeshExtractor Extractor;
    };

    class CpuBackend : public ExtractionBackendBase {
    public:
      CpuBackend(float isoThreshold, std::uint32_t threads) : ThreadCount(threads) {
        this->PerThread.reserve(threads);
        for (std::uint32_t i = 0; i < threads; ++i) {
          this->PerThread.push_back(std::make_unique<CpuChunkMeshExtractor>(isoThreshold));
        }
      }

      void ExtractBatch(std::vector<Chunkifier::DataChunk> const& chunks, std::size_t count,
                        std::vector<ChunkMesh>& out) override {
        // A chunk per thread. Each extractor owns its scratch, so there is nothing shared and
        // nothing to synchronise; the meshes land in their own slots and are welded in order
        // afterwards, which keeps the result independent of how the work was spread.
        auto const n = static_cast<std::int64_t>(count);
        #pragma omp parallel for schedule(dynamic) num_threads(this->ThreadCount)
        for (std::int64_t i = 0; i < n; ++i) {
          int const thread = omp_get_thread_num();
          this->PerThread[static_cast<std::size_t>(thread)]->Extract(chunks[i], out[i]);
        }
      }

      char const* Name() const override { return "cpu"; }
      std::uint32_t Threads() const override { return this->ThreadCount; }

    private:
      std::uint32_t ThreadCount;
      std::vector<std::unique_ptr<CpuChunkMeshExtractor>> PerThread;
    };

    /// What the extraction stage hands the writer: the vertices that were new, and the triangles
    /// of one chunk already renumbered onto the global indexing.
    struct WriteItem {
      std::vector<std::array<float, 3>> Vertices;
      std::vector<std::uint32_t> Indices;
    };

  } // namespace detail

  /// Reads an MHD volume, extracts its isosurface and writes it as a PLY.
  ///
  /// Three stages run concurrently: one thread reads slabs, one extracts and welds them, one
  /// writes. The queues between them are bounded, so the reader stalls of its own accord whenever
  /// the rest is behind and the volume is never ahead of the mesh by more than a slab or two.
  ///
  /// There is exactly one writer, and deliberately so. Writing a PLY is one sequential stream, and
  /// on a rotating disk splitting it across threads makes it slower rather than faster -- measured
  /// on a 7200 rpm drive, eight writers reach 103 MB/s against 190 MB/s for one.
  ///
  /// The welding stays on a single thread as well, because it is what assigns the global vertex
  /// indices and those have to come out in a defined order for the file to be reproducible.
  inline ExtractionReport ExtractVolumeToPLY(std::filesystem::path const& input,
                                             std::filesystem::path const& output,
                                             ExtractionSettings const& settings,
                                             std::ostream* progress = nullptr) {
    using Clock = std::chrono::steady_clock;
    auto const secondsSince = [](Clock::time_point start) {
      return std::chrono::duration<double>(Clock::now() - start).count();
    };

    SliceChunkedMHDIO volumeIO;
    volumeIO.SetFilePath(input);
    auto const metaData = volumeIO.ReadMetaData();

    ExtractionReport report;
    report.Dimensions = metaData.dim;
    report.Spacing    = metaData.spacing;
    report.Origin     = metaData.origin;

    auto const plan = PlanSlabs(metaData.dim, settings.LayersPerSlab);
    report.Slabs = plan.size();

    auto const sliceSize = static_cast<std::size_t>(metaData.dim[0]) * metaData.dim[1];
    report.SlabSlices = MaxSlabSliceCount(metaData.dim, settings.LayersPerSlab);
    report.SlabBytes  = static_cast<std::uint64_t>(sliceSize) * report.SlabSlices * sizeof(float);

    std::uint32_t threads = settings.ExtractionThreads;
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());

    std::unique_ptr<detail::ExtractionBackendBase> backend;
    if (settings.Backend == ExtractionBackend::Cpu) {
      backend = std::make_unique<detail::CpuBackend>(settings.IsoThreshold, threads);
    } else {
      backend = std::make_unique<detail::GpuBackend>(settings.IsoThreshold);
    }
    report.BackendName = backend->Name();
    report.ExtractionThreads = backend->Threads();

    auto const runStarted = Clock::now();

    // ------------------------------------------------------------------------------ slab buffers
    // Two of them, so the reader can be filling one while the extraction works through the other.
    constexpr std::size_t slabBuffers = 2;
    std::vector<std::vector<float>> slabs(slabBuffers);
    for (auto& buffer : slabs) buffer.resize(sliceSize * report.SlabSlices);

    struct FilledSlab { std::size_t Buffer; SlabWindow Window; };

    detail::BoundedQueue<std::size_t> freeSlabs(slabBuffers);
    detail::BoundedQueue<FilledSlab> filledSlabs(slabBuffers);
    for (std::size_t i = 0; i < slabBuffers; ++i) freeSlabs.Push(i);

    detail::BoundedQueue<detail::WriteItem> writeQueue(64);

    std::exception_ptr readerError, writerError;
    std::atomic<double> readSeconds{0.0}, writeSeconds{0.0};

    // ----------------------------------------------------------------------------- reader thread
    std::thread reader([&] {
      try {
        for (auto const& window : plan) {
          std::size_t buffer = 0;
          if (!freeSlabs.Pop(buffer)) return;      // the rest of the pipeline gave up

          auto const started = Clock::now();
          // Straight into the slab buffer: the conversion from the volume's native voxel type to
          // float happens element by element on the way, with nothing allocated in between.
          if (!volumeIO.ReadSlicesInto(static_cast<unsigned int>(window.ZBegin), window.ZCount,
                                       slabs[buffer].data())) {
            throw std::runtime_error("reading slices from the volume failed");
          }
          readSeconds.store(readSeconds.load() + secondsSince(started));

          if (!filledSlabs.Push(FilledSlab{buffer, window})) return;
        }
      } catch (...) {
        readerError = std::current_exception();
      }
      filledSlabs.Close();
    });

    // ----------------------------------------------------------------------------- writer thread
    PLYStreamWriter writer(output, settings.Format, settings.FaceBufferBytes);

    std::thread writerThread([&] {
      try {
        detail::WriteItem item;
        while (writeQueue.Pop(item)) {
          auto const started = Clock::now();
          writer.WriteVertices(item.Vertices);
          writer.WriteFaces(item.Indices);
          writeSeconds.store(writeSeconds.load() + secondsSince(started));
        }
      } catch (...) {
        writerError = std::current_exception();
        writeQueue.Close();   // stop the extraction from filling a queue nobody drains
      }
    });

    // ------------------------------------------------------ extraction and welding, on this one
    MeshAssembler assembler;
    std::vector<Chunkifier::DataChunk> batch(settings.ChunkBatch);
    std::vector<ChunkMesh> batchMeshes(settings.ChunkBatch);

    try {
      FilledSlab filled{};
      while (filledSlabs.Pop(filled)) {
        auto const& window = filled.Window;

        auto stage = Clock::now();
        Chunkifier chunkifier(slabs[filled.Buffer].data(), metaData.dim, window,
                              settings.IsoThreshold, settings.BackgroundValue);
        chunkifier.ComputeChunking(metaData.dim);
        report.Timing.Chunking += secondsSince(stage);

        auto iterator = chunkifier.begin();
        auto const end = chunkifier.end();

        while (iterator != end) {
          // Collect a batch. Dereferencing is what copies a chunk out of the slab.
          stage = Clock::now();
          std::size_t count = 0;
          while (count < batch.size() && iterator != end) {
            batch[count] = *iterator;
            ++iterator;
            ++count;
          }
          report.Timing.Materialize += secondsSince(stage);
          if (count == 0) break;

          stage = Clock::now();
          backend->ExtractBatch(batch, count, batchMeshes);
          report.Timing.Extract += secondsSince(stage);

          // Welded strictly in batch order, so that the vertex numbering -- and therefore the
          // file -- does not depend on how the extraction was scheduled.
          stage = Clock::now();
          for (std::size_t i = 0; i < count; ++i) {
            ++report.Chunks;
            if (batchMeshes[i].IsEmpty()) continue;
            ++report.ChunksExtracted;

            detail::WriteItem item;
            assembler.Add(batchMeshes[i], item.Vertices, item.Indices);

            if (!settings.VoxelCoordinates) {
              for (auto& vertex : item.Vertices) {
                for (int axis = 0; axis < 3; ++axis) {
                  vertex[axis] = metaData.origin[axis] + vertex[axis] * metaData.spacing[axis];
                }
              }
            }

            report.Timing.Weld += secondsSince(stage);
            if (!writeQueue.Push(std::move(item))) {
              throw std::runtime_error("the writer stopped before the mesh was complete");
            }
            stage = Clock::now();
          }
          report.Timing.Weld += secondsSince(stage);
        }

        assembler.PruneBelowZ(window.PruneBelowZ);
        freeSlabs.Push(filled.Buffer);

        if (progress) {
          *progress << "  slab layers [" << window.TileZBegin << ", "
                    << window.TileZBegin + window.TileZCount << "), "
                    << assembler.VertexCount() << " vertices so far\n";
        }
      }
    } catch (...) {
      // Unblock both neighbours before unwinding, or joining below would deadlock.
      freeSlabs.Close();
      filledSlabs.Close();
      writeQueue.Close();
      if (reader.joinable()) reader.join();
      if (writerThread.joinable()) writerThread.join();
      throw;
    }

    writeQueue.Close();
    freeSlabs.Close();
    reader.join();
    writerThread.join();

    if (readerError) std::rethrow_exception(readerError);
    if (writerError) std::rethrow_exception(writerError);

    auto const finishStarted = Clock::now();
    writer.Finish();
    report.Timing.Write = writeSeconds.load() + secondsSince(finishStarted);
    report.Timing.Read  = readSeconds.load();
    report.Timing.Total = secondsSince(runStarted);

    report.Vertices  = writer.VertexCount();
    report.Triangles = writer.TriangleCount();
    report.SpilledToDisk = writer.SpilledToDisk();
    return report;
  }

} // namespace parallel_mesh_extractor
