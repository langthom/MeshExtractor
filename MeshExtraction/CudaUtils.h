#pragma once

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace parallel_mesh_extractor {

  /// Thrown when a CUDA runtime call fails. Reporting by exception rather than by aborting keeps
  /// the failure recoverable: a unit test can provoke one and carry on, and the extractor stays
  /// destructible afterwards because every device allocation is owned by a DeviceBuffer.
  class CudaError : public std::runtime_error {
  public:
    explicit CudaError(std::string const& what) : std::runtime_error(what) {}
  };

  namespace detail {
    inline void throwOnCudaError(cudaError_t status, char const* expression,
                                 char const* file, int line) {
      if (status == cudaSuccess) return;

      std::ostringstream message;
      message << "CUDA call failed: " << expression << "\n  at " << file << ":" << line
              << "\n  " << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
      throw CudaError(message.str());
    }
  } // namespace detail

/// Wraps a CUDA runtime call and turns a non-successful status into a CudaError naming the call.
#define CUDA_CHECK(expression)                                                                 \
  ::parallel_mesh_extractor::detail::throwOnCudaError((expression), #expression, __FILE__, __LINE__)

/// Checks whether the most recent kernel launch was accepted. Launch failures are reported
/// asynchronously, so this has to be called explicitly after a launch rather than wrapped around
/// it.
#define CUDA_CHECK_LAST_LAUNCH()                                                               \
  ::parallel_mesh_extractor::detail::throwOnCudaError(cudaGetLastError(), "kernel launch",     \
                                                      __FILE__, __LINE__)

  /// Owning handle for a device allocation. The extraction allocates every one of its buffers once
  /// up front and reuses them for every chunk, so this deliberately offers no resizing: a buffer
  /// that could grow would reintroduce the per chunk allocation this design exists to avoid.
  template<class T>
  class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) : Count(count) {
      if (count != 0) CUDA_CHECK(cudaMalloc(&this->Pointer, count * sizeof(T)));
    }

    ~DeviceBuffer() {
      // Nothing useful can be done about a failure while unwinding, and cudaFree on a null pointer
      // is well defined, so the status is deliberately ignored here.
      if (this->Pointer) cudaFree(this->Pointer);
    }

    DeviceBuffer(DeviceBuffer&& other) noexcept
      : Pointer(std::exchange(other.Pointer, nullptr)), Count(std::exchange(other.Count, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
      if (this != &other) {
        if (this->Pointer) cudaFree(this->Pointer);
        this->Pointer = std::exchange(other.Pointer, nullptr);
        this->Count   = std::exchange(other.Count, 0);
      }
      return *this;
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    T* Get() const { return this->Pointer; }
    std::size_t Size() const { return this->Count; }
    std::size_t SizeInBytes() const { return this->Count * sizeof(T); }

  private:
    T* Pointer = nullptr;
    std::size_t Count = 0;
  };

  /// Owning handle for a page locked host allocation. Pinned staging memory is what lets the host
  /// to device copy of a chunk overlap with the kernels of the previous one.
  template<class T>
  class PinnedBuffer {
  public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(std::size_t count) : Count(count) {
      if (count != 0) CUDA_CHECK(cudaMallocHost(&this->Pointer, count * sizeof(T)));
    }

    ~PinnedBuffer() {
      if (this->Pointer) cudaFreeHost(this->Pointer);
    }

    PinnedBuffer(PinnedBuffer&& other) noexcept
      : Pointer(std::exchange(other.Pointer, nullptr)), Count(std::exchange(other.Count, 0)) {}

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
      if (this != &other) {
        if (this->Pointer) cudaFreeHost(this->Pointer);
        this->Pointer = std::exchange(other.Pointer, nullptr);
        this->Count   = std::exchange(other.Count, 0);
      }
      return *this;
    }

    PinnedBuffer(PinnedBuffer const&) = delete;
    PinnedBuffer& operator=(PinnedBuffer const&) = delete;

    T* Get() const { return this->Pointer; }
    std::size_t Size() const { return this->Count; }

  private:
    T* Pointer = nullptr;
    std::size_t Count = 0;
  };

} // namespace parallel_mesh_extractor
