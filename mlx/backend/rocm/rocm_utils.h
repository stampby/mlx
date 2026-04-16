// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

namespace mlx::core {

// Throw exception if the hip API does not succeed.
void check_hip_error(const char* name, hipError_t err);
void check_hip_error(const char* name, hipError_t err);

// The macro version that prints the command that failed.
#define CHECK_HIP_ERROR(cmd) check_hip_error(#cmd, (cmd))

// Base class for RAII managed HIP resources.
template <typename Handle, hipError_t (*Destroy)(Handle)>
class CudaHandle {
 public:
  CudaHandle(Handle handle = nullptr) : handle_(handle) {}

  CudaHandle(CudaHandle&& other) : handle_(other.handle_) {
    assert(this != &other);
    other.handle_ = nullptr;
  }

  ~CudaHandle() {
    // Skip if there was an error to avoid throwing in the destructors
    if (hipPeekAtLastError() != hipSuccess) {
      return;
    }
    reset();
  }

  CudaHandle(const CudaHandle&) = delete;
  CudaHandle& operator=(const CudaHandle&) = delete;

  CudaHandle& operator=(CudaHandle&& other) {
    assert(this != &other);
    reset();
    std::swap(handle_, other.handle_);
    return *this;
  }

  void reset() {
    if (handle_ != nullptr) {
      CHECK_HIP_ERROR(Destroy(handle_));
      handle_ = nullptr;
    }
  }

  operator Handle() const {
    return handle_;
  }

 protected:
  Handle handle_;
};

namespace cu {
class Device;
}; // namespace cu

// Wrappers of HIP resources.
class CudaGraph : public CudaHandle<hipGraph_t, hipGraphDestroy> {
 public:
  using CudaHandle::CudaHandle;
  explicit CudaGraph(cu::Device& device);
  void end_capture(hipStream_t stream);
};

class CudaGraphExec : public CudaHandle<hipGraphExec_t, hipGraphExecDestroy> {
 public:
  void instantiate(hipGraph_t graph);
};

class CudaStream : public CudaHandle<hipStream_t, hipStreamDestroy> {
 public:
  using CudaHandle::CudaHandle;
  explicit CudaStream(cu::Device& device);
};

} // namespace mlx::core

// Utility functions (replaces cuda:: namespace from CCCL)
namespace mlx::core::rocm {

template <typename T>
constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

} // namespace mlx::core::rocm
