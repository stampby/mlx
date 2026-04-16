#include "hip/hip_runtime.h"
// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/dtype_utils.h"

#include <fmt/format.h>
#include <hip/cmath>
#include <vector>

namespace mlx::core {

void check_hip_error(const char* name, hipError_t err) {
  if (err != hipSuccess) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, hipGetErrorString(err)));
  }
}

void check_hip_error(const char* name, hipError_t err) {
  if (err != hipSuccess) {
    const char* err_str = "Unknown error";
    hipDrvGetErrorString(err, &err_str);
    throw std::runtime_error(fmt::format("{} failed: {}", name, err_str));
  }
}

const char* dtype_to_hip_type(const Dtype& dtype) {
  switch (dtype) {
    case bool_:
      return "bool";
    case int8:
      return "int8_t";
    case int16:
      return "int16_t";
    case int32:
      return "int32_t";
    case int64:
      return "int64_t";
    case uint8:
      return "uint8_t";
    case uint16:
      return "uint16_t";
    case uint32:
      return "uint32_t";
    case uint64:
      return "uint64_t";
    case float16:
      return "__half";
    case bfloat16:
      return "__hip_bfloat16";
    case float32:
      return "float";
    case float64:
      return "double";
    case complex64:
      return "mlx::core::cu::complex64_t";
    default:
      return "unknown";
  }
}

CudaGraph::CudaGraph(cu::Device& device) {
  device.make_current();
  CHECK_HIP_ERROR(hipGraphCreate(&handle_, 0));
}

void CudaGraph::end_capture(hipStream_t stream) {
  CHECK_HIP_ERROR(hipStreamEndCapture(stream, &handle_));
}

void CudaGraphExec::instantiate(hipGraph_t graph) {
  assert(handle_ == nullptr);
  CHECK_HIP_ERROR(hipGraphInstantiate(&handle_, graph, nullptr, nullptr, 0));
}

CudaStream::CudaStream(cu::Device& device) {
  device.make_current();
  CHECK_HIP_ERROR(hipStreamCreateWithFlags(&handle_, hipStreamNonBlocking));
}

void* allocate_workspace(cu::CommandEncoder& encoder, size_t workspace_size) {
  if (workspace_size == 0) {
    return nullptr;
  }

  // Workspace allocation should not be captured.
#ifndef NDEBUG
  hipStreamCaptureStatus status;
  CHECK_HIP_ERROR(hipStreamIsCapturing(encoder.stream(), &status));
  assert(status == hipStreamCaptureStatusNone);
#endif

  // Ensure workspace is 256-byte aligned.
  int nbytes = mlx::core::rocm::ceil_div(workspace_size, 256) * 256;
  array workspace(cu::malloc_async(nbytes, encoder), {nbytes}, int8);
  encoder.add_temporary(workspace);
  return gpu_ptr<void>(workspace);
}

} // namespace mlx::core
