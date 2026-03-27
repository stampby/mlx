// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/gemms/hipblaslt_gemm.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdlib>
#include <iostream>
#include <mutex>

namespace mlx::core::rocm {

namespace {

// Maximum workspace size for hipBLASLt algorithms (32 MB).
// hipBLASLt may request scratch memory for certain algorithm choices.
constexpr size_t kMaxWorkspaceBytes = 32u * 1024u * 1024u;

// Per-device hipBLASLt handle cache. Lazily initialised, thread-safe.
struct HipblasltState {
  hipblasLtHandle_t handle{nullptr};
  bool initialized{false};
  bool available{false};
  std::mutex mutex;

  // Persistent workspace allocation (grown as needed, never shrunk).
  void* workspace{nullptr};
  size_t workspace_size{0};
};

// One state per device (indexed by HIP device ordinal).
// 16 devices should be more than enough for any system.
static constexpr int kMaxDevices = 16;
static HipblasltState g_state[kMaxDevices];

HipblasltState& get_state(int device_id) {
  if (device_id < 0 || device_id >= kMaxDevices) {
    throw std::runtime_error(
        "hipBLASLt: device id out of range: " + std::to_string(device_id));
  }
  return g_state[device_id];
}

// Initialise the hipBLASLt handle for the given device.
// Must be called with state.mutex held.
void init_handle(HipblasltState& state, int device_id) {
  if (state.initialized) {
    return;
  }
  state.initialized = true;

  hipblasStatus_t status = hipblasLtCreate(&state.handle);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    state.available = false;
    state.handle = nullptr;
    std::cerr << "Warning: hipBLASLt initialization failed (status "
              << static_cast<int>(status) << ")." << std::endl;
    return;
  }
  state.available = true;
}

hipblasLtHandle_t get_handle(int device_id) {
  auto& state = get_state(device_id);
  if (!state.initialized) {
    std::lock_guard<std::mutex> lock(state.mutex);
    init_handle(state, device_id);
  }
  if (!state.available) {
    throw std::runtime_error("hipBLASLt is not available on this device.");
  }
  return state.handle;
}

// Ensure the per-device workspace is at least `required` bytes.
// Returns the workspace pointer and the actual allocated size.
// Must be called from within a launch_kernel callback (i.e., on the
// stream-submission thread for this device), so no extra locking is needed
// beyond the device serialisation that CommandEncoder already provides.
std::pair<void*, size_t> ensure_workspace(int device_id, size_t required) {
  auto& state = get_state(device_id);
  if (required <= state.workspace_size && state.workspace != nullptr) {
    return {state.workspace, state.workspace_size};
  }
  // Free old allocation (hipFree is a no-op on nullptr).
  if (state.workspace) {
    (void)hipFree(state.workspace);
    state.workspace = nullptr;
    state.workspace_size = 0;
  }
  if (required == 0) {
    return {nullptr, 0};
  }
  hipError_t err = hipMalloc(&state.workspace, required);
  if (err != hipSuccess) {
    state.workspace = nullptr;
    state.workspace_size = 0;
    return {nullptr, 0};
  }
  state.workspace_size = required;
  return {state.workspace, state.workspace_size};
}

hipDataType to_hipblaslt_dtype(Dtype dtype) {
  switch (dtype) {
    case float32:
      return HIP_R_32F;
    case float16:
      return HIP_R_16F;
    case bfloat16:
      return HIP_R_16BF;
    default:
      throw std::runtime_error("Unsupported dtype for hipBLASLt GEMM");
  }
}

hipblasOperation_t to_hipblas_op(bool transpose) {
  return transpose ? HIPBLAS_OP_T : HIPBLAS_OP_N;
}

// RAII wrappers for hipBLASLt descriptors to avoid leaks on error paths.
struct MatmulDescGuard {
  hipblasLtMatmulDesc_t desc{nullptr};
  ~MatmulDescGuard() {
    if (desc)
      hipblasLtMatmulDescDestroy(desc);
  }
};
struct MatrixLayoutGuard {
  hipblasLtMatrixLayout_t layout{nullptr};
  ~MatrixLayoutGuard() {
    if (layout)
      hipblasLtMatrixLayoutDestroy(layout);
  }
};
struct PreferenceGuard {
  hipblasLtMatmulPreference_t pref{nullptr};
  ~PreferenceGuard() {
    if (pref)
      hipblasLtMatmulPreferenceDestroy(pref);
  }
};

// Core implementation: set up descriptors, find the best algorithm, and
// execute the matmul on the given stream.
void hipblaslt_gemm_impl(
    hipblasLtHandle_t handle,
    int device_id,
    hipblasOperation_t op_a,
    hipblasOperation_t op_b,
    int M,
    int N,
    int K,
    const float* alpha,
    const void* a_ptr,
    int lda,
    int64_t stride_a,
    const void* b_ptr,
    int ldb,
    int64_t stride_b,
    const float* beta,
    void* c_ptr,
    int ldc,
    int64_t stride_c,
    int batch_count,
    hipDataType data_type,
    hipStream_t stream) {
  hipblasStatus_t status;

  // Compute type: always fp32 accumulation for half-precision inputs.
  hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
  hipDataType scale_type = HIP_R_32F;

  // --- Matmul descriptor ---
  MatmulDescGuard matmul_guard;
  status =
      hipblasLtMatmulDescCreate(&matmul_guard.desc, compute_type, scale_type);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatmulDescCreate failed: " +
        std::to_string(static_cast<int>(status)));
  }

  // Set transpose attributes.
  int32_t trans_a_val = static_cast<int32_t>(op_a);
  int32_t trans_b_val = static_cast<int32_t>(op_b);
  hipblasLtMatmulDescSetAttribute(
      matmul_guard.desc,
      HIPBLASLT_MATMUL_DESC_TRANSA,
      &trans_a_val,
      sizeof(trans_a_val));
  hipblasLtMatmulDescSetAttribute(
      matmul_guard.desc,
      HIPBLASLT_MATMUL_DESC_TRANSB,
      &trans_b_val,
      sizeof(trans_b_val));

  // --- Matrix layouts (column-major, as expected by BLAS) ---
  // A is (op_a == N) ? M x K : K x M  in column-major
  // B is (op_b == N) ? K x N : N x K  in column-major
  // C is M x N in column-major
  uint64_t a_rows = (op_a == HIPBLAS_OP_N) ? M : K;
  uint64_t a_cols = (op_a == HIPBLAS_OP_N) ? K : M;
  uint64_t b_rows = (op_b == HIPBLAS_OP_N) ? K : N;
  uint64_t b_cols = (op_b == HIPBLAS_OP_N) ? N : K;

  MatrixLayoutGuard layout_a, layout_b, layout_c, layout_d;

  status = hipblasLtMatrixLayoutCreate(
      &layout_a.layout, data_type, a_rows, a_cols, lda);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatrixLayoutCreate(A) failed: " +
        std::to_string(static_cast<int>(status)));
  }

  status = hipblasLtMatrixLayoutCreate(
      &layout_b.layout, data_type, b_rows, b_cols, ldb);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatrixLayoutCreate(B) failed: " +
        std::to_string(static_cast<int>(status)));
  }

  status = hipblasLtMatrixLayoutCreate(
      &layout_c.layout, data_type, M, N, ldc);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatrixLayoutCreate(C) failed: " +
        std::to_string(static_cast<int>(status)));
  }

  // D has the same layout as C (in-place: D == C).
  status = hipblasLtMatrixLayoutCreate(
      &layout_d.layout, data_type, M, N, ldc);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatrixLayoutCreate(D) failed: " +
        std::to_string(static_cast<int>(status)));
  }

  // Set batch attributes when doing strided batched GEMM.
  if (batch_count > 1) {
    int32_t bc = batch_count;
    hipblasLtMatrixLayoutSetAttribute(
        layout_a.layout,
        HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &bc,
        sizeof(bc));
    hipblasLtMatrixLayoutSetAttribute(
        layout_a.layout,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_a,
        sizeof(stride_a));

    hipblasLtMatrixLayoutSetAttribute(
        layout_b.layout,
        HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &bc,
        sizeof(bc));
    hipblasLtMatrixLayoutSetAttribute(
        layout_b.layout,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_b,
        sizeof(stride_b));

    hipblasLtMatrixLayoutSetAttribute(
        layout_c.layout,
        HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &bc,
        sizeof(bc));
    hipblasLtMatrixLayoutSetAttribute(
        layout_c.layout,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_c,
        sizeof(stride_c));

    hipblasLtMatrixLayoutSetAttribute(
        layout_d.layout,
        HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &bc,
        sizeof(bc));
    hipblasLtMatrixLayoutSetAttribute(
        layout_d.layout,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride_c,
        sizeof(stride_c));
  }

  // --- Algorithm selection via heuristic ---
  PreferenceGuard pref_guard;
  status = hipblasLtMatmulPreferenceCreate(&pref_guard.pref);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatmulPreferenceCreate failed: " +
        std::to_string(static_cast<int>(status)));
  }

  uint64_t max_ws = kMaxWorkspaceBytes;
  hipblasLtMatmulPreferenceSetAttribute(
      pref_guard.pref,
      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_ws,
      sizeof(max_ws));

  hipblasLtMatmulHeuristicResult_t heuristic;
  int returned_algo_count = 0;

  status = hipblasLtMatmulAlgoGetHeuristic(
      handle,
      matmul_guard.desc,
      layout_a.layout,
      layout_b.layout,
      layout_c.layout,
      layout_d.layout,
      pref_guard.pref,
      1, // requestedAlgoCount
      &heuristic,
      &returned_algo_count);

  if (status != HIPBLAS_STATUS_SUCCESS || returned_algo_count == 0) {
    throw std::runtime_error(
        "hipblasLtMatmulAlgoGetHeuristic failed (status=" +
        std::to_string(static_cast<int>(status)) +
        ", returned=" + std::to_string(returned_algo_count) + ")");
  }

  // --- Workspace allocation ---
  size_t ws_needed = heuristic.workspaceSize;
  void* ws_ptr = nullptr;
  size_t ws_actual = 0;
  if (ws_needed > 0) {
    auto [p, s] = ensure_workspace(device_id, ws_needed);
    ws_ptr = p;
    ws_actual = s;
    if (ws_ptr == nullptr && ws_needed > 0) {
      throw std::runtime_error(
          "hipBLASLt: failed to allocate workspace of " +
          std::to_string(ws_needed) + " bytes");
    }
  }

  // --- Execute the matmul ---
  status = hipblasLtMatmul(
      handle,
      matmul_guard.desc,
      alpha,
      a_ptr,
      layout_a.layout,
      b_ptr,
      layout_b.layout,
      beta,
      c_ptr,
      layout_c.layout,
      c_ptr, // D == C (in-place)
      layout_d.layout,
      &heuristic.algo,
      ws_ptr,
      ws_actual,
      stream);

  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "hipblasLtMatmul failed: " +
        std::to_string(static_cast<int>(status)));
  }
}

} // namespace

bool is_hipblaslt_available() {
  int device_id = 0;
  (void)hipGetDevice(&device_id);
  auto& state = get_state(device_id);
  if (!state.initialized) {
    std::lock_guard<std::mutex> lock(state.mutex);
    init_handle(state, device_id);
  }
  return state.available;
}

void hipblaslt_gemm(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    const array& b,
    int ldb,
    float beta,
    array& c,
    int ldc,
    Dtype dtype) {
  int device_id = encoder.device().hip_device();
  hipblasLtHandle_t handle = get_handle(device_id);
  hipDataType hip_dtype = to_hipblaslt_dtype(dtype);

  // hipBLASLt uses column-major layout. MLX stores row-major, so we swap A
  // and B and compute C^T = B^T * A^T, just like the rocBLAS path.
  hipblasOperation_t op_a = to_hipblas_op(transpose_b);
  hipblasOperation_t op_b = to_hipblas_op(transpose_a);

  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* c_ptr = gpu_ptr<void>(c);

  encoder.launch_kernel(
      [=, &encoder](hipStream_t stream) {
        hipblaslt_gemm_impl(
            handle,
            device_id,
            op_a,
            op_b,
            N, // swap M/N for col-major trick
            M,
            K,
            &alpha,
            b_ptr, // swap A/B
            ldb,
            0, // stride_a (unused for non-batched)
            a_ptr,
            lda,
            0, // stride_b (unused for non-batched)
            &beta,
            c_ptr,
            ldc,
            0, // stride_c (unused for non-batched)
            1, // batch_count
            hip_dtype,
            stream);
      });
}

void hipblaslt_gemm_batched(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    int64_t stride_a,
    const array& b,
    int ldb,
    int64_t stride_b,
    float beta,
    array& c,
    int ldc,
    int64_t stride_c,
    int batch_count,
    Dtype dtype) {
  int device_id = encoder.device().hip_device();
  hipblasLtHandle_t handle = get_handle(device_id);
  hipDataType hip_dtype = to_hipblaslt_dtype(dtype);

  // Same column-major swap as above.
  hipblasOperation_t op_a = to_hipblas_op(transpose_b);
  hipblasOperation_t op_b = to_hipblas_op(transpose_a);

  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* c_ptr = gpu_ptr<void>(c);

  encoder.launch_kernel(
      [=, &encoder](hipStream_t stream) {
        hipblaslt_gemm_impl(
            handle,
            device_id,
            op_a,
            op_b,
            N,
            M,
            K,
            &alpha,
            b_ptr,
            ldb,
            stride_b, // swapped: was b, now is "A" in col-major
            a_ptr,
            lda,
            stride_a, // swapped: was a, now is "B" in col-major
            &beta,
            c_ptr,
            ldc,
            stride_c,
            batch_count,
            hip_dtype,
            stream);
      });
}

} // namespace mlx::core::rocm
