// Copyright © 2025 Apple Inc.

#include "mlx/backend/hip/cublas_utils.h"
#include "mlx/backend/hip/hip/hip_runtime.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace cublas_utils {

hipblasLtMatrixLayout_t create_matrix_layout(
    hipDataType type,
    uint64_t rows,
    uint64_t cols,
    bool transposed,
    int64_t ld,
    int32_t batch_count,
    int64_t batch_stride) {
  hipblasLtMatrixLayout_t desc;
  if (transposed) {
    std::swap(rows, cols);
  }
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&desc, type, rows, cols, ld));
  if (batch_count > 1) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
        desc,
        HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(int32_t)));
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
        desc,
        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batch_stride,
        sizeof(int64_t)));
  }
  return desc;
}

} // namespace cublas_utils

namespace {

auto& cublas_handles_cache() {
  struct CublasHandles {
    ~CublasHandles() {
      if (handle) {
        CHECK_HIPBLAS_ERROR(hipblasLtDestroy(handle));
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
      }
    }
    hipblasLtHandle_t handle{nullptr};
    hipblasLtMatmulPreference_t pref{nullptr};
  };
  static thread_local std::vector<CublasHandles> cache(gpu::device_count());
  return cache;
}

auto get_cublas_handles(cu::Device& device) {
  auto& storage = cublas_handles_cache().at(device.hip_device());
  if (!storage.handle) {
    // Create cublasLt handle.
    device.make_current();
    CHECK_HIPBLAS_ERROR(hipblasLtCreate(&storage.handle));
    // The recommended cublas workspace size is 4 MiB for pre-Hopper and 32
    // MiB for Hopper+:
    // https://docs.nvidia.com/hip/cublas/#cublassetworkspace
    uint64_t MiB = 1024 * 1024;
    uint64_t workspace_size =
        device.compute_capability_major() >= 9 ? 32 * MiB : 4 * MiB;
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceCreate(&storage.pref));
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceSetAttribute(
        storage.pref,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(uint64_t)));
  }
  return std::make_tuple(storage.handle, storage.pref);
}

} // namespace

void check_cublas_error(const char* name, hipblasStatus_t err) {
  if (err != HIPBLAS_STATUS_SUCCESS) {
    // TODO: Use cublasGetStatusString when it is widely available.
    throw std::runtime_error(
        fmt::format("{} failed with code: {}.", name, static_cast<int>(err)));
  }
}

void init_cublas_handles_cache() {
  cublas_handles_cache();
}

CublasMatmulBase::~CublasMatmulBase() {
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(a_desc_));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(b_desc_));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(c_desc_));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(out_desc_));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescDestroy(matmul_desc_));
}

void CublasMatmulBase::init_base(
    cu::Device& device,
    hipDataType scale_type,
    hipblasComputeType_t compute_type,
    hipDataType data_type,
    hipDataType output_type,
    bool a_transposed,
    uint64_t a_rows,
    uint64_t a_cols,
    int64_t lda,
    bool b_transposed,
    uint64_t b_rows,
    uint64_t b_cols,
    int64_t ldb,
    int32_t batch_count,
    int64_t a_batch_stride,
    int64_t b_batch_stride) {
  M_ = a_rows;
  N_ = b_cols;
  scale_type_ = scale_type;
  std::tie(handle_, pref_) = get_cublas_handles(device);
  heuristic_.state = HIPBLAS_STATUS_NOT_INITIALIZED;

  CHECK_HIPBLAS_ERROR(
      hipblasLtMatmulDescCreate(&matmul_desc_, compute_type, scale_type));

  // In cublasLt matrices use column-major layout, while it is possible to use
  // the CUBLASLT_ORDER_ROW option to switch to row-major layout, the bias
  // epilogue does not work with the option. So instead we swap A and B to make
  // cublasLt return the row-major result, which works because:
  // - the data of a matrix in row-major layout is identical to its transpose in
  //   column-major layout
  // - C^T = (A @ B)^T = B^T @ A^T
  hipblasOperation_t a_op = b_transposed ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul_desc_,
      HIPBLASLT_MATMUL_DESC_TRANSA,
      &a_op,
      sizeof(hipblasOperation_t)));
  hipblasOperation_t b_op = a_transposed ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul_desc_,
      HIPBLASLT_MATMUL_DESC_TRANSB,
      &b_op,
      sizeof(hipblasOperation_t)));

  a_desc_ = cublas_utils::create_matrix_layout(
      data_type,
      b_cols,
      b_rows,
      b_transposed,
      ldb,
      batch_count,
      b_batch_stride);
  b_desc_ = cublas_utils::create_matrix_layout(
      data_type,
      a_cols,
      a_rows,
      a_transposed,
      lda,
      batch_count,
      a_batch_stride);
  out_desc_ = cublas_utils::create_matrix_layout(
      output_type, b_cols, a_rows, false, b_cols, batch_count, b_cols * a_rows);
}

void CublasMatmulBase::execute_matmul(
    cu::CommandEncoder& encoder,
    void* out,
    const void* a,
    const void* b,
    const void* c,
    const void* alpha_ptr,
    const void* beta_ptr) {
  if (heuristic_.state != HIPBLAS_STATUS_SUCCESS) {
    int ret = 0;
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulAlgoGetHeuristic(
        handle_,
        matmul_desc_,
        a_desc_,
        b_desc_,
        c ? c_desc_ : out_desc_,
        out_desc_,
        pref_,
        1,
        &heuristic_,
        &ret));
    if (ret == 0) {
      throw std::runtime_error("Can not find algorithm for matmul.");
    }
  }

  void* workspace_ptr = allocate_workspace(encoder, heuristic_.workspaceSize);

  // Execute matmul
  auto capture = encoder.capture_context();
  CHECK_HIPBLAS_ERROR(hipblasLtMatmul(
      handle_,
      matmul_desc_,
      alpha_ptr,
      b, // a and b are swapped for row-major layout
      a_desc_,
      a,
      b_desc_,
      beta_ptr,
      c ? c : out,
      c ? c_desc_ : out_desc_,
      out,
      out_desc_,
      &heuristic_.algo,
      workspace_ptr,
      heuristic_.workspaceSize,
      encoder.stream()));
}

void CublasMatmulBase::set_bias(
    cu::CommandEncoder& encoder,
    const array& bias) {
  encoder.set_input_array(bias);
  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul_desc_,
      HIPBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue,
      sizeof(epilogue)));
  auto* bias_ptr = gpu_ptr<void>(bias);
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul_desc_,
      HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
      &bias_ptr,
      sizeof(bias_ptr)));
}

} // namespace mlx::core
