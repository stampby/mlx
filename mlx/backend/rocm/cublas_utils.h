// Copyright © 2025 Apple Inc.
#pragma once

#include "mlx/array.h"
#include "mlx/backend/hip/device.h"
#include "mlx/dtype_utils.h"

#include <hipblaslt.h>

namespace mlx::core {
namespace cublas_utils {

hipblasLtMatrixLayout_t create_matrix_layout(
    hipDataType type,
    uint64_t rows,
    uint64_t cols,
    bool transposed,
    int64_t ld,
    int32_t batch_count,
    int64_t batch_stride);

inline hipDataType dtype_to_cublas_type(Dtype dtype, std::string_view tag) {
  switch (dtype) {
    case float16:
      return HIP_R_16F;
    case bfloat16:
      return HIP_R_16BF;
    case float32:
      return HIP_R_32F;
    case float64:
      return HIP_R_64F;
    case complex64:
      return HIP_C_32F;
    default:
      throw std::runtime_error(
          fmt::format(
              "Unsupported dtype in {}: {}.", tag, dtype_to_string(dtype)));
  }
}

} // namespace cublas_utils

void check_cublas_error(const char* name, hipblasStatus_t err);

#define CHECK_HIPBLAS_ERROR(cmd) check_cublas_error(#cmd, (cmd))

void init_cublas_handles_cache();

class CublasMatmulBase {
 public:
  virtual ~CublasMatmulBase();

  void set_bias(cu::CommandEncoder& encoder, const array& bias);

 protected:
  CublasMatmulBase() = default;

  // Common member variables shared by all matmul types
  uint64_t M_;
  uint64_t N_;
  hipDataType scale_type_;
  hipblasLtMatmulPreference_t pref_{nullptr};
  hipblasLtHandle_t handle_{nullptr};
  hipblasLtMatmulDesc_t matmul_desc_{nullptr};
  hipblasLtMatrixLayout_t a_desc_{nullptr};
  hipblasLtMatrixLayout_t b_desc_{nullptr};
  hipblasLtMatrixLayout_t c_desc_{nullptr};
  hipblasLtMatrixLayout_t out_desc_{nullptr};
  hipblasLtMatmulHeuristicResult_t heuristic_;

  void init_base(
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
      int64_t b_batch_stride);

  void execute_matmul(
      cu::CommandEncoder& encoder,
      void* out,
      const void* a,
      const void* b,
      const void* c,
      const void* alpha_ptr,
      const void* beta_ptr);
};

} // namespace mlx::core
