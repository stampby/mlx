// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"

namespace mlx::core::rocm {

// hipBLASLt GEMM wrapper functions
// hipBLASLt provides optimized GEMM kernels that can outperform rocBLAS
// for half-precision (fp16/bf16) matrix multiplications by using hardware
// matrix cores more efficiently and selecting algorithms via heuristics.

// Returns true if hipBLASLt is available and usable on the current device.
bool is_hipblaslt_available();

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
    Dtype dtype);

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
    Dtype dtype);

} // namespace mlx::core::rocm
