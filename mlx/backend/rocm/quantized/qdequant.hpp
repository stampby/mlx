// Shared dequantization utilities for optimized QMM kernels.
// Used by qmv_kernel.hip (GEMV) and qmm_kernel.hip (GEMM).

#pragma once

#include "mlx/backend/rocm/device/config.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core::rocm {

// --- Compile-time constants ---

// Number of quantized values packed per uint32 word.
// 4-bit: 8 values, 2-bit: 16 values, 8-bit: 4 values.
template <int BITS>
inline constexpr int pack_factor_u32 = 32 / BITS;

// Number of uint32 words each thread loads per K-iteration.
// Chosen so that values_per_thread = 16 for all bit widths.
template <int BITS>
inline constexpr int packs_per_thread = 16 / pack_factor_u32<BITS>;
// 4-bit: 16/8=2, 2-bit: 16/16=1, 8-bit: 16/4=4

// Number of quantized values each thread processes per K-iteration.
template <int BITS>
inline constexpr int values_per_thread = 16;

// Number of K-elements consumed per warp per iteration.
// = values_per_thread * WARP_SIZE = 16 * 32 = 512
inline constexpr int block_size_k = values_per_thread<4> * WARP_SIZE;

// Number of output rows computed per thread block.
inline constexpr int ROWS_PER_BLOCK = 8;

// --- Warp reduction ---

__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

// --- Dequant-and-dot: integer dot product + x-sum accumulation ---
//
// Metal-compatible accumulation: accumulates raw integer dot product and
// x-sum separately. The caller applies scale and bias ONCE per group:
//   result += scale * total_qdot + bias * total_xsum
//
// This matches Metal's qdot() which returns scale * accum + sum * bias,
// where accum and sum span all values_per_thread elements at once.
//
// The naive per-element form `acc += x[i] * (scale * q[i] + bias)` is
// mathematically equivalent but produces different float32 rounding due to
// a different number of scale/bias multiply operations, causing LLM output
// to degenerate into repetitive loops after ~10 tokens.

template <int BITS>
__device__ __forceinline__ void dequant_and_dot(
    uint32_t packed,
    const float* __restrict__ x_local,
    float& qdot_acc,
    float& x_sum)
{
  constexpr int pf = pack_factor_u32<BITS>;
  constexpr uint32_t mask = (1u << BITS) - 1u;

  #pragma unroll
  for (int i = 0; i < pf; i++) {
    float q = static_cast<float>((packed >> (i * BITS)) & mask);
    qdot_acc += x_local[i] * q;
    x_sum += x_local[i];
  }
}

// --- Type conversion helpers ---

__device__ __forceinline__ float to_float(__half x) {
  return __half2float(x);
}

__device__ __forceinline__ float to_float(hip_bfloat16 x) {
  return static_cast<float>(x);
}

__device__ __forceinline__ float to_float(float x) {
  return x;
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ hip_bfloat16 from_float<hip_bfloat16>(float x) {
  return hip_bfloat16(x);
}

template <>
__device__ __forceinline__ float from_float<float>(float x) {
  return x;
}

} // namespace mlx::core::rocm
