// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/quantized/quantized.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <rocblas/rocblas.h>

namespace mlx::core {

namespace {

inline array ensure_row_contiguous_matrix(
    const array& x,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

} // namespace

namespace rocm {

template <int BITS>
__device__ inline uint8_t unpack_packed_value(
    const uint8_t* packed_row,
    int k,
    int row_bytes) {
  constexpr uint8_t mask = (1u << BITS) - 1u;
  if constexpr (BITS == 2 || BITS == 4 || BITS == 8) {
    constexpr int pack_factor = 8 / BITS;
    int pack_idx = k / pack_factor;
    int bit_offset = (k % pack_factor) * BITS;
    return (packed_row[pack_idx] >> bit_offset) & mask;
  } else {
    int bit_index = k * BITS;
    int byte_idx = bit_index >> 3;
    int bit_offset = bit_index & 0x7;

    uint32_t window = static_cast<uint32_t>(packed_row[byte_idx]);
    if (byte_idx + 1 < row_bytes) {
      window |= static_cast<uint32_t>(packed_row[byte_idx + 1]) << 8;
    }
    return static_cast<uint8_t>((window >> bit_offset) & mask);
  }
}

__device__ inline float fp4_e2m1_to_float(uint8_t val) {
  switch (val & 0xF) {
    case 0x0:
      return 0.0f;
    case 0x1:
      return 0.5f;
    case 0x2:
      return 1.0f;
    case 0x3:
      return 1.5f;
    case 0x4:
      return 2.0f;
    case 0x5:
      return 3.0f;
    case 0x6:
      return 4.0f;
    case 0x7:
      return 6.0f;
    case 0x8:
      return -0.0f;
    case 0x9:
      return -0.5f;
    case 0xA:
      return -1.0f;
    case 0xB:
      return -1.5f;
    case 0xC:
      return -2.0f;
    case 0xD:
      return -3.0f;
    case 0xE:
      return -4.0f;
    case 0xF:
      return -6.0f;
    default:
      return 0.0f;
  }
}

__device__ inline float fp8_e4m3_to_float(uint8_t val) {
  uint32_t sign = (val >> 7) & 0x1;
  uint32_t exp = (val >> 3) & 0xF;
  uint32_t mant = val & 0x7;

  float result;
  if (exp == 0) {
    if (mant == 0) {
      result = 0.0f;
    } else {
      result = ldexpf(static_cast<float>(mant), -9);
    }
  } else if (exp == 15 && mant == 7) {
    result = __uint_as_float(0x7FC00000);
  } else {
    uint32_t float_exp = exp - 7 + 127;
    uint32_t float_mant = mant << 20;
    uint32_t bits = (sign << 31) | (float_exp << 23) | float_mant;
    result = __uint_as_float(bits);
  }

  return sign ? -fabsf(result) : result;
}

template <int GROUP_SIZE>
__device__ inline float fp_scale_to_float(uint8_t s) {
  if constexpr (GROUP_SIZE == 16) {
    return fp8_e4m3_to_float(s);
  } else {
    union {
      uint16_t i;
      hip_bfloat16 f;
    } out;
    out.i = (s == 0 ? 0x40 : (static_cast<uint16_t>(s) << 7));
    return static_cast<float>(out.f);
  }
}

template <typename ScaleT, int GROUP_SIZE, bool AFFINE>
__device__ inline float load_scale_value(ScaleT raw) {
  if constexpr (AFFINE) {
    return static_cast<float>(raw);
  } else {
    return fp_scale_to_float<GROUP_SIZE>(static_cast<uint8_t>(raw));
  }
}

template <int BITS, bool AFFINE>
__device__ inline float dequantize_value(uint8_t quant_val, float scale, float bias) {
  if constexpr (AFFINE) {
    return static_cast<float>(quant_val) * scale + bias;
  } else {
    (void)bias;
    if constexpr (BITS == 8) {
      return fp8_e4m3_to_float(quant_val) * scale;
    } else {
      return fp4_e2m1_to_float(quant_val) * scale;
    }
  }
}

// Quantized matrix-vector multiply kernel
// Performs: out = x @ dequantize(w, scales, biases)
// where w is quantized weights, scales and biases are per-group parameters
template <typename T, typename ScaleT, int BITS, int GROUP_SIZE, bool AFFINE>
__global__ void qmv_kernel(
    const T* __restrict__ x,           // [M, K]
    const uint8_t* __restrict__ w,     // [N, K * BITS / 8] packed
    const ScaleT* __restrict__ scales, // [N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases, // [N, K/GROUP_SIZE] or nullptr
    T* __restrict__ out,               // [M, N]
    int M,
    int N,
    int K,
    bool has_bias) {
  
  const int row = blockIdx.x;  // output row (M dimension)
  const int col = blockIdx.y * blockDim.x + threadIdx.x;  // output col (N dimension)
  
  if (row >= M || col >= N) return;
  
  float acc = 0.0f;
  
  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  
  const int row_bytes = (K * BITS) / 8;
  const uint8_t* w_row = w + col * row_bytes;

  for (int g = 0; g < num_groups; ++g) {
    float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(
        scales[col * num_groups + g]);
    float bias = has_bias ? static_cast<float>(biases[col * num_groups + g]) : 0.0f;
    
    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);
    
    for (int k = k_start; k < k_end; ++k) {
      uint8_t quant_val = unpack_packed_value<BITS>(w_row, k, row_bytes);
      float w_val = dequantize_value<BITS, AFFINE>(quant_val, scale, bias);
      
      // Accumulate
      acc += static_cast<float>(x[row * K + k]) * w_val;
    }
  }
  
  out[row * N + col] = static_cast<T>(acc);
}

// Transposed quantized matrix-vector multiply kernel
// Performs: out = x @ dequantize(w, scales, biases).T
template <typename T, typename ScaleT, int BITS, int GROUP_SIZE, bool AFFINE>
__global__ void qmv_t_kernel(
    const T* __restrict__ x,           // [M, K]
    const uint8_t* __restrict__ w,     // [N, K * BITS / 8] packed
    const ScaleT* __restrict__ scales, // [N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases, // [N, K/GROUP_SIZE] or nullptr
    T* __restrict__ out,               // [M, N]
    int M,
    int N,
    int K,
    bool has_bias) {
  
  const int row = blockIdx.x;  // output row (M dimension)
  const int col = blockIdx.y * blockDim.x + threadIdx.x;  // output col (N dimension)
  
  if (row >= M || col >= N) return;
  
  float acc = 0.0f;
  
  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  
  const int row_bytes = (K * BITS) / 8;
  const uint8_t* w_row = w + col * row_bytes;

  for (int g = 0; g < num_groups; ++g) {
    float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(
        scales[col * num_groups + g]);
    float bias = has_bias ? static_cast<float>(biases[col * num_groups + g]) : 0.0f;
    
    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);
    
    for (int k = k_start; k < k_end; ++k) {
      uint8_t quant_val = unpack_packed_value<BITS>(w_row, k, row_bytes);
      float w_val = dequantize_value<BITS, AFFINE>(quant_val, scale, bias);
      
      // Accumulate
      acc += static_cast<float>(x[row * K + k]) * w_val;
    }
  }
  
  out[row * N + col] = static_cast<T>(acc);
}

} // namespace rocm

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  out.set_data(allocator::malloc(out.nbytes()));

  // Make sure the last two dims of x and w, s, b are contiguous
  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array w = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;
  bool has_bias = (mode_ == QuantizationMode::Affine) && (inputs.size() == 4);
  if (has_bias) {
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
  }

  enc.set_input_array(x);
  enc.set_input_array(w);
  enc.set_input_array(scales);
  if (has_bias) {
    enc.set_input_array(biases.value());
  }
  enc.set_output_array(out);

  // Extract the matmul shapes
  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  int block_size = 256;
  dim3 grid((M + 0) / 1, (N + block_size - 1) / block_size);
  grid.x = M;

  const void* x_ptr = gpu_ptr<void>(x);
  const uint8_t* w_ptr = gpu_ptr<const uint8_t>(w);
  const void* scales_ptr = gpu_ptr<void>(scales);
  const void* biases_ptr = has_bias ? gpu_ptr<void>(*biases) : nullptr;
  void* out_ptr = gpu_ptr<void>(out);
  
  enc.launch_kernel([&, x_ptr, w_ptr, scales_ptr, biases_ptr, out_ptr](hipStream_t stream) {
    #define LAUNCH_QMV(T, ScaleT, BITS, GROUP_SIZE) \
      if (mode_ == QuantizationMode::Affine) { \
        if (transpose_) { \
          hipLaunchKernelGGL( \
              (rocm::qmv_t_kernel<T, ScaleT, BITS, GROUP_SIZE, true>), \
              grid, dim3(block_size), 0, stream, \
              static_cast<const T*>(x_ptr), w_ptr, \
              static_cast<const ScaleT*>(scales_ptr), \
              has_bias ? static_cast<const ScaleT*>(biases_ptr) : nullptr, \
              static_cast<T*>(out_ptr), M, N, K, has_bias); \
        } else { \
          hipLaunchKernelGGL( \
              (rocm::qmv_kernel<T, ScaleT, BITS, GROUP_SIZE, true>), \
              grid, dim3(block_size), 0, stream, \
              static_cast<const T*>(x_ptr), w_ptr, \
              static_cast<const ScaleT*>(scales_ptr), \
              has_bias ? static_cast<const ScaleT*>(biases_ptr) : nullptr, \
              static_cast<T*>(out_ptr), M, N, K, has_bias); \
        } \
      } else { \
        if (transpose_) { \
          hipLaunchKernelGGL( \
              (rocm::qmv_t_kernel<T, ScaleT, BITS, GROUP_SIZE, false>), \
              grid, dim3(block_size), 0, stream, \
              static_cast<const T*>(x_ptr), w_ptr, \
              static_cast<const ScaleT*>(scales_ptr), \
              has_bias ? static_cast<const ScaleT*>(biases_ptr) : nullptr, \
              static_cast<T*>(out_ptr), M, N, K, has_bias); \
        } else { \
          hipLaunchKernelGGL( \
              (rocm::qmv_kernel<T, ScaleT, BITS, GROUP_SIZE, false>), \
              grid, dim3(block_size), 0, stream, \
              static_cast<const T*>(x_ptr), w_ptr, \
              static_cast<const ScaleT*>(scales_ptr), \
              has_bias ? static_cast<const ScaleT*>(biases_ptr) : nullptr, \
              static_cast<T*>(out_ptr), M, N, K, has_bias); \
        } \
      }
    
    #define DISPATCH_GROUP_SIZE(T, ScaleT, BITS) \
      switch (group_size_) { \
        case 16: LAUNCH_QMV(T, ScaleT, BITS, 16); break; \
        case 32: LAUNCH_QMV(T, ScaleT, BITS, 32); break; \
        case 64: LAUNCH_QMV(T, ScaleT, BITS, 64); break; \
        case 128: LAUNCH_QMV(T, ScaleT, BITS, 128); break; \
        default: throw std::runtime_error("Unsupported group_size for QuantizedMatmul: " + std::to_string(group_size_)); \
      }
    
    #define DISPATCH_BITS_AFFINE(T, ScaleT) \
      switch (bits_) { \
        case 2: DISPATCH_GROUP_SIZE(T, ScaleT, 2); break; \
        case 3: DISPATCH_GROUP_SIZE(T, ScaleT, 3); break; \
        case 4: DISPATCH_GROUP_SIZE(T, ScaleT, 4); break; \
        case 5: DISPATCH_GROUP_SIZE(T, ScaleT, 5); break; \
        case 6: DISPATCH_GROUP_SIZE(T, ScaleT, 6); break; \
        case 8: DISPATCH_GROUP_SIZE(T, ScaleT, 8); break; \
        default: throw std::runtime_error("Unsupported bits for QuantizedMatmul: " + std::to_string(bits_)); \
      }

    #define DISPATCH_BITS_FP(T) \
      switch (bits_) { \
        case 4: DISPATCH_GROUP_SIZE(T, uint8_t, 4); break; \
        case 8: DISPATCH_GROUP_SIZE(T, uint8_t, 8); break; \
        default: throw std::runtime_error("Unsupported fp bits for QuantizedMatmul: " + std::to_string(bits_)); \
      }

    switch (x.dtype()) {
      case float32:
        if (mode_ == QuantizationMode::Affine) {
          DISPATCH_BITS_AFFINE(float, float);
        } else {
          DISPATCH_BITS_FP(float);
        }
        break;
      case float16:
        if (mode_ == QuantizationMode::Affine) {
          DISPATCH_BITS_AFFINE(__half, __half);
        } else {
          DISPATCH_BITS_FP(__half);
        }
        break;
      case bfloat16:
        if (mode_ == QuantizationMode::Affine) {
          DISPATCH_BITS_AFFINE(hip_bfloat16, hip_bfloat16);
        } else {
          DISPATCH_BITS_FP(hip_bfloat16);
        }
        break;
      default:
        throw std::runtime_error("Unsupported dtype for QuantizedMatmul");
    }
    
    #undef DISPATCH_BITS_FP
    #undef DISPATCH_BITS_AFFINE
    #undef DISPATCH_GROUP_SIZE
    #undef LAUNCH_QMV
  });
}

// GatherQMM kernel - gather-based quantized matrix multiply
namespace rocm {

template <typename T, typename ScaleT, int BITS, int GROUP_SIZE, bool AFFINE>
__global__ void gather_qmv_kernel(
    const T* __restrict__ x,           // [B, M, K]
    const uint8_t* __restrict__ w,     // [E, N, K * BITS / 8] packed
    const ScaleT* __restrict__ scales, // [E, N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases, // [E, N, K/GROUP_SIZE] or nullptr
    const uint32_t* __restrict__ lhs_indices, // [B]
    const uint32_t* __restrict__ rhs_indices, // [B]
    const Shape batch_shape,
    const Strides lhs_idx_strides,
    const Strides rhs_idx_strides,
    int batch_ndim,
    T* __restrict__ out,               // [B, M, N]
    int B,
    int M,
    int N,
    int K,
    int E,
    bool has_bias) {
  
  int batch = blockIdx.z;
  int row = blockIdx.x;  // output row (M dimension)
  int col = blockIdx.y * blockDim.x + threadIdx.x;  // output col (N dimension)
  
  if (batch >= B || row >= M || col >= N) return;

  int64_t lhs_idx_loc = 0;
  int64_t rhs_idx_loc = 0;
  if (batch_ndim == 1) {
    lhs_idx_loc = static_cast<int64_t>(batch) * lhs_idx_strides[0];
    rhs_idx_loc = static_cast<int64_t>(batch) * rhs_idx_strides[0];
  } else if (batch_ndim > 1) {
    elem_to_loc<int64_t>(
        static_cast<int64_t>(batch),
        batch_shape.data_,
        lhs_idx_strides.data_,
        rhs_idx_strides.data_,
        batch_ndim,
        lhs_idx_loc,
        rhs_idx_loc);
  }

  uint32_t lhs_idx = lhs_indices[lhs_idx_loc];
  uint32_t rhs_idx = rhs_indices[rhs_idx_loc];
  
  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = (K * BITS) / 8;

  const T* x_ptr = x + lhs_idx * M * K + row * K;
  const uint8_t* w_ptr = w + rhs_idx * N * row_bytes + col * row_bytes;
  const ScaleT* scales_ptr = scales + rhs_idx * N * num_groups + col * num_groups;
  const ScaleT* biases_ptr =
      has_bias ? biases + rhs_idx * N * num_groups + col * num_groups : nullptr;

  float acc = 0.0f;
  
  for (int g = 0; g < num_groups; ++g) {
    float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_ptr[g]);
    float bias = has_bias ? static_cast<float>(biases_ptr[g]) : 0.0f;
    
    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);
    
    for (int k = k_start; k < k_end; ++k) {
      uint8_t quant_val = unpack_packed_value<BITS>(w_ptr, k, row_bytes);
      float w_val = dequantize_value<BITS, AFFINE>(quant_val, scale, bias);
      
      // Accumulate
      acc += static_cast<float>(x_ptr[k]) * w_val;
    }
  }
  
  out[batch * M * N + row * N + col] = static_cast<T>(acc);
}

} // namespace rocm

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  out.set_data(allocator::malloc(out.nbytes()));

  // Make sure the last two dims of x and w, s, b are contiguous
  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array w = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;
  bool has_bias = (mode_ == QuantizationMode::Affine) && (inputs.size() == 6);
  if (has_bias) {
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
  }
  const array& lhs_indices = inputs[inputs.size() - 2];
  const array& rhs_indices = inputs[inputs.size() - 1];

  auto [batch_shape, batch_strides] = collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  auto batch_shape_param = const_param(batch_shape);
  auto lhs_idx_strides_param = const_param<MAX_NDIM, int64_t>(batch_strides[0]);
  auto rhs_idx_strides_param = const_param<MAX_NDIM, int64_t>(batch_strides[1]);
  int batch_ndim = batch_shape.size();

  enc.set_input_array(x);
  enc.set_input_array(w);
  enc.set_input_array(scales);
  if (has_bias) {
    enc.set_input_array(biases.value());
  }
  enc.set_input_array(lhs_indices);
  enc.set_input_array(rhs_indices);
  enc.set_output_array(out);

  // Extract the matmul shapes
  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;
  int E = w.size() / w.shape(-1) / w.shape(-2);

  int block_size = 256;
  dim3 grid(M, (N + block_size - 1) / block_size, B);

  const void* x_ptr = gpu_ptr<void>(x);
  const uint8_t* w_ptr = gpu_ptr<const uint8_t>(w);
  const void* scales_ptr = gpu_ptr<void>(scales);
  const void* biases_ptr = has_bias ? gpu_ptr<void>(*biases) : nullptr;
  const uint32_t* lhs_indices_ptr = gpu_ptr<const uint32_t>(lhs_indices);
  const uint32_t* rhs_indices_ptr = gpu_ptr<const uint32_t>(rhs_indices);
  void* out_ptr = gpu_ptr<void>(out);
  
  enc.launch_kernel([
                        &,
                        x_ptr,
                        w_ptr,
                        scales_ptr,
                        biases_ptr,
                        lhs_indices_ptr,
                        rhs_indices_ptr,
                        out_ptr](hipStream_t stream) {
    #define LAUNCH_GATHER_QMV(T, ScaleT, BITS, GROUP_SIZE) \
      if (mode_ == QuantizationMode::Affine) { \
        hipLaunchKernelGGL( \
            (rocm::gather_qmv_kernel<T, ScaleT, BITS, GROUP_SIZE, true>), \
            grid, dim3(block_size), 0, stream, \
            static_cast<const T*>(x_ptr), w_ptr, \
            static_cast<const ScaleT*>(scales_ptr), \
            has_bias ? static_cast<const ScaleT*>(biases_ptr) : nullptr, \
            lhs_indices_ptr, rhs_indices_ptr, \
            batch_shape_param, lhs_idx_strides_param, rhs_idx_strides_param, \
            batch_ndim, static_cast<T*>(out_ptr), B, M, N, K, E, has_bias); \
      } else { \
        hipLaunchKernelGGL( \
            (rocm::gather_qmv_kernel<T, ScaleT, BITS, GROUP_SIZE, false>), \
            grid, dim3(block_size), 0, stream, \
            static_cast<const T*>(x_ptr), w_ptr, \
            static_cast<const ScaleT*>(scales_ptr), \
            has_bias ? static_cast<const ScaleT*>(biases_ptr) : nullptr, \
            lhs_indices_ptr, rhs_indices_ptr, \
            batch_shape_param, lhs_idx_strides_param, rhs_idx_strides_param, \
            batch_ndim, static_cast<T*>(out_ptr), B, M, N, K, E, has_bias); \
      }
    
    #define DISPATCH_GROUP_SIZE_GATHER(T, ScaleT, BITS) \
      switch (group_size_) { \
        case 16: LAUNCH_GATHER_QMV(T, ScaleT, BITS, 16); break; \
        case 32: LAUNCH_GATHER_QMV(T, ScaleT, BITS, 32); break; \
        case 64: LAUNCH_GATHER_QMV(T, ScaleT, BITS, 64); break; \
        case 128: LAUNCH_GATHER_QMV(T, ScaleT, BITS, 128); break; \
        default: throw std::runtime_error("Unsupported group_size for GatherQMM: " + std::to_string(group_size_)); \
      }
    
    #define DISPATCH_BITS_GATHER_AFFINE(T, ScaleT) \
      switch (bits_) { \
        case 2: DISPATCH_GROUP_SIZE_GATHER(T, ScaleT, 2); break; \
        case 3: DISPATCH_GROUP_SIZE_GATHER(T, ScaleT, 3); break; \
        case 4: DISPATCH_GROUP_SIZE_GATHER(T, ScaleT, 4); break; \
        case 5: DISPATCH_GROUP_SIZE_GATHER(T, ScaleT, 5); break; \
        case 6: DISPATCH_GROUP_SIZE_GATHER(T, ScaleT, 6); break; \
        case 8: DISPATCH_GROUP_SIZE_GATHER(T, ScaleT, 8); break; \
        default: throw std::runtime_error("Unsupported bits for GatherQMM: " + std::to_string(bits_)); \
      }

    #define DISPATCH_BITS_GATHER_FP(T) \
      switch (bits_) { \
        case 4: DISPATCH_GROUP_SIZE_GATHER(T, uint8_t, 4); break; \
        case 8: DISPATCH_GROUP_SIZE_GATHER(T, uint8_t, 8); break; \
        default: throw std::runtime_error("Unsupported fp bits for GatherQMM: " + std::to_string(bits_)); \
      }
    
    switch (x.dtype()) {
      case float32:
        if (mode_ == QuantizationMode::Affine) {
          DISPATCH_BITS_GATHER_AFFINE(float, float);
        } else {
          DISPATCH_BITS_GATHER_FP(float);
        }
        break;
      case float16:
        if (mode_ == QuantizationMode::Affine) {
          DISPATCH_BITS_GATHER_AFFINE(__half, __half);
        } else {
          DISPATCH_BITS_GATHER_FP(__half);
        }
        break;
      case bfloat16:
        if (mode_ == QuantizationMode::Affine) {
          DISPATCH_BITS_GATHER_AFFINE(hip_bfloat16, hip_bfloat16);
        } else {
          DISPATCH_BITS_GATHER_FP(hip_bfloat16);
        }
        break;
      default:
        throw std::runtime_error("Unsupported dtype for GatherQMM");
    }
    
    #undef DISPATCH_BITS_GATHER_FP
    #undef DISPATCH_BITS_GATHER_AFFINE
    #undef DISPATCH_GROUP_SIZE_GATHER
    #undef LAUNCH_GATHER_QMV
  });
}

} // namespace mlx::core
