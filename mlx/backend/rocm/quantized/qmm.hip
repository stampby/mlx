// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/gemms/naive_gemm.h"
#include "mlx/backend/rocm/gemms/rocblas_gemm.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/quantized/quantized.h"
#include "mlx/primitives.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <list>
#include <mutex>
#include <unordered_map>

namespace mlx::core {

namespace {

template <typename T>
struct local_type_identity {
  using type = T;
};

inline array ensure_row_contiguous_matrix(
    const array& x,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  if (x.ndim() == 0) {
    return x;
  }

  bool row_major_contiguous = true;
  int64_t expected_stride = 1;
  for (int i = x.ndim() - 1; i >= 0; --i) {
    if (x.shape(i) > 1) {
      if (x.strides()[i] != expected_stride) {
        row_major_contiguous = false;
        break;
      }
      expected_stride *= x.shape(i);
    }
  }

  if (row_major_contiguous) {
    return x;
  }

  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

inline int parse_cols_per_block_env(const char* env_name) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return 0;
  }

  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0') {
    return 0;
  }

  return (value == 4 || value == 8 || value == 16 || value == 32 || value == 64)
      ? static_cast<int>(value)
      : 0;
}

inline int parse_threads_per_col_env(const char* env_name) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return 0;
  }

  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0') {
    return 0;
  }

  return (value == 16 || value == WARP_SIZE) ? static_cast<int>(value) : 0;
}

inline bool parse_warp_kernel_env(const char* env_name, bool default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }

  if (raw[0] == '0' && raw[1] == '\0') {
    return false;
  }
  if (raw[0] == '1' && raw[1] == '\0') {
    return true;
  }
  return default_value;
}

inline int parse_positive_int_env(const char* env_name, int default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }

  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value <= 0) {
    return default_value;
  }
  return static_cast<int>(value);
}

inline size_t parse_non_negative_size_t_env(
    const char* env_name,
    size_t default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }

  char* end = nullptr;
  unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw || *end != '\0') {
    return default_value;
  }
  return static_cast<size_t>(value);
}

inline int parse_non_negative_int_env(const char* env_name, int default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }

  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value < 0) {
    return default_value;
  }
  return static_cast<int>(value);
}

// Check if rocBLAS dequant fast path should be used
// Default ON
inline bool use_rocblas_dequant_path() {
  static bool checked = false;
  static bool enabled = true;
  if (!checked) {
    const char* raw = std::getenv("MLX_ROCM_QMM_DEQUANT_GEMM");
    if (raw != nullptr) {
      enabled = (raw[0] == '1' && raw[1] == '\0');
    }
    checked = true;
  }
  return enabled;
}

inline bool has_only_singleton_batch_dims(const array& x) {
  if (x.ndim() <= 2) {
    return true;
  }
  for (int i = 0; i < x.ndim() - 2; ++i) {
    if (x.shape(i) != 1) {
      return false;
    }
  }
  return true;
}

inline int select_qmv_cols_per_block(int K, int N, int bits) {
  int env_cols = parse_cols_per_block_env("MLX_ROCM_QMV_COLS_PER_BLOCK");
  if (env_cols > 0) {
    return env_cols;
  }

  (void)K;

  if (N < 256) {
    return 4;
  }
  if (K <= 1024) {
    return (N < 1024) ? 8 : 16;
  }
  if (bits == 8) {
    if (N < 1024) {
      return 8;
    }
    if (N < 4096) {
      return 32;
    }
    return 16;
  }
  if (N < 1024) {
    return 8;
  }
  return 16;
}

inline int select_qmv_threads_per_col(int K, int N, int bits, int batch_count) {
  int threads_per_col = 16;
  if (WARP_SIZE == 32) {
    bool quant_bits_supported =
        (bits == 2 || bits == 4 || bits == 5 || bits == 6 || bits == 8);
    bool large_decode_like = (batch_count == 1) && (N >= 4096 || K >= 4096);
    if (quant_bits_supported && large_decode_like) {
      threads_per_col = WARP_SIZE;
    }
  }
  return threads_per_col;
}

inline bool should_use_dequant_gemm_path(
    int M,
    int N,
    int K,
    int batch_count,
    bool non_batched,
    bool can_use_batched_qmv) {
  int env_threshold =
      parse_positive_int_env("MLX_ROCM_QMM_DEQUANT_M_THRESHOLD", -1);
  if (env_threshold > 0) {
    return M >= env_threshold;
  }

  if (batch_count > 1) {
    if (!can_use_batched_qmv) {
      return true;
    }
    if (M <= 4) {
      return false;
    }
    if (M >= 32) {
      return true;
    }
    return (N >= 4096 && K >= 2048) || (N >= 8192 && M >= 8);
  }

  if (!non_batched) {
    return M >= 24;
  }

  if (M <= 8) {
    return false;
  }
  if (M >= 64) {
    return true;
  }
  if (K <= 1024 && N <= 2048) {
    return false;
  }
  if (N >= 8192 && K >= 2048) {
    return M >= 16;
  }
  return M >= 24;
}

struct DequantCacheKey {
  std::uintptr_t w_ptr;
  std::uintptr_t scales_ptr;
  std::uintptr_t biases_ptr;
  int group_size;
  int bits;
  int stream_index;
  bool transpose;
  Dtype dtype;

  bool operator==(const DequantCacheKey& other) const {
    return w_ptr == other.w_ptr && scales_ptr == other.scales_ptr &&
        biases_ptr == other.biases_ptr && group_size == other.group_size &&
        bits == other.bits && stream_index == other.stream_index &&
        transpose == other.transpose && dtype == other.dtype;
  }
};

struct DequantCacheKeyHasher {
  size_t operator()(const DequantCacheKey& key) const {
    size_t h = std::hash<std::uintptr_t>{}(key.w_ptr);
    h ^= std::hash<std::uintptr_t>{}(key.scales_ptr) + 0x9e3779b9 + (h << 6) +
        (h >> 2);
    h ^= std::hash<std::uintptr_t>{}(key.biases_ptr) + 0x9e3779b9 + (h << 6) +
        (h >> 2);
    h ^= std::hash<int>{}(key.group_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.bits) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.stream_index) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(static_cast<int>(key.transpose)) + 0x9e3779b9 +
        (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(static_cast<int>(key.dtype.val())) + 0x9e3779b9 +
        (h << 6) + (h >> 2);
    return h;
  }
};

struct DequantCacheEntry {
  array weight;
  array w_source;
  array scales_source;
  std::optional<array> biases_source;
  size_t bytes;
  std::list<DequantCacheKey>::iterator lru_it;
};

inline int dequant_cache_capacity() {
  static int capacity = []() {
    const char* raw = std::getenv("MLX_ROCM_QMM_DEQUANT_CACHE_SIZE");
    if (raw == nullptr || *raw == '\0') {
      return 8;
    }
    char* end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 0) {
      return 8;
    }
    return static_cast<int>(value);
  }();
  return capacity;
}

inline size_t dequant_cache_max_bytes() {
  static size_t max_bytes = parse_non_negative_size_t_env(
      "MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES", 256ULL * 1024ULL * 1024ULL);
  return max_bytes;
}

inline int qmm_gemm_solution_index_f32(bool batched) {
  static int single_index =
      parse_non_negative_int_env("MLX_ROCM_GEMM_F32_SOLUTION_INDEX", 0);
  static int batched_index = parse_non_negative_int_env(
      "MLX_ROCM_GEMM_F32_BATCHED_SOLUTION_INDEX", -1);
  if (!batched) {
    return single_index;
  }
  return batched_index >= 0 ? batched_index : single_index;
}

inline int qmm_gemm_solution_index_bf16(bool batched) {
  static int single_index =
      parse_non_negative_int_env("MLX_ROCM_GEMM_BF16_SOLUTION_INDEX", 0);
  static int batched_index = parse_non_negative_int_env(
      "MLX_ROCM_GEMM_BF16_BATCHED_SOLUTION_INDEX", -1);
  if (!batched) {
    return single_index;
  }
  return batched_index >= 0 ? batched_index : single_index;
}

inline rocblas_operation to_rocblas_op(bool transpose) {
  return transpose ? rocblas_operation_transpose : rocblas_operation_none;
}

void dequant_rocblas_gemm(
    rocm::CommandEncoder& enc,
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
  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* c_ptr = gpu_ptr<void>(c);

  enc.launch_kernel([&, a_ptr, b_ptr, c_ptr](hipStream_t stream) {
    enc.device().set_rocblas_stream(stream);
    rocblas_handle handle = enc.device().get_rocblas_handle();

    rocblas_operation op_a = to_rocblas_op(transpose_a);
    rocblas_operation op_b = to_rocblas_op(transpose_b);

    switch (dtype) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = qmm_gemm_solution_index_f32(false);
        static std::atomic<bool> solution_valid{true};

        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          rocblas_status status = rocblas_gemm_ex(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_f32_r,
              ldb,
              a_ptr,
              rocblas_datatype_f32_r,
              lda,
              &beta_f,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_solution_index,
              solution_index,
              0);
          if (status != rocblas_status_success) {
            solution_valid.store(false, std::memory_order_relaxed);
            rocblas_sgemm(
                handle,
                op_b,
                op_a,
                N,
                M,
                K,
                &alpha_f,
                static_cast<const float*>(b_ptr),
                ldb,
                static_cast<const float*>(a_ptr),
                lda,
                &beta_f,
                static_cast<float*>(c_ptr),
                ldc);
          }
        } else {
          rocblas_sgemm(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const float*>(b_ptr),
              ldb,
              static_cast<const float*>(a_ptr),
              lda,
              &beta_f,
              static_cast<float*>(c_ptr),
              ldc);
        }
        break;
      }
      case float16: {
        rocblas_half alpha_h, beta_h;
        __half alpha_f16 = static_cast<__half>(alpha);
        __half beta_f16 = static_cast<__half>(beta);
        std::memcpy(&alpha_h, &alpha_f16, sizeof(rocblas_half));
        std::memcpy(&beta_h, &beta_f16, sizeof(rocblas_half));
        rocblas_hgemm(
            handle,
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(b_ptr),
            ldb,
            reinterpret_cast<const rocblas_half*>(a_ptr),
            lda,
            &beta_h,
            reinterpret_cast<rocblas_half*>(c_ptr),
            ldc);
        break;
      }
      case bfloat16: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = qmm_gemm_solution_index_bf16(false);
        static std::atomic<bool> solution_valid{true};

        rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          algo = rocblas_gemm_algo_solution_index;
        } else {
          solution_index = 0;
        }

        rocblas_status status = rocblas_gemm_ex(
            handle,
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_f,
            b_ptr,
            rocblas_datatype_bf16_r,
            ldb,
            a_ptr,
            rocblas_datatype_bf16_r,
            lda,
            &beta_f,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
            rocblas_datatype_f32_r,
            algo,
            solution_index,
            0);

        if (status != rocblas_status_success &&
            algo == rocblas_gemm_algo_solution_index) {
          solution_valid.store(false, std::memory_order_relaxed);
          rocblas_gemm_ex(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_bf16_r,
              ldb,
              a_ptr,
              rocblas_datatype_bf16_r,
              lda,
              &beta_f,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_standard,
              0,
              0);
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported dtype for rocBLAS GEMM");
    }
  });
}

void dequant_rocblas_gemm_batched(
    rocm::CommandEncoder& enc,
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
  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* c_ptr = gpu_ptr<void>(c);

  enc.launch_kernel([&, a_ptr, b_ptr, c_ptr](hipStream_t stream) {
    enc.device().set_rocblas_stream(stream);
    rocblas_handle handle = enc.device().get_rocblas_handle();

    rocblas_operation op_a = to_rocblas_op(transpose_a);
    rocblas_operation op_b = to_rocblas_op(transpose_b);

    switch (dtype) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = qmm_gemm_solution_index_f32(true);
        static std::atomic<bool> solution_valid{true};

        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          rocblas_status status = rocblas_gemm_strided_batched_ex(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_f32_r,
              ldb,
              stride_b,
              a_ptr,
              rocblas_datatype_f32_r,
              lda,
              stride_a,
              &beta_f,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
              stride_c,
              c_ptr,
              rocblas_datatype_f32_r,
              ldc,
              stride_c,
              batch_count,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_solution_index,
              solution_index,
              0);
          if (status != rocblas_status_success) {
            solution_valid.store(false, std::memory_order_relaxed);
            rocblas_sgemm_strided_batched(
                handle,
                op_b,
                op_a,
                N,
                M,
                K,
                &alpha_f,
                static_cast<const float*>(b_ptr),
                ldb,
                stride_b,
                static_cast<const float*>(a_ptr),
                lda,
                stride_a,
                &beta_f,
                static_cast<float*>(c_ptr),
                ldc,
                stride_c,
                batch_count);
          }
        } else {
          rocblas_sgemm_strided_batched(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const float*>(b_ptr),
              ldb,
              stride_b,
              static_cast<const float*>(a_ptr),
              lda,
              stride_a,
              &beta_f,
              static_cast<float*>(c_ptr),
              ldc,
              stride_c,
              batch_count);
        }
        break;
      }
      case float16: {
        rocblas_half alpha_h, beta_h;
        __half alpha_f16 = static_cast<__half>(alpha);
        __half beta_f16 = static_cast<__half>(beta);
        std::memcpy(&alpha_h, &alpha_f16, sizeof(rocblas_half));
        std::memcpy(&beta_h, &beta_f16, sizeof(rocblas_half));
        rocblas_hgemm_strided_batched(
            handle,
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(b_ptr),
            ldb,
            stride_b,
            reinterpret_cast<const rocblas_half*>(a_ptr),
            lda,
            stride_a,
            &beta_h,
            reinterpret_cast<rocblas_half*>(c_ptr),
            ldc,
            stride_c,
            batch_count);
        break;
      }
      case bfloat16: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = qmm_gemm_solution_index_bf16(true);
        static std::atomic<bool> solution_valid{true};

        rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          algo = rocblas_gemm_algo_solution_index;
        } else {
          solution_index = 0;
        }

        rocblas_status status = rocblas_gemm_strided_batched_ex(
            handle,
            op_b,
            op_a,
            N,
            M,
            K,
            &alpha_f,
            b_ptr,
            rocblas_datatype_bf16_r,
            ldb,
            stride_b,
            a_ptr,
            rocblas_datatype_bf16_r,
            lda,
            stride_a,
            &beta_f,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
            stride_c,
            c_ptr,
            rocblas_datatype_bf16_r,
            ldc,
            stride_c,
            batch_count,
            rocblas_datatype_f32_r,
            algo,
            solution_index,
            0);

        if (status != rocblas_status_success &&
            algo == rocblas_gemm_algo_solution_index) {
          solution_valid.store(false, std::memory_order_relaxed);
          rocblas_gemm_strided_batched_ex(
              handle,
              op_b,
              op_a,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_bf16_r,
              ldb,
              stride_b,
              a_ptr,
              rocblas_datatype_bf16_r,
              lda,
              stride_a,
              &beta_f,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
              stride_c,
              c_ptr,
              rocblas_datatype_bf16_r,
              ldc,
              stride_c,
              batch_count,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_standard,
              0,
              0);
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported dtype for rocBLAS batched GEMM");
    }
  });
}

} // namespace

namespace rocm {

template <int BITS>
__device__ inline uint8_t
unpack_packed_value(const uint8_t* packed_row, int k, int row_bytes) {
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

template <int BITS>
__device__ inline uint8_t
unpack_packed_value_fast(const uint8_t* packed_row, int k, int row_bytes) {
  if constexpr (BITS == 8) {
    (void)row_bytes;
    return packed_row[k];
  } else if constexpr (BITS == 4) {
    (void)row_bytes;
    uint8_t packed = packed_row[k >> 1];
    return (k & 1) ? (packed >> 4) : (packed & 0xF);
  } else if constexpr (BITS == 2) {
    (void)row_bytes;
    uint8_t packed = packed_row[k >> 2];
    return (packed >> ((k & 0x3) * 2)) & 0x3;
  } else {
    return unpack_packed_value<BITS>(packed_row, k, row_bytes);
  }
}

template <int SUBGROUP_SIZE, typename T>
__device__ __forceinline__ T subgroup_reduce_sum_qmm(T val) {
  static_assert((SUBGROUP_SIZE & (SUBGROUP_SIZE - 1)) == 0);
  static_assert(SUBGROUP_SIZE <= WARP_SIZE);

#pragma unroll
  for (int offset = SUBGROUP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum_qmm(T val) {
  return subgroup_reduce_sum_qmm<WARP_SIZE>(val);
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

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t val) {
  // Use a simple array lookup or bit manipulation.
  // Actually, MI300 supports hardware fp8 conversion:
  // But we can just use a fast bit manipulation without branches.

  uint32_t sign = (val >> 7) & 0x1;
  uint32_t exp = (val >> 3) & 0xF;
  uint32_t mant = val & 0x7;

  if (exp == 0 && mant == 0) {
    return sign ? -0.0f : 0.0f;
  }

  uint32_t float_exp = exp == 0 ? 0 : exp - 7 + 127;
  // Handle subnormals approximately or cleanly if needed,
  // but for performance, we can just do:
  if (exp == 0) {
    float subnormal = static_cast<float>(mant) * 0.001953125f; // 2^-9
    return sign ? -subnormal : subnormal;
  }

  uint32_t float_mant = mant << 20;
  uint32_t bits = (sign << 31) | (float_exp << 23) | float_mant;
  return __uint_as_float(bits);
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
__device__ inline float
dequantize_value(uint8_t quant_val, float scale, float bias) {
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

template <
    typename T,
    typename ScaleT,
    int BITS,
    int GROUP_SIZE,
    bool AFFINE,
    int THREADS_PER_COL>
__global__ void __launch_bounds__(1024) qmv_warp_shared_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  const int lane = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int col = blockIdx.x * blockDim.y + warp_idx;
  const int row = blockIdx.y;

  const bool valid = (row < M) && (col < N);

  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;

  const T* x_row = (row < M) ? (x + row * K) : nullptr;
  const uint8_t* w_row = valid ? (w + col * row_bytes) : nullptr;
  const ScaleT* scales_row = valid ? (scales + col * num_groups) : nullptr;
  const ScaleT* biases_row =
      (valid && has_bias) ? (biases + col * num_groups) : nullptr;

  float acc = 0.0f;

  // We load a chunk of X into shared memory.
  // We use a chunk size of 1024 elements.
  constexpr int CHUNK_SIZE = 2048;
  __shared__ float shared_x[CHUNK_SIZE];

  for (int chunk_start = 0; chunk_start < K; chunk_start += CHUNK_SIZE) {
    int chunk_end = min(chunk_start + CHUNK_SIZE, K);
    int chunk_len = chunk_end - chunk_start;

    // Collaboratively load X chunk into shared memory
    int tid = warp_idx * blockDim.x + lane;
    for (int i = tid; i < chunk_len; i += blockDim.x * blockDim.y) {
      shared_x[i] = static_cast<float>(x_row[chunk_start + i]);
    }
    __syncthreads();

    if (valid) {
      int g_start = chunk_start / GROUP_SIZE;
      int g_end = (chunk_end + GROUP_SIZE - 1) / GROUP_SIZE;

      for (int g = g_start; g < g_end; ++g) {
        int k_start = max(g * GROUP_SIZE, chunk_start);
        int k_end_g = min((g + 1) * GROUP_SIZE, chunk_end);

        float scale =
            load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_row[g]);
        float bias_val = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;

        if constexpr (AFFINE) {
          float qx_acc0 = 0.0f, qx_acc1 = 0.0f, qx_acc2 = 0.0f, qx_acc3 = 0.0f;
          float x_group_sum = 0.0f;
          float qx_acc = 0.0f;

          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;

              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = static_cast<float>(w_packed & 0xFF);
              float w1 = static_cast<float>((w_packed >> 8) & 0xFF);
              float w2 = static_cast<float>((w_packed >> 16) & 0xFF);
              float w3 = static_cast<float>((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);

              if (has_bias)
                x_group_sum += x0 + x1 + x2 + x3;
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = static_cast<float>(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
              if (has_bias)
                x_group_sum += x_val;
            }
          } else if constexpr (BITS == 4) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            for (; k_start + k_local + 7 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed =
                  *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
              float w0 = static_cast<float>(w_packed & 0xF);
              float w1 = static_cast<float>((w_packed >> 4) & 0xF);
              float w2 = static_cast<float>((w_packed >> 8) & 0xF);
              float w3 = static_cast<float>((w_packed >> 12) & 0xF);
              float w4 = static_cast<float>((w_packed >> 16) & 0xF);
              float w5 = static_cast<float>((w_packed >> 20) & 0xF);
              float w6 = static_cast<float>((w_packed >> 24) & 0xF);
              float w7 = static_cast<float>((w_packed >> 28) & 0xF);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
              if (has_bias)
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias)
                x_group_sum += x_val;
            }
          } else if constexpr (BITS == 6) {
            // Process 8 weights at a time (48 bits = 6 bytes)
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            // Need at least 7 bytes of room after byte_idx for safe 8-byte load
            // row_bytes = (K * 6 + 7) / 8, so we need byte_idx + 7 < row_bytes
            int max_safe_k =
                ((row_bytes - 7) * 8) / 6; // Max k where 8-byte load is safe
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              // 8 weights * 6 bits = 48 bits, starting at bit position k*6
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              // Safe to load 8 bytes (we checked bounds above)
              uint64_t w_packed;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              // Extract 8 6-bit weights
              float w0 = static_cast<float>(w_packed & 0x3F);
              float w1 = static_cast<float>((w_packed >> 6) & 0x3F);
              float w2 = static_cast<float>((w_packed >> 12) & 0x3F);
              float w3 = static_cast<float>((w_packed >> 18) & 0x3F);
              float w4 = static_cast<float>((w_packed >> 24) & 0x3F);
              float w5 = static_cast<float>((w_packed >> 30) & 0x3F);
              float w6 = static_cast<float>((w_packed >> 36) & 0x3F);
              float w7 = static_cast<float>((w_packed >> 42) & 0x3F);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
              if (has_bias)
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias)
                x_group_sum += x_val;
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias)
                x_group_sum += x_val;
            }
          }
          acc += scale * qx_acc;
          if (has_bias)
            acc += bias_val * x_group_sum;
        } else {
          float qx_acc0 = 0.0f, qx_acc1 = 0.0f, qx_acc2 = 0.0f, qx_acc3 = 0.0f;
          float qx_acc = 0.0f;
          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = fp8_e4m3_to_float(w_packed & 0xFF);
              float w1 = fp8_e4m3_to_float((w_packed >> 8) & 0xFF);
              float w2 = fp8_e4m3_to_float((w_packed >> 16) & 0xFF);
              float w3 = fp8_e4m3_to_float((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = fp8_e4m3_to_float(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
            }
          } else if constexpr (BITS == 4) {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else if constexpr (BITS == 6) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            int max_safe_k = ((row_bytes - 7) * 8) / 6;
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              uint64_t w_packed = 0;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              float w0 =
                  dequantize_value<6, false>(w_packed & 0x3F, 1.0f, 0.0f);
              float w1 = dequantize_value<6, false>(
                  (w_packed >> 6) & 0x3F, 1.0f, 0.0f);
              float w2 = dequantize_value<6, false>(
                  (w_packed >> 12) & 0x3F, 1.0f, 0.0f);
              float w3 = dequantize_value<6, false>(
                  (w_packed >> 18) & 0x3F, 1.0f, 0.0f);
              float w4 = dequantize_value<6, false>(
                  (w_packed >> 24) & 0x3F, 1.0f, 0.0f);
              float w5 = dequantize_value<6, false>(
                  (w_packed >> 30) & 0x3F, 1.0f, 0.0f);
              float w6 = dequantize_value<6, false>(
                  (w_packed >> 36) & 0x3F, 1.0f, 0.0f);
              float w7 = dequantize_value<6, false>(
                  (w_packed >> 42) & 0x3F, 1.0f, 0.0f);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);
              qx_acc0 = fmaf(x4, w4, qx_acc0);
              qx_acc1 = fmaf(x5, w5, qx_acc1);
              qx_acc2 = fmaf(x6, w6, qx_acc2);
              qx_acc3 = fmaf(x7, w7, qx_acc3);
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          }
          acc += scale * qx_acc;
        }
      }
    }
    __syncthreads(); // ensure all warps are done before loading next chunk
  }

  acc = subgroup_reduce_sum_qmm<THREADS_PER_COL>(acc);
  if (valid && lane == 0) {
    out[row * N + col] = static_cast<T>(acc);
  }
}

template <
    typename T,
    typename ScaleT,
    int BITS,
    int GROUP_SIZE,
    bool AFFINE,
    int THREADS_PER_COL>
__global__ void __launch_bounds__(1024) qmv_warp_shared_batched_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    int64_t x_batch_stride,
    int64_t w_batch_stride,
    int64_t sb_batch_stride,
    int64_t out_batch_stride,
    bool has_bias) {
  const int lane = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int col = blockIdx.x * blockDim.y + warp_idx;
  const int row = blockIdx.y;
  const int batch = blockIdx.z;

  const bool valid = (row < M) && (col < N);

  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;

  const T* x_batch_ptr = x + static_cast<int64_t>(batch) * x_batch_stride;
  const uint8_t* w_batch_ptr = w + static_cast<int64_t>(batch) * w_batch_stride;
  const ScaleT* scales_batch_ptr =
      scales + static_cast<int64_t>(batch) * sb_batch_stride;
  const ScaleT* biases_batch_ptr = has_bias
      ? (biases + static_cast<int64_t>(batch) * sb_batch_stride)
      : nullptr;
  T* out_batch_ptr = out + static_cast<int64_t>(batch) * out_batch_stride;

  const T* x_row =
      (row < M) ? (x_batch_ptr + static_cast<int64_t>(row) * K) : nullptr;
  const uint8_t* w_row =
      valid ? (w_batch_ptr + static_cast<int64_t>(col) * row_bytes) : nullptr;
  const ScaleT* scales_row = valid
      ? (scales_batch_ptr + static_cast<int64_t>(col) * num_groups)
      : nullptr;
  const ScaleT* biases_row = (valid && has_bias)
      ? (biases_batch_ptr + static_cast<int64_t>(col) * num_groups)
      : nullptr;

  float acc = 0.0f;

  constexpr int CHUNK_SIZE = 2048;
  __shared__ float shared_x[CHUNK_SIZE];

  for (int chunk_start = 0; chunk_start < K; chunk_start += CHUNK_SIZE) {
    int chunk_end = min(chunk_start + CHUNK_SIZE, K);
    int chunk_len = chunk_end - chunk_start;

    int tid = warp_idx * blockDim.x + lane;
    for (int i = tid; i < chunk_len; i += blockDim.x * blockDim.y) {
      shared_x[i] = static_cast<float>(x_row[chunk_start + i]);
    }
    __syncthreads();

    if (valid) {
      int g_start = chunk_start / GROUP_SIZE;
      int g_end = (chunk_end + GROUP_SIZE - 1) / GROUP_SIZE;

      for (int g = g_start; g < g_end; ++g) {
        int k_start = max(g * GROUP_SIZE, chunk_start);
        int k_end_g = min((g + 1) * GROUP_SIZE, chunk_end);

        float scale =
            load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_row[g]);
        float bias_val = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;

        if constexpr (AFFINE) {
          float qx_acc0 = 0.0f;
          float qx_acc1 = 0.0f;
          float qx_acc2 = 0.0f;
          float qx_acc3 = 0.0f;
          float x_group_sum = 0.0f;
          float qx_acc = 0.0f;

          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;

              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = static_cast<float>(w_packed & 0xFF);
              float w1 = static_cast<float>((w_packed >> 8) & 0xFF);
              float w2 = static_cast<float>((w_packed >> 16) & 0xFF);
              float w3 = static_cast<float>((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);

              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3;
              }
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = static_cast<float>(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else if constexpr (BITS == 4) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            for (; k_start + k_local + 7 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed =
                  *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
              float w0 = static_cast<float>(w_packed & 0xF);
              float w1 = static_cast<float>((w_packed >> 4) & 0xF);
              float w2 = static_cast<float>((w_packed >> 8) & 0xF);
              float w3 = static_cast<float>((w_packed >> 12) & 0xF);
              float w4 = static_cast<float>((w_packed >> 16) & 0xF);
              float w5 = static_cast<float>((w_packed >> 20) & 0xF);
              float w6 = static_cast<float>((w_packed >> 24) & 0xF);
              float w7 = static_cast<float>((w_packed >> 28) & 0xF);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
              }
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else if constexpr (BITS == 6) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            int max_safe_k = ((row_bytes - 7) * 8) / 6;
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              uint64_t w_packed = 0;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              float w0 = static_cast<float>(w_packed & 0x3F);
              float w1 = static_cast<float>((w_packed >> 6) & 0x3F);
              float w2 = static_cast<float>((w_packed >> 12) & 0x3F);
              float w3 = static_cast<float>((w_packed >> 18) & 0x3F);
              float w4 = static_cast<float>((w_packed >> 24) & 0x3F);
              float w5 = static_cast<float>((w_packed >> 30) & 0x3F);
              float w6 = static_cast<float>((w_packed >> 36) & 0x3F);
              float w7 = static_cast<float>((w_packed >> 42) & 0x3F);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
              }
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          }
          acc += scale * qx_acc;
          if (has_bias) {
            acc += bias_val * x_group_sum;
          }
        } else {
          float qx_acc0 = 0.0f;
          float qx_acc1 = 0.0f;
          float qx_acc2 = 0.0f;
          float qx_acc3 = 0.0f;
          float qx_acc = 0.0f;
          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;

              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = fp8_e4m3_to_float(w_packed & 0xFF);
              float w1 = fp8_e4m3_to_float((w_packed >> 8) & 0xFF);
              float w2 = fp8_e4m3_to_float((w_packed >> 16) & 0xFF);
              float w3 = fp8_e4m3_to_float((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = fp8_e4m3_to_float(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
            }
          } else if constexpr (BITS == 4) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            for (; k_start + k_local + 7 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed =
                  *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
              float w0 = dequantize_value<4, false>(w_packed & 0xF, 1.0f, 0.0f);
              float w1 =
                  dequantize_value<4, false>((w_packed >> 4) & 0xF, 1.0f, 0.0f);
              float w2 =
                  dequantize_value<4, false>((w_packed >> 8) & 0xF, 1.0f, 0.0f);
              float w3 = dequantize_value<4, false>(
                  (w_packed >> 12) & 0xF, 1.0f, 0.0f);
              float w4 = dequantize_value<4, false>(
                  (w_packed >> 16) & 0xF, 1.0f, 0.0f);
              float w5 = dequantize_value<4, false>(
                  (w_packed >> 20) & 0xF, 1.0f, 0.0f);
              float w6 = dequantize_value<4, false>(
                  (w_packed >> 24) & 0xF, 1.0f, 0.0f);
              float w7 = dequantize_value<4, false>(
                  (w_packed >> 28) & 0xF, 1.0f, 0.0f);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else if constexpr (BITS == 6) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            int max_safe_k = ((row_bytes - 7) * 8) / 6;
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              uint64_t w_packed = 0;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              float w0 =
                  dequantize_value<6, false>(w_packed & 0x3F, 1.0f, 0.0f);
              float w1 = dequantize_value<6, false>(
                  (w_packed >> 6) & 0x3F, 1.0f, 0.0f);
              float w2 = dequantize_value<6, false>(
                  (w_packed >> 12) & 0x3F, 1.0f, 0.0f);
              float w3 = dequantize_value<6, false>(
                  (w_packed >> 18) & 0x3F, 1.0f, 0.0f);
              float w4 = dequantize_value<6, false>(
                  (w_packed >> 24) & 0x3F, 1.0f, 0.0f);
              float w5 = dequantize_value<6, false>(
                  (w_packed >> 30) & 0x3F, 1.0f, 0.0f);
              float w6 = dequantize_value<6, false>(
                  (w_packed >> 36) & 0x3F, 1.0f, 0.0f);
              float w7 = dequantize_value<6, false>(
                  (w_packed >> 42) & 0x3F, 1.0f, 0.0f);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          }
          acc += scale * qx_acc;
        }
      }
    }
    __syncthreads();
  }

  acc = subgroup_reduce_sum_qmm<THREADS_PER_COL>(acc);
  if (valid && lane == 0) {
    out_batch_ptr[static_cast<int64_t>(row) * N + col] = static_cast<T>(acc);
  }
}

template <
    typename T,
    typename ScaleT,
    int BITS,
    int GROUP_SIZE,
    bool AFFINE,
    int THREADS_PER_COL>
__global__ void qmv_warp_noshared_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  const int lane = threadIdx.x;
  const int col = blockIdx.x * blockDim.y + threadIdx.y;
  const int row = blockIdx.y;

  const bool row_valid = (row < M);
  const bool valid = row_valid && (col < N);

  constexpr int kThreadsPerCol = THREADS_PER_COL;
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;

  const T* x_row = row_valid ? (x + row * K) : nullptr;
  const uint8_t* w_row = valid ? (w + col * row_bytes) : nullptr;
  const ScaleT* scales_row = valid ? (scales + col * num_groups) : nullptr;
  const ScaleT* biases_row =
      (valid && has_bias) ? (biases + col * num_groups) : nullptr;

  float acc = 0.0f;

  for (int g = 0; g < num_groups; ++g) {
    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);

    if (valid) {
      float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_row[g]);
      float bias = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;

      if constexpr (AFFINE) {
        float qx_acc0 = 0.0f, qx_acc1 = 0.0f, qx_acc2 = 0.0f, qx_acc3 = 0.0f;
        float x_group_sum = 0.0f;
        float qx_acc = 0.0f;

        if constexpr (BITS == 8) {
          int k_local = lane * 4;
          int step = kThreadsPerCol * 4;
          for (; k_start + k_local + 3 < k_end; k_local += step) {
            int k = k_start + k_local;

            // Read 4 weights at once
            uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
            float w0 = static_cast<float>(w_packed & 0xFF);
            float w1 = static_cast<float>((w_packed >> 8) & 0xFF);
            float w2 = static_cast<float>((w_packed >> 16) & 0xFF);
            float w3 = static_cast<float>((w_packed >> 24) & 0xFF);

            float x0 = static_cast<float>(x_row[k]);
            float x1 = static_cast<float>(x_row[k + 1]);
            float x2 = static_cast<float>(x_row[k + 2]);
            float x3 = static_cast<float>(x_row[k + 3]);

            qx_acc0 = fmaf(x0, w0, qx_acc0);
            qx_acc1 = fmaf(x1, w1, qx_acc1);
            qx_acc2 = fmaf(x2, w2, qx_acc2);
            qx_acc3 = fmaf(x3, w3, qx_acc3);

            if (has_bias) {
              x_group_sum += x0 + x1 + x2 + x3;
            }
          }

          float qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;

          // Tail loop
          for (; k_start + k_local < k_end; k_local++) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            float w_val = static_cast<float>(w_row[k]);
            qx_acc = fmaf(x_val, w_val, qx_acc);
            if (has_bias)
              x_group_sum += x_val;
          }
        } else {
          for (int k_local = lane; k_start + k_local < k_end;
               k_local += kThreadsPerCol) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            uint8_t quant_val =
                unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
            qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
            if (has_bias)
              x_group_sum += x_val;
          }
        }

        float group_acc = scale * qx_acc;
        if (has_bias) {
          group_acc = fmaf(bias, x_group_sum, group_acc);
        }
        acc += group_acc;
      } else {
        float qx_acc0 = 0.0f, qx_acc1 = 0.0f, qx_acc2 = 0.0f, qx_acc3 = 0.0f;
        float qx_acc = 0.0f;

        if constexpr (BITS == 8) {
          int k_local = lane * 4;
          int step = kThreadsPerCol * 4;
          for (; k_start + k_local + 3 < k_end; k_local += step) {
            int k = k_start + k_local;

            // Read 4 weights at once
            uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
            float w0 = fp8_e4m3_to_float(w_packed & 0xFF);
            float w1 = fp8_e4m3_to_float((w_packed >> 8) & 0xFF);
            float w2 = fp8_e4m3_to_float((w_packed >> 16) & 0xFF);
            float w3 = fp8_e4m3_to_float((w_packed >> 24) & 0xFF);

            float x0 = static_cast<float>(x_row[k]);
            float x1 = static_cast<float>(x_row[k + 1]);
            float x2 = static_cast<float>(x_row[k + 2]);
            float x3 = static_cast<float>(x_row[k + 3]);

            qx_acc0 = fmaf(x0, w0, qx_acc0);
            qx_acc1 = fmaf(x1, w1, qx_acc1);
            qx_acc2 = fmaf(x2, w2, qx_acc2);
            qx_acc3 = fmaf(x3, w3, qx_acc3);
          }

          float qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;

          for (; k_start + k_local < k_end; k_local++) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            float w_val = fp8_e4m3_to_float(w_row[k]);
            qx_acc = fmaf(x_val, w_val, qx_acc);
          }
          acc += scale * qx_acc;
        } else {
          float qx_acc = 0.0f;
          for (int k_local = lane; k_start + k_local < k_end;
               k_local += kThreadsPerCol) {
            int k = k_start + k_local;
            float x_val = static_cast<float>(x_row[k]);
            uint8_t quant_val =
                unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
            qx_acc = fmaf(
                x_val,
                dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                qx_acc);
          }
          acc += scale * qx_acc;
        }
      }
    }
  }

  acc = subgroup_reduce_sum_qmm<kThreadsPerCol>(acc);
  if (valid && lane == 0) {
    out[row * N + col] = static_cast<T>(acc);
  }
}

template <typename T, typename ScaleT, int BITS, int GROUP_SIZE, bool AFFINE>
__global__ void qmv_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;

  if (row >= M || col >= N)
    return;

  float acc = 0.0f;
  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;
  const uint8_t* w_row = w + col * row_bytes;

  for (int g = 0; g < num_groups; ++g) {
    float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(
        scales[col * num_groups + g]);
    float bias =
        has_bias ? static_cast<float>(biases[col * num_groups + g]) : 0.0f;

    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);

    float qx_acc = 0.0f;
    if constexpr (BITS == 8) {
      int k = k_start;
      for (; k + 3 < k_end; k += 4) {
        uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
        float w0 = dequantize_value<8, AFFINE>(w_packed & 0xFF, scale, bias);
        float w1 =
            dequantize_value<8, AFFINE>((w_packed >> 8) & 0xFF, scale, bias);
        float w2 =
            dequantize_value<8, AFFINE>((w_packed >> 16) & 0xFF, scale, bias);
        float w3 =
            dequantize_value<8, AFFINE>((w_packed >> 24) & 0xFF, scale, bias);

        qx_acc += static_cast<float>(x[row * K + k]) * w0;
        qx_acc += static_cast<float>(x[row * K + k + 1]) * w1;
        qx_acc += static_cast<float>(x[row * K + k + 2]) * w2;
        qx_acc += static_cast<float>(x[row * K + k + 3]) * w3;
      }
      for (; k < k_end; ++k) {
        float w_val = dequantize_value<8, AFFINE>(w_row[k], scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    } else if constexpr (BITS == 6) {
      int k = k_start;
      int max_safe_k = ((row_bytes - 7) * 8) / 6;
      for (; k + 7 < k_end && k < max_safe_k; k += 8) {
        int byte_idx = (k * 6) / 8;
        int bit_offset = (k * 6) % 8;
        uint64_t w_packed = 0;
        memcpy(&w_packed, &w_row[byte_idx], 8);
        w_packed >>= bit_offset;

        float w0 = dequantize_value<6, AFFINE>(w_packed & 0x3F, scale, bias);
        float w1 =
            dequantize_value<6, AFFINE>((w_packed >> 6) & 0x3F, scale, bias);
        float w2 =
            dequantize_value<6, AFFINE>((w_packed >> 12) & 0x3F, scale, bias);
        float w3 =
            dequantize_value<6, AFFINE>((w_packed >> 18) & 0x3F, scale, bias);
        float w4 =
            dequantize_value<6, AFFINE>((w_packed >> 24) & 0x3F, scale, bias);
        float w5 =
            dequantize_value<6, AFFINE>((w_packed >> 30) & 0x3F, scale, bias);
        float w6 =
            dequantize_value<6, AFFINE>((w_packed >> 36) & 0x3F, scale, bias);
        float w7 =
            dequantize_value<6, AFFINE>((w_packed >> 42) & 0x3F, scale, bias);

        qx_acc += static_cast<float>(x[row * K + k]) * w0;
        qx_acc += static_cast<float>(x[row * K + k + 1]) * w1;
        qx_acc += static_cast<float>(x[row * K + k + 2]) * w2;
        qx_acc += static_cast<float>(x[row * K + k + 3]) * w3;
        qx_acc += static_cast<float>(x[row * K + k + 4]) * w4;
        qx_acc += static_cast<float>(x[row * K + k + 5]) * w5;
        qx_acc += static_cast<float>(x[row * K + k + 6]) * w6;
        qx_acc += static_cast<float>(x[row * K + k + 7]) * w7;
      }
      for (; k < k_end; ++k) {
        uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
        float w_val = dequantize_value<6, AFFINE>(quant_val, scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    } else {
      for (int k = k_start; k < k_end; ++k) {
        uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
        float w_val = dequantize_value<BITS, AFFINE>(quant_val, scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    }
    acc += qx_acc;
  }

  out[row * N + col] = static_cast<T>(acc);
}

template <typename T, typename ScaleT, int BITS, int GROUP_SIZE, bool AFFINE>
__global__ void qmv_t_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;

  if (row >= M || col >= N)
    return;

  float acc = 0.0f;
  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;
  const uint8_t* w_row = w + col * row_bytes;

  for (int g = 0; g < num_groups; ++g) {
    float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(
        scales[col * num_groups + g]);
    float bias =
        has_bias ? static_cast<float>(biases[col * num_groups + g]) : 0.0f;

    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);

    float qx_acc = 0.0f;
    if constexpr (BITS == 8) {
      int k = k_start;
      for (; k + 3 < k_end; k += 4) {
        uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
        float w0 = dequantize_value<8, AFFINE>(w_packed & 0xFF, scale, bias);
        float w1 =
            dequantize_value<8, AFFINE>((w_packed >> 8) & 0xFF, scale, bias);
        float w2 =
            dequantize_value<8, AFFINE>((w_packed >> 16) & 0xFF, scale, bias);
        float w3 =
            dequantize_value<8, AFFINE>((w_packed >> 24) & 0xFF, scale, bias);

        qx_acc += static_cast<float>(x[row * K + k]) * w0;
        qx_acc += static_cast<float>(x[row * K + k + 1]) * w1;
        qx_acc += static_cast<float>(x[row * K + k + 2]) * w2;
        qx_acc += static_cast<float>(x[row * K + k + 3]) * w3;
      }
      for (; k < k_end; ++k) {
        float w_val = dequantize_value<8, AFFINE>(w_row[k], scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    } else if constexpr (BITS == 6) {
      int k = k_start;
      int max_safe_k = ((row_bytes - 7) * 8) / 6;
      for (; k + 7 < k_end && k < max_safe_k; k += 8) {
        int byte_idx = (k * 6) / 8;
        int bit_offset = (k * 6) % 8;
        uint64_t w_packed = 0;
        memcpy(&w_packed, &w_row[byte_idx], 8);
        w_packed >>= bit_offset;

        float w0 = dequantize_value<6, AFFINE>(w_packed & 0x3F, scale, bias);
        float w1 =
            dequantize_value<6, AFFINE>((w_packed >> 6) & 0x3F, scale, bias);
        float w2 =
            dequantize_value<6, AFFINE>((w_packed >> 12) & 0x3F, scale, bias);
        float w3 =
            dequantize_value<6, AFFINE>((w_packed >> 18) & 0x3F, scale, bias);
        float w4 =
            dequantize_value<6, AFFINE>((w_packed >> 24) & 0x3F, scale, bias);
        float w5 =
            dequantize_value<6, AFFINE>((w_packed >> 30) & 0x3F, scale, bias);
        float w6 =
            dequantize_value<6, AFFINE>((w_packed >> 36) & 0x3F, scale, bias);
        float w7 =
            dequantize_value<6, AFFINE>((w_packed >> 42) & 0x3F, scale, bias);

        qx_acc += static_cast<float>(x[row * K + k]) * w0;
        qx_acc += static_cast<float>(x[row * K + k + 1]) * w1;
        qx_acc += static_cast<float>(x[row * K + k + 2]) * w2;
        qx_acc += static_cast<float>(x[row * K + k + 3]) * w3;
        qx_acc += static_cast<float>(x[row * K + k + 4]) * w4;
        qx_acc += static_cast<float>(x[row * K + k + 5]) * w5;
        qx_acc += static_cast<float>(x[row * K + k + 6]) * w6;
        qx_acc += static_cast<float>(x[row * K + k + 7]) * w7;
      }
      for (; k < k_end; ++k) {
        uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
        float w_val = dequantize_value<6, AFFINE>(quant_val, scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    } else {
      for (int k = k_start; k < k_end; ++k) {
        uint8_t quant_val = unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
        float w_val = dequantize_value<BITS, AFFINE>(quant_val, scale, bias);
        qx_acc += static_cast<float>(x[row * K + k]) * w_val;
      }
    }
    acc += qx_acc;
  }

  out[row * N + col] = static_cast<T>(acc);
}

} // namespace rocm

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);

  out.set_data(allocator::malloc(out.nbytes()));

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
  if (has_bias)
    enc.set_input_array(biases.value());
  enc.set_output_array(out);

  int K = x.shape(-1);
  int M = out.shape(-2);
  int N = out.shape(-1);

  int64_t matrix_size = static_cast<int64_t>(M) * N;
  int batch_count = static_cast<int>(out.size() / matrix_size);
  int x_batch_count = static_cast<int>(
      x.size() /
      (static_cast<int64_t>(x.shape(-2)) * static_cast<int64_t>(x.shape(-1))));
  int w_batch_count = static_cast<int>(
      w.size() /
      (static_cast<int64_t>(w.shape(-2)) * static_cast<int64_t>(w.shape(-1))));

  bool x_singleton_batch = has_only_singleton_batch_dims(x);
  bool w_singleton_batch = has_only_singleton_batch_dims(w);
  bool non_batched =
      (batch_count == 1) && x_singleton_batch && w_singleton_batch;

  bool bits_supported_by_qmv = (bits_ == 2 || bits_ == 4 || bits_ == 8) ||
      (mode_ == QuantizationMode::Affine && (bits_ == 5 || bits_ == 6));
  bool valid_x_batch = (x_batch_count == 1) || (x_batch_count == batch_count);
  bool valid_w_batch = (w_batch_count == 1) || (w_batch_count == batch_count);
  bool can_use_batched_qmv = transpose_ && bits_supported_by_qmv &&
      (batch_count > 1) && valid_x_batch && valid_w_batch;
  bool force_dequant_gemm = !transpose_ || !bits_supported_by_qmv ||
      ((batch_count > 1) && !can_use_batched_qmv) ||
      (w.ndim() > 2 && !w_singleton_batch && !can_use_batched_qmv);
  bool dequant_gemm_supported_mode = (mode_ == QuantizationMode::Affine);
  bool should_prefer_dequant = should_use_dequant_gemm_path(
      M, N, K, batch_count, non_batched, can_use_batched_qmv);

  // Dequant + rocBLAS GEMM path
  // Disable with MLX_ROCM_QMM_DEQUANT_GEMM=0 if needed
  if (dequant_gemm_supported_mode && d.is_rocblas_available() &&
      use_rocblas_dequant_path() &&
      (force_dequant_gemm || should_prefer_dequant)) {
    if (!((x_batch_count == 1) || (x_batch_count == batch_count))) {
      throw std::runtime_error(
          "Unsupported x batch shape for dequant GEMM fallback");
    }
    if (!((w_batch_count == 1) || (w_batch_count == batch_count))) {
      throw std::runtime_error(
          "Unsupported w batch shape for dequant GEMM fallback");
    }

    int dequant_rows = transpose_ ? N : K;
    int dequant_cols = transpose_ ? K : N;

    Shape w_dequant_shape = w.shape();
    w_dequant_shape[w_dequant_shape.size() - 2] = dequant_rows;
    w_dequant_shape[w_dequant_shape.size() - 1] = dequant_cols;

    array w_dequant(w_dequant_shape, x.dtype(), nullptr, {});
    bool cache_hit = false;
    int cache_cap = dequant_cache_capacity();
    size_t cache_max_bytes = dequant_cache_max_bytes();
    if (cache_cap > 0 && cache_max_bytes > 0) {
      static std::mutex cache_mutex;
      static std::list<DequantCacheKey> lru;
      static size_t cached_bytes = 0;
      static std::unordered_map<
          DequantCacheKey,
          DequantCacheEntry,
          DequantCacheKeyHasher>
          cache;

      DequantCacheKey key{
          reinterpret_cast<std::uintptr_t>(gpu_ptr<void>(w)),
          reinterpret_cast<std::uintptr_t>(gpu_ptr<void>(scales)),
          has_bias ? reinterpret_cast<std::uintptr_t>(gpu_ptr<void>(*biases))
                   : 0,
          group_size_,
          bits_,
          s.index,
          transpose_,
          x.dtype()};

      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end() && it->second.weight.shape() == w_dequant_shape) {
          lru.splice(lru.begin(), lru, it->second.lru_it);
          w_dequant = it->second.weight;
          cache_hit = true;
        }
      }

      if (!cache_hit) {
        w_dequant.set_data(allocator::malloc(w_dequant.nbytes()));

        if (mode_ == QuantizationMode::Affine) {
          affine_dequantize(
              w, scales, biases, w_dequant, group_size_, bits_, enc, s);
        } else {
          fp_dequantize(w, scales, w_dequant, group_size_, bits_, enc, s);
        }

        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it == cache.end()) {
          size_t entry_bytes = w_dequant.nbytes();
          if (entry_bytes <= cache_max_bytes) {
            lru.push_front(key);
            cache.emplace(
                key,
                DequantCacheEntry{
                    w_dequant,
                    w,
                    scales,
                    has_bias ? std::optional<array>(*biases) : std::nullopt,
                    entry_bytes,
                    lru.begin()});
            cached_bytes += entry_bytes;

            while (static_cast<int>(cache.size()) > cache_cap ||
                   cached_bytes > cache_max_bytes) {
              auto evict = lru.back();
              auto evict_it = cache.find(evict);
              if (evict_it != cache.end()) {
                cached_bytes -= evict_it->second.bytes;
                cache.erase(evict_it);
              }
              lru.pop_back();
            }
          }
        } else {
          size_t entry_bytes = w_dequant.nbytes();
          if (entry_bytes > cache_max_bytes) {
            cached_bytes -= it->second.bytes;
            lru.erase(it->second.lru_it);
            cache.erase(it);
          } else {
            cached_bytes -= it->second.bytes;
            it->second.w_source = w;
            it->second.scales_source = scales;
            it->second.biases_source =
                has_bias ? std::optional<array>(*biases) : std::nullopt;
            it->second.weight = w_dequant;
            it->second.bytes = entry_bytes;
            cached_bytes += it->second.bytes;
            lru.splice(lru.begin(), lru, it->second.lru_it);

            while (static_cast<int>(cache.size()) > cache_cap ||
                   cached_bytes > cache_max_bytes) {
              auto evict = lru.back();
              auto evict_it = cache.find(evict);
              if (evict_it != cache.end()) {
                cached_bytes -= evict_it->second.bytes;
                cache.erase(evict_it);
              }
              lru.pop_back();
            }
          }
        }
      }
    } else {
      w_dequant.set_data(allocator::malloc(w_dequant.nbytes()));

      if (mode_ == QuantizationMode::Affine) {
        affine_dequantize(
            w, scales, biases, w_dequant, group_size_, bits_, enc, s);
      } else {
        fp_dequantize(w, scales, w_dequant, group_size_, bits_, enc, s);
      }
    }

    if (!cache_hit) {
      enc.add_temporary(w_dequant);
    }

    int lda = K;
    int ldb = transpose_ ? K : N;

    if (batch_count == 1 && x_batch_count == 1 && w_batch_count == 1) {
      dequant_rocblas_gemm(
          enc,
          false,
          transpose_,
          M,
          N,
          K,
          1.0f,
          x,
          lda,
          w_dequant,
          ldb,
          0.0f,
          out,
          N,
          x.dtype());
    } else {
      int64_t stride_a =
          (x_batch_count == 1) ? 0 : static_cast<int64_t>(x.shape(-2)) * K;
      int64_t stride_b = (w_batch_count == 1)
          ? 0
          : static_cast<int64_t>(dequant_rows) * dequant_cols;
      int64_t stride_c = static_cast<int64_t>(M) * N;

      dequant_rocblas_gemm_batched(
          enc,
          false,
          transpose_,
          M,
          N,
          K,
          1.0f,
          x,
          lda,
          stride_a,
          w_dequant,
          ldb,
          stride_b,
          0.0f,
          out,
          N,
          stride_c,
          batch_count,
          x.dtype());
    }
    return;
  }

  bool use_fast_qmv = transpose_ && (non_batched || can_use_batched_qmv);
  use_fast_qmv = parse_warp_kernel_env("MLX_ROCM_QMV_USE_WARP", use_fast_qmv);
  if (can_use_batched_qmv) {
    use_fast_qmv = true;
  }

  int block_size = 256;
  dim3 grid(M, (N + block_size - 1) / block_size);

  int fast_threads_per_col =
      select_qmv_threads_per_col(K, N, bits_, batch_count);
  int fast_threads_env =
      parse_threads_per_col_env("MLX_ROCM_QMV_THREADS_PER_COL");
  if (fast_threads_env > 0)
    fast_threads_per_col = fast_threads_env;

  int fast_cols_per_block = select_qmv_cols_per_block(K, N, bits_);
  if (fast_threads_per_col == 16 && bits_ == 8 && N >= 2048) {
    fast_cols_per_block = std::max(fast_cols_per_block, 64);
  }
  int max_cols_per_block = rocm::kMaxThreadsPerBlock / fast_threads_per_col;
  while (fast_cols_per_block > max_cols_per_block)
    fast_cols_per_block /= 2;
  while (fast_cols_per_block > 1 && (N % fast_cols_per_block) != 0 &&
         fast_cols_per_block > 8) {
    fast_cols_per_block /= 2;
  }

  dim3 fast_block(fast_threads_per_col, fast_cols_per_block);
  dim3 fast_grid((N + fast_cols_per_block - 1) / fast_cols_per_block, M);
  dim3 fast_grid_batched(
      (N + fast_cols_per_block - 1) / fast_cols_per_block, M, batch_count);

  int64_t x_matrix_stride =
      static_cast<int64_t>(x.shape(-2)) * static_cast<int64_t>(x.shape(-1));
  int64_t w_matrix_stride = static_cast<int64_t>(w.shape(-2)) *
      static_cast<int64_t>(w.shape(-1)) *
      static_cast<int64_t>(size_of(w.dtype()));
  int num_groups = (K + group_size_ - 1) / group_size_;
  int64_t sb_matrix_stride =
      static_cast<int64_t>(w.shape(-2)) * static_cast<int64_t>(num_groups);
  int64_t out_matrix_stride = static_cast<int64_t>(M) * N;

  int64_t x_batch_stride = (x_batch_count == 1) ? 0 : x_matrix_stride;
  int64_t w_batch_stride = (w_batch_count == 1) ? 0 : w_matrix_stride;
  int64_t sb_batch_stride = (w_batch_count == 1) ? 0 : sb_matrix_stride;

  const void* x_ptr = gpu_ptr<void>(x);
  const uint8_t* w_ptr = gpu_ptr<const uint8_t>(w);
  const void* scales_ptr = gpu_ptr<void>(scales);
  const void* biases_ptr = has_bias ? gpu_ptr<void>(*biases) : nullptr;
  void* out_ptr = gpu_ptr<void>(out);

  enc.launch_kernel([&,
                     x_ptr,
                     w_ptr,
                     scales_ptr,
                     biases_ptr,
                     out_ptr,
                     fast_threads_per_col,
                     x_batch_stride,
                     w_batch_stride,
                     sb_batch_stride,
                     out_matrix_stride](hipStream_t stream) {
    auto launch_qmv =
        [&](auto type_tag, auto scale_tag, auto bits_tag, auto gs_tag) {
          using T = typename decltype(type_tag)::type;
          using ScaleT = typename decltype(scale_tag)::type;
          constexpr int BITS = bits_tag.value;
          constexpr int GROUP_SIZE = gs_tag.value;

          if (mode_ == QuantizationMode::Affine) {
            if (use_fast_qmv) {
              if (can_use_batched_qmv) {
                if (fast_threads_per_col == 16) {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          true,
                          16>),
                      fast_grid_batched,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      x_batch_stride,
                      w_batch_stride,
                      sb_batch_stride,
                      out_matrix_stride,
                      has_bias);
                } else {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          true,
                          WARP_SIZE>),
                      fast_grid_batched,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      x_batch_stride,
                      w_batch_stride,
                      sb_batch_stride,
                      out_matrix_stride,
                      has_bias);
                }
              } else {
                if (fast_threads_per_col == 16) {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          true,
                          16>),
                      fast_grid,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      has_bias);
                } else {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          true,
                          WARP_SIZE>),
                      fast_grid,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      has_bias);
                }
              }
            } else if (transpose_) {
              hipLaunchKernelGGL(
                  (rocm::qmv_t_kernel<T, ScaleT, BITS, GROUP_SIZE, true>),
                  grid,
                  dim3(block_size),
                  0,
                  stream,
                  (const T*)x_ptr,
                  w_ptr,
                  (const ScaleT*)scales_ptr,
                  (const ScaleT*)biases_ptr,
                  (T*)out_ptr,
                  M,
                  N,
                  K,
                  has_bias);
            } else {
              hipLaunchKernelGGL(
                  (rocm::qmv_kernel<T, ScaleT, BITS, GROUP_SIZE, true>),
                  grid,
                  dim3(block_size),
                  0,
                  stream,
                  (const T*)x_ptr,
                  w_ptr,
                  (const ScaleT*)scales_ptr,
                  (const ScaleT*)biases_ptr,
                  (T*)out_ptr,
                  M,
                  N,
                  K,
                  has_bias);
            }
          } else {
            if (use_fast_qmv) {
              if (can_use_batched_qmv) {
                if (fast_threads_per_col == 16) {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          false,
                          16>),
                      fast_grid_batched,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      x_batch_stride,
                      w_batch_stride,
                      sb_batch_stride,
                      out_matrix_stride,
                      has_bias);
                } else {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_batched_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          false,
                          WARP_SIZE>),
                      fast_grid_batched,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      x_batch_stride,
                      w_batch_stride,
                      sb_batch_stride,
                      out_matrix_stride,
                      has_bias);
                }
              } else {
                if (fast_threads_per_col == 16) {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          false,
                          16>),
                      fast_grid,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      has_bias);
                } else {
                  hipLaunchKernelGGL(
                      (rocm::qmv_warp_shared_kernel<
                          T,
                          ScaleT,
                          BITS,
                          GROUP_SIZE,
                          false,
                          WARP_SIZE>),
                      fast_grid,
                      fast_block,
                      0,
                      stream,
                      (const T*)x_ptr,
                      w_ptr,
                      (const ScaleT*)scales_ptr,
                      (const ScaleT*)biases_ptr,
                      (T*)out_ptr,
                      M,
                      N,
                      K,
                      has_bias);
                }
              }
            } else if (transpose_) {
              hipLaunchKernelGGL(
                  (rocm::qmv_t_kernel<T, ScaleT, BITS, GROUP_SIZE, false>),
                  grid,
                  dim3(block_size),
                  0,
                  stream,
                  (const T*)x_ptr,
                  w_ptr,
                  (const ScaleT*)scales_ptr,
                  (const ScaleT*)biases_ptr,
                  (T*)out_ptr,
                  M,
                  N,
                  K,
                  has_bias);
            } else {
              hipLaunchKernelGGL(
                  (rocm::qmv_kernel<T, ScaleT, BITS, GROUP_SIZE, false>),
                  grid,
                  dim3(block_size),
                  0,
                  stream,
                  (const T*)x_ptr,
                  w_ptr,
                  (const ScaleT*)scales_ptr,
                  (const ScaleT*)biases_ptr,
                  (T*)out_ptr,
                  M,
                  N,
                  K,
                  has_bias);
            }
          }
        };

    // Type aliases to avoid template angle brackets in macro args
    using float_id = local_type_identity<float>;
    using half_id = local_type_identity<__half>;
    using bf16_id = local_type_identity<hip_bfloat16>;
    using bits2 = std::integral_constant<int, 2>;
    using bits4 = std::integral_constant<int, 4>;
    using bits5 = std::integral_constant<int, 5>;
    using bits6 = std::integral_constant<int, 6>;
    using bits8 = std::integral_constant<int, 8>;
    using gs32 = std::integral_constant<int, 32>;
    using gs64 = std::integral_constant<int, 64>;
    using gs128 = std::integral_constant<int, 128>;

// Helper macro to dispatch group_size
#define DISPATCH_GROUP_SIZE(type_tag, scale_tag, bits_tag)   \
  do {                                                       \
    switch (group_size_) {                                   \
      case 32:                                               \
        launch_qmv(type_tag, scale_tag, bits_tag, gs32{});   \
        break;                                               \
      case 64:                                               \
        launch_qmv(type_tag, scale_tag, bits_tag, gs64{});   \
        break;                                               \
      case 128:                                              \
        launch_qmv(type_tag, scale_tag, bits_tag, gs128{});  \
        break;                                               \
      default:                                               \
        throw std::runtime_error(                            \
            "Unsupported group_size for QuantizedMatmul: " + \
            std::to_string(group_size_));                    \
    }                                                        \
  } while (0)

    if (x.dtype() == float32) {
      if (bits_ == 8)
        DISPATCH_GROUP_SIZE(float_id{}, float_id{}, bits8{});
      else if (bits_ == 5 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(float_id{}, float_id{}, bits5{});
      } else if (bits_ == 6 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(float_id{}, float_id{}, bits6{});
      } else if (bits_ == 4)
        DISPATCH_GROUP_SIZE(float_id{}, float_id{}, bits4{});
      else if (bits_ == 2)
        DISPATCH_GROUP_SIZE(float_id{}, float_id{}, bits2{});
      else
        throw std::runtime_error(
            "Unsupported bits for QuantizedMatmul float32: " +
            std::to_string(bits_));
    } else if (x.dtype() == float16) {
      if (bits_ == 8)
        DISPATCH_GROUP_SIZE(half_id{}, half_id{}, bits8{});
      else if (bits_ == 5 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(half_id{}, half_id{}, bits5{});
      } else if (bits_ == 6 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(half_id{}, half_id{}, bits6{});
      } else if (bits_ == 4)
        DISPATCH_GROUP_SIZE(half_id{}, half_id{}, bits4{});
      else if (bits_ == 2)
        DISPATCH_GROUP_SIZE(half_id{}, half_id{}, bits2{});
      else
        throw std::runtime_error(
            "Unsupported bits for QuantizedMatmul float16: " +
            std::to_string(bits_));
    } else if (x.dtype() == bfloat16) {
      if (bits_ == 8)
        DISPATCH_GROUP_SIZE(bf16_id{}, bf16_id{}, bits8{});
      else if (bits_ == 5 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(bf16_id{}, bf16_id{}, bits5{});
      } else if (bits_ == 6 && mode_ == QuantizationMode::Affine) {
        DISPATCH_GROUP_SIZE(bf16_id{}, bf16_id{}, bits6{});
      } else if (bits_ == 4)
        DISPATCH_GROUP_SIZE(bf16_id{}, bf16_id{}, bits4{});
      else if (bits_ == 2)
        DISPATCH_GROUP_SIZE(bf16_id{}, bf16_id{}, bits2{});
      else
        throw std::runtime_error(
            "Unsupported bits for QuantizedMatmul bfloat16: " +
            std::to_string(bits_));
    } else {
      throw std::runtime_error("Unsupported dtype for QuantizedMatmul");
    }

#undef DISPATCH_GROUP_SIZE
  });
}

namespace rocm {
template <
    typename T,
    typename ScaleT,
    int BITS,
    int GROUP_SIZE,
    bool AFFINE,
    int THREADS_PER_COL>
__global__ void __launch_bounds__(1024) gather_qmv_warp_shared_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    const uint32_t* __restrict__ lhs_indices,
    const uint32_t* __restrict__ rhs_indices,
    const rocm::Shape batch_shape,
    const rocm::Strides lhs_idx_strides,
    const rocm::Strides rhs_idx_strides,
    int batch_ndim,
    T* __restrict__ out,
    int B,
    int M,
    int N,
    int K,
    int E,
    bool has_bias,
    bool implicit_lhs = false,
    int64_t implicit_x_batch_stride = 0) {
  const int lane = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int col = blockIdx.y * blockDim.y + warp_idx;
  const int row = blockIdx.x;
  const int batch = blockIdx.z;

  if (batch >= B || row >= M) {
    return;
  }

  int64_t rhs_idx_loc = 0;
  int64_t lhs_idx_loc = 0;
  if (batch_ndim == 1) {
    rhs_idx_loc = static_cast<int64_t>(batch) * rhs_idx_strides[0];
    if (!implicit_lhs) {
      lhs_idx_loc = static_cast<int64_t>(batch) * lhs_idx_strides[0];
    }
  } else if (batch_ndim > 1) {
    int64_t elem = static_cast<int64_t>(batch);
    for (int i = batch_ndim - 1; i >= 0; --i) {
      int64_t coord = elem % batch_shape.data_[i];
      rhs_idx_loc += coord * rhs_idx_strides.data_[i];
      if (!implicit_lhs) {
        lhs_idx_loc += coord * lhs_idx_strides.data_[i];
      }
      elem /= batch_shape.data_[i];
    }
  }

  uint32_t lhs_idx = implicit_lhs ? 0u : lhs_indices[lhs_idx_loc];
  uint32_t rhs_idx = rhs_indices[rhs_idx_loc];

  const bool col_valid = col < N;
  const bool expert_valid = rhs_idx < static_cast<uint32_t>(E);
  const bool valid = col_valid && expert_valid;

  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = (K * BITS + 7) / 8;

  int64_t x_batch_stride = static_cast<int64_t>(M) * K;
  int64_t w_batch_stride = static_cast<int64_t>(N) * row_bytes;
  int64_t sb_batch_stride = static_cast<int64_t>(N) * num_groups;
  int64_t col_w_offset = static_cast<int64_t>(col) * row_bytes;
  int64_t col_sb_offset = static_cast<int64_t>(col) * num_groups;

  int64_t x_batch_offset = implicit_lhs
      ? (static_cast<int64_t>(batch) * implicit_x_batch_stride)
      : (static_cast<int64_t>(lhs_idx) * x_batch_stride);
  const T* x_row = x + x_batch_offset + static_cast<int64_t>(row) * K;
  const uint8_t* w_row = valid
      ? (w + static_cast<int64_t>(rhs_idx) * w_batch_stride + col_w_offset)
      : nullptr;
  const ScaleT* scales_row = valid
      ? (scales + static_cast<int64_t>(rhs_idx) * sb_batch_stride +
         col_sb_offset)
      : nullptr;
  const ScaleT* biases_row = (valid && has_bias)
      ? (biases + static_cast<int64_t>(rhs_idx) * sb_batch_stride +
         col_sb_offset)
      : nullptr;

  float acc = 0.0f;

  constexpr int CHUNK_SIZE = 2048;
  __shared__ float shared_x[CHUNK_SIZE];

  for (int chunk_start = 0; chunk_start < K; chunk_start += CHUNK_SIZE) {
    int chunk_end = min(chunk_start + CHUNK_SIZE, K);
    int chunk_len = chunk_end - chunk_start;

    int tid = warp_idx * blockDim.x + lane;
    for (int i = tid; i < chunk_len; i += blockDim.x * blockDim.y) {
      shared_x[i] = static_cast<float>(x_row[chunk_start + i]);
    }
    __syncthreads();

    if (valid) {
      int g_start = chunk_start / GROUP_SIZE;
      int g_end = (chunk_end + GROUP_SIZE - 1) / GROUP_SIZE;

      for (int g = g_start; g < g_end; ++g) {
        int k_start = max(g * GROUP_SIZE, chunk_start);
        int k_end_g = min((g + 1) * GROUP_SIZE, chunk_end);

        float scale =
            load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_row[g]);
        float bias_val = has_bias ? static_cast<float>(biases_row[g]) : 0.0f;

        if constexpr (AFFINE) {
          float qx_acc0 = 0.0f;
          float qx_acc1 = 0.0f;
          float qx_acc2 = 0.0f;
          float qx_acc3 = 0.0f;
          float x_group_sum = 0.0f;
          float qx_acc = 0.0f;

          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;

              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = static_cast<float>(w_packed & 0xFF);
              float w1 = static_cast<float>((w_packed >> 8) & 0xFF);
              float w2 = static_cast<float>((w_packed >> 16) & 0xFF);
              float w3 = static_cast<float>((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);

              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3;
              }
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = static_cast<float>(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else if constexpr (BITS == 4) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            for (; k_start + k_local + 7 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed =
                  *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
              float w0 = static_cast<float>(w_packed & 0xF);
              float w1 = static_cast<float>((w_packed >> 4) & 0xF);
              float w2 = static_cast<float>((w_packed >> 8) & 0xF);
              float w3 = static_cast<float>((w_packed >> 12) & 0xF);
              float w4 = static_cast<float>((w_packed >> 16) & 0xF);
              float w5 = static_cast<float>((w_packed >> 20) & 0xF);
              float w6 = static_cast<float>((w_packed >> 24) & 0xF);
              float w7 = static_cast<float>((w_packed >> 28) & 0xF);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
              }
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else if constexpr (BITS == 6) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            int max_safe_k = ((row_bytes - 7) * 8) / 6;
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              uint64_t w_packed = 0;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              float w0 = static_cast<float>(w_packed & 0x3F);
              float w1 = static_cast<float>((w_packed >> 6) & 0x3F);
              float w2 = static_cast<float>((w_packed >> 12) & 0x3F);
              float w3 = static_cast<float>((w_packed >> 18) & 0x3F);
              float w4 = static_cast<float>((w_packed >> 24) & 0x3F);
              float w5 = static_cast<float>((w_packed >> 30) & 0x3F);
              float w6 = static_cast<float>((w_packed >> 36) & 0x3F);
              float w7 = static_cast<float>((w_packed >> 42) & 0x3F);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
              if (has_bias) {
                x_group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
              }
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(x_val, static_cast<float>(quant_val), qx_acc);
              if (has_bias) {
                x_group_sum += x_val;
              }
            }
          }
          acc += scale * qx_acc;
          if (has_bias) {
            acc += bias_val * x_group_sum;
          }
        } else {
          float qx_acc0 = 0.0f;
          float qx_acc1 = 0.0f;
          float qx_acc2 = 0.0f;
          float qx_acc3 = 0.0f;
          float qx_acc = 0.0f;
          if constexpr (BITS == 8) {
            int k_local = lane * 4;
            int step = THREADS_PER_COL * 4;
            for (; k_start + k_local + 3 < k_end_g; k_local += step) {
              int k = k_start + k_local;

              uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_row[k]);
              float w0 = fp8_e4m3_to_float(w_packed & 0xFF);
              float w1 = fp8_e4m3_to_float((w_packed >> 8) & 0xFF);
              float w2 = fp8_e4m3_to_float((w_packed >> 16) & 0xFF);
              float w3 = fp8_e4m3_to_float((w_packed >> 24) & 0xFF);

              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];

              qx_acc0 = fmaf(x0, w0, qx_acc0);
              qx_acc1 = fmaf(x1, w1, qx_acc1);
              qx_acc2 = fmaf(x2, w2, qx_acc2);
              qx_acc3 = fmaf(x3, w3, qx_acc3);
            }
            qx_acc = qx_acc0 + qx_acc1 + qx_acc2 + qx_acc3;
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              float w_val = fp8_e4m3_to_float(w_row[k]);
              qx_acc = fmaf(x_val, w_val, qx_acc);
            }
          } else if constexpr (BITS == 4) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            for (; k_start + k_local + 7 < k_end_g; k_local += step) {
              int k = k_start + k_local;
              uint32_t w_packed =
                  *reinterpret_cast<const uint32_t*>(&w_row[k / 2]);
              float w0 = dequantize_value<4, false>(w_packed & 0xF, 1.0f, 0.0f);
              float w1 =
                  dequantize_value<4, false>((w_packed >> 4) & 0xF, 1.0f, 0.0f);
              float w2 =
                  dequantize_value<4, false>((w_packed >> 8) & 0xF, 1.0f, 0.0f);
              float w3 = dequantize_value<4, false>(
                  (w_packed >> 12) & 0xF, 1.0f, 0.0f);
              float w4 = dequantize_value<4, false>(
                  (w_packed >> 16) & 0xF, 1.0f, 0.0f);
              float w5 = dequantize_value<4, false>(
                  (w_packed >> 20) & 0xF, 1.0f, 0.0f);
              float w6 = dequantize_value<4, false>(
                  (w_packed >> 24) & 0xF, 1.0f, 0.0f);
              float w7 = dequantize_value<4, false>(
                  (w_packed >> 28) & 0xF, 1.0f, 0.0f);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else if constexpr (BITS == 6) {
            int k_local = lane * 8;
            int step = THREADS_PER_COL * 8;
            int max_safe_k = ((row_bytes - 7) * 8) / 6;
            for (; k_start + k_local + 7 < k_end_g &&
                 k_start + k_local < max_safe_k;
                 k_local += step) {
              int k = k_start + k_local;
              int byte_idx = (k * 6) / 8;
              int bit_offset = (k * 6) % 8;
              uint64_t w_packed = 0;
              memcpy(&w_packed, &w_row[byte_idx], 8);
              w_packed >>= bit_offset;
              float w0 =
                  dequantize_value<6, false>(w_packed & 0x3F, 1.0f, 0.0f);
              float w1 = dequantize_value<6, false>(
                  (w_packed >> 6) & 0x3F, 1.0f, 0.0f);
              float w2 = dequantize_value<6, false>(
                  (w_packed >> 12) & 0x3F, 1.0f, 0.0f);
              float w3 = dequantize_value<6, false>(
                  (w_packed >> 18) & 0x3F, 1.0f, 0.0f);
              float w4 = dequantize_value<6, false>(
                  (w_packed >> 24) & 0x3F, 1.0f, 0.0f);
              float w5 = dequantize_value<6, false>(
                  (w_packed >> 30) & 0x3F, 1.0f, 0.0f);
              float w6 = dequantize_value<6, false>(
                  (w_packed >> 36) & 0x3F, 1.0f, 0.0f);
              float w7 = dequantize_value<6, false>(
                  (w_packed >> 42) & 0x3F, 1.0f, 0.0f);
              float x0 = shared_x[k - chunk_start];
              float x1 = shared_x[k - chunk_start + 1];
              float x2 = shared_x[k - chunk_start + 2];
              float x3 = shared_x[k - chunk_start + 3];
              float x4 = shared_x[k - chunk_start + 4];
              float x5 = shared_x[k - chunk_start + 5];
              float x6 = shared_x[k - chunk_start + 6];
              float x7 = shared_x[k - chunk_start + 7];
              qx_acc = fmaf(x0, w0, qx_acc);
              qx_acc = fmaf(x1, w1, qx_acc);
              qx_acc = fmaf(x2, w2, qx_acc);
              qx_acc = fmaf(x3, w3, qx_acc);
              qx_acc = fmaf(x4, w4, qx_acc);
              qx_acc = fmaf(x5, w5, qx_acc);
              qx_acc = fmaf(x6, w6, qx_acc);
              qx_acc = fmaf(x7, w7, qx_acc);
            }
            for (; k_start + k_local < k_end_g; k_local++) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          } else {
            for (int k_local = lane; k_start + k_local < k_end_g;
                 k_local += THREADS_PER_COL) {
              int k = k_start + k_local;
              float x_val = shared_x[k - chunk_start];
              uint8_t quant_val =
                  unpack_packed_value_fast<BITS>(w_row, k, row_bytes);
              qx_acc = fmaf(
                  x_val,
                  dequantize_value<BITS, false>(quant_val, 1.0f, 0.0f),
                  qx_acc);
            }
          }
          acc += scale * qx_acc;
        }
      }
    }
    __syncthreads();
  }

  acc = subgroup_reduce_sum_qmm<THREADS_PER_COL>(acc);
  if (col_valid && lane == 0) {
    int64_t out_offset = (static_cast<int64_t>(batch) * M + row) * N + col;
    out[out_offset] = expert_valid ? static_cast<T>(acc) : static_cast<T>(0);
  }
}

template <typename T, typename ScaleT, int BITS, int GROUP_SIZE, bool AFFINE>
__global__ void gather_qmv_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    const uint32_t* __restrict__ lhs_indices,
    const uint32_t* __restrict__ rhs_indices,
    const rocm::Shape batch_shape,
    const rocm::Strides lhs_idx_strides,
    const rocm::Strides rhs_idx_strides,
    int batch_ndim,
    T* __restrict__ out,
    int B,
    int M,
    int N,
    int K,
    int E,
    bool has_bias,
    bool implicit_lhs = false,
    int64_t implicit_x_batch_stride = 0) {
  int batch = blockIdx.z;
  int row = blockIdx.x;
  int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (batch >= B || row >= M || col >= N)
    return;
  int64_t lhs_idx_loc = 0;
  int64_t rhs_idx_loc = 0;
  if (batch_ndim == 1) {
    rhs_idx_loc = (int64_t)batch * rhs_idx_strides[0];
    if (!implicit_lhs) {
      lhs_idx_loc = (int64_t)batch * lhs_idx_strides[0];
    }
  } else if (batch_ndim > 1) {
    int64_t elem = (int64_t)batch;
    for (int i = batch_ndim - 1; i >= 0; --i) {
      int64_t coord = elem % batch_shape.data_[i];
      rhs_idx_loc += coord * rhs_idx_strides.data_[i];
      if (!implicit_lhs) {
        lhs_idx_loc += coord * lhs_idx_strides.data_[i];
      }
      elem /= batch_shape.data_[i];
    }
  }
  uint32_t lhs_idx = implicit_lhs ? 0u : lhs_indices[lhs_idx_loc];
  uint32_t rhs_idx = rhs_indices[rhs_idx_loc];
  if (rhs_idx >= static_cast<uint32_t>(E)) {
    out[batch * M * N + row * N + col] = static_cast<T>(0);
    return;
  }

  int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  int row_bytes = (K * BITS + 7) / 8;
  int64_t x_batch_stride = static_cast<int64_t>(M) * K;
  int64_t w_batch_stride = static_cast<int64_t>(N) * row_bytes;
  int64_t sb_batch_stride = static_cast<int64_t>(N) * num_groups;
  int64_t col_w_offset = static_cast<int64_t>(col) * row_bytes;
  int64_t col_sb_offset = static_cast<int64_t>(col) * num_groups;

  int64_t x_batch_offset = implicit_lhs
      ? (static_cast<int64_t>(batch) * implicit_x_batch_stride)
      : (static_cast<int64_t>(lhs_idx) * x_batch_stride);
  const T* x_ptr = x + x_batch_offset + static_cast<int64_t>(row) * K;
  const uint8_t* w_ptr =
      w + static_cast<int64_t>(rhs_idx) * w_batch_stride + col_w_offset;
  const ScaleT* scales_ptr =
      scales + static_cast<int64_t>(rhs_idx) * sb_batch_stride + col_sb_offset;
  const ScaleT* biases_ptr = has_bias
      ? biases + static_cast<int64_t>(rhs_idx) * sb_batch_stride + col_sb_offset
      : nullptr;
  float acc = 0.0f;
  for (int g = 0; g < num_groups; ++g) {
    float scale = load_scale_value<ScaleT, GROUP_SIZE, AFFINE>(scales_ptr[g]);
    float bias = has_bias ? (float)biases_ptr[g] : 0.0f;
    int k_start = g * GROUP_SIZE;
    int k_end = min(k_start + GROUP_SIZE, K);

    if constexpr (BITS == 8) {
      int k = k_start;
      for (; k + 3 < k_end; k += 4) {
        uint32_t w_packed = *reinterpret_cast<const uint32_t*>(&w_ptr[k]);
        float w0 = dequantize_value<8, AFFINE>(w_packed & 0xFF, scale, bias);
        float w1 =
            dequantize_value<8, AFFINE>((w_packed >> 8) & 0xFF, scale, bias);
        float w2 =
            dequantize_value<8, AFFINE>((w_packed >> 16) & 0xFF, scale, bias);
        float w3 =
            dequantize_value<8, AFFINE>((w_packed >> 24) & 0xFF, scale, bias);

        acc += (float)x_ptr[k] * w0;
        acc += (float)x_ptr[k + 1] * w1;
        acc += (float)x_ptr[k + 2] * w2;
        acc += (float)x_ptr[k + 3] * w3;
      }
      for (; k < k_end; ++k) {
        float w_val = dequantize_value<8, AFFINE>(w_ptr[k], scale, bias);
        acc += (float)x_ptr[k] * w_val;
      }
    } else {
      for (int k = k_start; k < k_end; ++k) {
        uint8_t qv = unpack_packed_value_fast<BITS>(w_ptr, k, row_bytes);
        acc +=
            (float)x_ptr[k] * dequantize_value<BITS, AFFINE>(qv, scale, bias);
      }
    }
  }
  out[batch * M * N + row * N + col] = (T)acc;
}
} // namespace rocm

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = rocm::device(s.device);
  auto& enc = d.get_command_encoder(s);
  out.set_data(allocator::malloc(out.nbytes()));
  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array w = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;
  bool has_bias = (mode_ == QuantizationMode::Affine) && (inputs.size() == 6);
  if (has_bias)
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
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
  if (has_bias)
    enc.set_input_array(biases.value());
  enc.set_input_array(lhs_indices);
  enc.set_input_array(rhs_indices);
  enc.set_output_array(out);
  int K = x.shape(-1), M = x.shape(-2), N = out.shape(-1),
      B = out.size() / M / N, E = w.size() / w.shape(-1) / w.shape(-2);

  int64_t x_batch_count = x.size() / (static_cast<int64_t>(M) * K);
  bool use_sorted_rhs_schedule = transpose_ && right_sorted_ && (M == 1) &&
      (B >= 16) && (E > 0) && (B / E >= 4) &&
      (x_batch_count == 1 || x_batch_count == B);
  int64_t implicit_x_batch_stride =
      (x_batch_count == 1) ? 0 : static_cast<int64_t>(M) * K;

  int block_size = 256;
  dim3 grid(M, (N + block_size - 1) / block_size, B);

  int fast_threads_per_col = select_qmv_threads_per_col(K, N, bits_, B);
  int fast_threads_env =
      parse_threads_per_col_env("MLX_ROCM_GATHER_QMV_THREADS_PER_COL");
  if (fast_threads_env <= 0) {
    fast_threads_env =
        parse_threads_per_col_env("MLX_ROCM_QMV_THREADS_PER_COL");
  }
  if (fast_threads_env > 0) {
    fast_threads_per_col = fast_threads_env;
  }

  int fast_cols_per_block = select_qmv_cols_per_block(K, N, bits_);
  if (fast_threads_per_col == 16 && bits_ == 8 && N >= 2048) {
    fast_cols_per_block = std::max(fast_cols_per_block, 64);
  }
  int max_cols_per_block = rocm::kMaxThreadsPerBlock / fast_threads_per_col;
  while (fast_cols_per_block > max_cols_per_block) {
    fast_cols_per_block /= 2;
  }
  while (fast_cols_per_block > 1 && (N % fast_cols_per_block) != 0 &&
         fast_cols_per_block > 8) {
    fast_cols_per_block /= 2;
  }

  dim3 fast_block(fast_threads_per_col, fast_cols_per_block);
  dim3 fast_grid(M, (N + fast_cols_per_block - 1) / fast_cols_per_block, B);

  bool bits_supported_by_fast = (bits_ == 2 || bits_ == 4 || bits_ == 8) ||
      (mode_ == QuantizationMode::Affine && (bits_ == 5 || bits_ == 6));
  bool use_fast_gather_qmv = transpose_ && bits_supported_by_fast;
  use_fast_gather_qmv = parse_warp_kernel_env(
      "MLX_ROCM_GATHER_QMV_USE_WARP", use_fast_gather_qmv);
  const void *x_ptr = gpu_ptr<void>(x), *w_ptr = gpu_ptr<void>(w),
             *scales_ptr = gpu_ptr<void>(scales),
             *biases_ptr = has_bias ? gpu_ptr<void>(*biases) : nullptr;
  const uint32_t *li_ptr = gpu_ptr<const uint32_t>(lhs_indices),
                 *ri_ptr = gpu_ptr<const uint32_t>(rhs_indices);
  void* out_ptr = gpu_ptr<void>(out);
  enc.launch_kernel([&](hipStream_t stream) {
    if (use_fast_gather_qmv && mode_ == QuantizationMode::Affine &&
        x.dtype() == bfloat16 && group_size_ == 64 &&
        (bits_ == 6 || bits_ == 8)) {
      auto launch_fast_kernel = [&](auto bits_tag) {
        constexpr int BITS = decltype(bits_tag)::value;
        if (fast_threads_per_col == 16) {
          hipLaunchKernelGGL(
              (rocm::gather_qmv_warp_shared_kernel<
                  hip_bfloat16,
                  hip_bfloat16,
                  BITS,
                  64,
                  true,
                  16>),
              fast_grid,
              fast_block,
              0,
              stream,
              (const hip_bfloat16*)x_ptr,
              (const uint8_t*)w_ptr,
              (const hip_bfloat16*)scales_ptr,
              (const hip_bfloat16*)biases_ptr,
              li_ptr,
              ri_ptr,
              batch_shape_param,
              lhs_idx_strides_param,
              rhs_idx_strides_param,
              batch_ndim,
              (hip_bfloat16*)out_ptr,
              B,
              M,
              N,
              K,
              E,
              has_bias,
              use_sorted_rhs_schedule,
              implicit_x_batch_stride);
        } else {
          hipLaunchKernelGGL(
              (rocm::gather_qmv_warp_shared_kernel<
                  hip_bfloat16,
                  hip_bfloat16,
                  BITS,
                  64,
                  true,
                  WARP_SIZE>),
              fast_grid,
              fast_block,
              0,
              stream,
              (const hip_bfloat16*)x_ptr,
              (const uint8_t*)w_ptr,
              (const hip_bfloat16*)scales_ptr,
              (const hip_bfloat16*)biases_ptr,
              li_ptr,
              ri_ptr,
              batch_shape_param,
              lhs_idx_strides_param,
              rhs_idx_strides_param,
              batch_ndim,
              (hip_bfloat16*)out_ptr,
              B,
              M,
              N,
              K,
              E,
              has_bias,
              use_sorted_rhs_schedule,
              implicit_x_batch_stride);
        }
      };

      if (bits_ == 6) {
        launch_fast_kernel(std::integral_constant<int, 6>{});
      } else {
        launch_fast_kernel(std::integral_constant<int, 8>{});
      }
      return;
    }

    if (x.dtype() == float32) {
      if (bits_ == 8 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 8, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 8, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 8, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 5, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 5, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 5, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 6, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 6, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 6, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 4, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 4, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 4, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 2, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 2, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<float, float, 2, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const float*)x_ptr,
            (const uint8_t*)w_ptr,
            (const float*)scales_ptr,
            (const float*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (float*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else {
        throw std::runtime_error(
            "Unsupported dtype/bits/group_size combination for float32: bits=" +
            std::to_string(bits_) + " gs=" + std::to_string(group_size_));
      }
    } else if (x.dtype() == float16) {
      if (bits_ == 8 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 8, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 8, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 8, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 5, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 5, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 5, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 6, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 6, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 6, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 4, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 4, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 4, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 2, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 2, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<__half, __half, 2, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const __half*)x_ptr,
            (const uint8_t*)w_ptr,
            (const __half*)scales_ptr,
            (const __half*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (__half*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else {
        throw std::runtime_error(
            "Unsupported dtype/bits/group_size combination for float16: bits=" +
            std::to_string(bits_) + " gs=" + std::to_string(group_size_));
      }
    } else if (x.dtype() == bfloat16) {
      if (bits_ == 8 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 8, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 8, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 8 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 8, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 5, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 5, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 5 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 5, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 6, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 6, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 6 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 6, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 4, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 4, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 4 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 4, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 32) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 2, 32, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 64) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 2, 64, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else if (bits_ == 2 && group_size_ == 128) {
        hipLaunchKernelGGL(
            (rocm::gather_qmv_kernel<hip_bfloat16, hip_bfloat16, 2, 128, true>),
            grid,
            dim3(block_size),
            0,
            stream,
            (const hip_bfloat16*)x_ptr,
            (const uint8_t*)w_ptr,
            (const hip_bfloat16*)scales_ptr,
            (const hip_bfloat16*)biases_ptr,
            li_ptr,
            ri_ptr,
            batch_shape_param,
            lhs_idx_strides_param,
            rhs_idx_strides_param,
            batch_ndim,
            (hip_bfloat16*)out_ptr,
            B,
            M,
            N,
            K,
            E,
            has_bias);
      } else {
        throw std::runtime_error(
            "Unsupported dtype/bits/group_size combination for bfloat16: bits=" +
            std::to_string(bits_) + " gs=" + std::to_string(group_size_));
      }
    }
  });
}

} // namespace mlx::core
