// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/gemms/gemv.h"
#include "mlx/backend/rocm/gemms/naive_gemm.h"
#include "mlx/primitives.h"
#include "mlx/types/half_types.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstring>
#include <numeric>

namespace mlx::core {

namespace {

std::tuple<bool, int64_t, array>
check_transpose(rocm::CommandEncoder& enc, const Stream& s, const array& arr) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  if (sty == 1 && stx == arr.shape(-1)) {
    return std::make_tuple(false, stx, arr);
  } else if (stx == 1 && sty == arr.shape(-2)) {
    return std::make_tuple(true, sty, arr);
  } else {
    array arr_copy = contiguous_copy_gpu(arr, s);
    enc.add_temporary(arr_copy);
    return std::make_tuple(false, arr.shape(-1), arr_copy);
  }
}

std::tuple<bool, int64_t, array> ensure_batch_contiguous(
    const array& x,
    rocm::CommandEncoder& encoder,
    Stream s) {
  if (x.flags().row_contiguous) {
    return std::make_tuple(false, x.strides(-2), x);
  }

  bool rc = true;
  for (int i = 0; i < x.ndim() - 3; i++) {
    rc &= (x.strides(i + 1) * x.shape(i)) == x.strides(i);
  }
  if (rc) {
    return check_transpose(encoder, s, x);
  }

  array x_copy = contiguous_copy_gpu(x, s);
  encoder.add_temporary(x_copy);
  return std::make_tuple(false, x_copy.strides(-2), x_copy);
}

void gemm_rocblas(
    rocm::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    array& out,
    const array& a,
    const array& b,
    float alpha = 1.0f,
    float beta = 0.0f) {
  auto& device = encoder.device();

  // Use naive_gemm for all types to avoid rocBLAS Tensile initialization
  // being affected by pending GPU errors from other kernels.
  // TODO: Re-enable rocBLAS once gather_qmm memory corruption is resolved.
  // The naive_gemm (tiled shared-memory) is correct for all types and archs.
  {
    naive_gemm(
        encoder, a, b, out, M, N, K,
        a_transposed, a_transposed ? M : K,
        b_transposed, b_transposed ? K : N,
        alpha, beta);
    return;
  }

  rocblas_handle handle = device.get_rocblas_handle();

  rocblas_operation trans_a =
      b_transposed ? rocblas_operation_none : rocblas_operation_transpose;
  rocblas_operation trans_b =
      a_transposed ? rocblas_operation_none : rocblas_operation_transpose;

  // Try rocBLAS first; if it fails (e.g., missing Tensile kernel for this
  // GPU arch + GEMM config), fall back to naive_gemm.
  bool rocblas_ok = true;

  encoder.launch_kernel([&](hipStream_t stream) {
    rocblas_set_stream(handle, stream);
    rocblas_status status = rocblas_status_not_implemented;

    switch (a.dtype()) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        status = rocblas_sgemm(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_f,
            gpu_ptr<float>(b),
            b_transposed ? K : N,
            gpu_ptr<float>(a),
            a_transposed ? M : K,
            &beta_f,
            gpu_ptr<float>(out),
            N);
        break;
      }
      case float64: {
        double alpha_d = static_cast<double>(alpha);
        double beta_d = static_cast<double>(beta);
        status = rocblas_dgemm(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_d,
            gpu_ptr<double>(b),
            b_transposed ? K : N,
            gpu_ptr<double>(a),
            a_transposed ? M : K,
            &beta_d,
            gpu_ptr<double>(out),
            N);
        break;
      }
      case float16: {
        rocblas_half alpha_h, beta_h;
        float16_t alpha_f16 = static_cast<float16_t>(alpha);
        float16_t beta_f16 = static_cast<float16_t>(beta);
        std::memcpy(&alpha_h, &alpha_f16, sizeof(rocblas_half));
        std::memcpy(&beta_h, &beta_f16, sizeof(rocblas_half));
        status = rocblas_hgemm(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(gpu_ptr<void>(b)),
            b_transposed ? K : N,
            reinterpret_cast<const rocblas_half*>(gpu_ptr<void>(a)),
            a_transposed ? M : K,
            &beta_h,
            reinterpret_cast<rocblas_half*>(gpu_ptr<void>(out)),
            N);
        break;
      }
      case bfloat16: {
        float alpha_f = alpha;
        float beta_f = beta;
        status = rocblas_gemm_ex(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_f,
            gpu_ptr<void>(b),
            rocblas_datatype_bf16_r,
            b_transposed ? K : N,
            gpu_ptr<void>(a),
            rocblas_datatype_bf16_r,
            a_transposed ? M : K,
            &beta_f,
            gpu_ptr<void>(out),
            rocblas_datatype_bf16_r,
            N,
            gpu_ptr<void>(out),
            rocblas_datatype_bf16_r,
            N,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard,
            0,
            0);
        break;
      }
      default:
        throw std::runtime_error("Unsupported dtype for matmul on ROCm");
    }

    if (status != rocblas_status_success) {
      rocblas_ok = false;
    }
  });

  if (!rocblas_ok) {
    // Clear any GPU error state from the failed rocBLAS call
    (void)hipGetLastError();
    // Fall back to naive GEMM
    naive_gemm(
        encoder,
        a,
        b,
        out,
        M,
        N,
        K,
        a_transposed,
        a_transposed ? M : K,
        b_transposed,
        b_transposed ? K : N,
        alpha,
        beta);
  }
}

void gemm_strided_batched_rocblas(
    rocm::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t stride_a,
    bool b_transposed,
    int64_t ldb,
    int64_t stride_b,
    int64_t stride_c,
    int batch_count,
    array& out,
    const array& a,
    const array& b,
    float alpha = 1.0f,
    float beta = 0.0f) {
  auto& device = encoder.device();

  // Use naive_gemm for all types (see single GEMM comment above).
  {
    naive_gemm_batched(
        encoder, a, b, out, M, N, K,
        a_transposed, a_transposed ? M : K, stride_a,
        b_transposed, b_transposed ? K : N, stride_b,
        stride_c, batch_count, alpha, beta);
    return;
  }

  rocblas_handle handle = device.get_rocblas_handle();

  rocblas_operation trans_a =
      b_transposed ? rocblas_operation_none : rocblas_operation_transpose;
  rocblas_operation trans_b =
      a_transposed ? rocblas_operation_none : rocblas_operation_transpose;

  bool rocblas_ok = true;

  encoder.launch_kernel([&](hipStream_t stream) {
    rocblas_set_stream(handle, stream);
    rocblas_status status = rocblas_status_not_implemented;

    switch (a.dtype()) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        status = rocblas_sgemm_strided_batched(
            handle, trans_a, trans_b, N, M, K,
            &alpha_f, gpu_ptr<float>(b), b_transposed ? K : N, stride_b,
            gpu_ptr<float>(a), a_transposed ? M : K, stride_a,
            &beta_f, gpu_ptr<float>(out), N, stride_c, batch_count);
        break;
      }
      case float64: {
        double alpha_d = static_cast<double>(alpha);
        double beta_d = static_cast<double>(beta);
        status = rocblas_dgemm_strided_batched(
            handle, trans_a, trans_b, N, M, K,
            &alpha_d, gpu_ptr<double>(b), b_transposed ? K : N, stride_b,
            gpu_ptr<double>(a), a_transposed ? M : K, stride_a,
            &beta_d, gpu_ptr<double>(out), N, stride_c, batch_count);
        break;
      }
      case float16: {
        rocblas_half alpha_h, beta_h;
        float16_t alpha_f16 = static_cast<float16_t>(alpha);
        float16_t beta_f16 = static_cast<float16_t>(beta);
        std::memcpy(&alpha_h, &alpha_f16, sizeof(rocblas_half));
        std::memcpy(&beta_h, &beta_f16, sizeof(rocblas_half));
        status = rocblas_hgemm_strided_batched(
            handle, trans_a, trans_b, N, M, K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(gpu_ptr<void>(b)),
            b_transposed ? K : N, stride_b,
            reinterpret_cast<const rocblas_half*>(gpu_ptr<void>(a)),
            a_transposed ? M : K, stride_a,
            &beta_h,
            reinterpret_cast<rocblas_half*>(gpu_ptr<void>(out)),
            N, stride_c, batch_count);
        break;
      }
      case bfloat16: {
        float alpha_f = alpha;
        float beta_f = beta;
        status = rocblas_gemm_strided_batched_ex(
            handle, trans_a, trans_b, N, M, K,
            &alpha_f,
            gpu_ptr<void>(b), rocblas_datatype_bf16_r,
            b_transposed ? K : N, stride_b,
            gpu_ptr<void>(a), rocblas_datatype_bf16_r,
            a_transposed ? M : K, stride_a,
            &beta_f,
            gpu_ptr<void>(out), rocblas_datatype_bf16_r, N, stride_c,
            gpu_ptr<void>(out), rocblas_datatype_bf16_r, N, stride_c,
            batch_count,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        break;
      }
      default:
        throw std::runtime_error(
            "Unsupported dtype for batched matmul on ROCm");
    }

    if (status != rocblas_status_success) {
      rocblas_ok = false;
    }
  });

  if (!rocblas_ok) {
    (void)hipGetLastError();
    naive_gemm_batched(
        encoder, a, b, out, M, N, K,
        a_transposed, a_transposed ? M : K, stride_a,
        b_transposed, b_transposed ? K : N, stride_b,
        stride_c, batch_count, alpha, beta);
  }
}

void gemm_and_bias(
    rocm::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    array& out,
    const array& a,
    const array& b,
    float alpha = 1.0f,
    float beta = 0.0f) {
  // Check and collapse batch dimensions
  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a, b);

  auto batch_count = out.size() / (M * N);

  // Collapse batches into M if needed
  if (batch_count > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && a_batch_strides.back() == M * K &&
      b_batch_strides.back() == 0) {
    M *= batch_shape.back();
    batch_count = 1;

    a_batch_strides = {0};
    b_batch_strides = {0};
    batch_shape = {1};
  }

  // Use GEMV when possible
  if (rocm::can_use_gemv(M, N, K, a_transposed, b_transposed)) {
    rocm::gemv(
        a,
        b,
        out,
        M,
        N,
        K,
        batch_count,
        batch_shape,
        a_batch_strides,
        b_batch_strides,
        encoder);
    return;
  }

  // Check if rocBLAS is available
  bool use_rocblas = encoder.device().is_rocblas_available();

  if (batch_count == 1) {
    // Simple single GEMM
    if (use_rocblas) {
      gemm_rocblas(
          encoder,
          M,
          N,
          K,
          a_transposed,
          lda,
          b_transposed,
          ldb,
          out,
          a,
          b,
          alpha,
          beta);
    } else {
      // Use naive GEMM fallback
      rocm::naive_gemm(
          encoder,
          a,
          b,
          out,
          M,
          N,
          K,
          a_transposed,
          lda,
          b_transposed,
          ldb,
          alpha,
          beta);
    }
  } else if (
      batch_shape.size() == 1 && a_batch_strides.back() > 0 &&
      b_batch_strides.back() > 0) {
    // Use strided batched GEMM for uniform batches
    if (use_rocblas) {
      gemm_strided_batched_rocblas(
          encoder,
          M,
          N,
          K,
          a_transposed,
          lda,
          a_batch_strides.back(),
          b_transposed,
          ldb,
          b_batch_strides.back(),
          M * N,
          batch_count,
          out,
          a,
          b,
          alpha,
          beta);
    } else {
      // Use naive batched GEMM fallback
      rocm::naive_gemm_batched(
          encoder,
          a,
          b,
          out,
          M,
          N,
          K,
          a_transposed,
          lda,
          a_batch_strides.back(),
          b_transposed,
          ldb,
          b_batch_strides.back(),
          M * N,
          batch_count,
          alpha,
          beta);
    }
  } else {
    // Fallback: loop over batches for non-uniform strides
    if (use_rocblas) {
      for (int64_t batch = 0; batch < batch_count; ++batch) {
        int64_t a_offset = 0, b_offset = 0;
        int64_t batch_idx = batch;
        for (int i = batch_shape.size() - 1; i >= 0; --i) {
          int64_t idx = batch_idx % batch_shape[i];
          batch_idx /= batch_shape[i];
          a_offset += idx * a_batch_strides[i];
          b_offset += idx * b_batch_strides[i];
        }

        encoder.launch_kernel(
            [&, a_offset, b_offset, batch](hipStream_t stream) {
              auto& device = encoder.device();
              rocblas_handle handle = device.get_rocblas_handle();
              rocblas_set_stream(handle, stream);

              rocblas_operation trans_a = b_transposed
                  ? rocblas_operation_none
                  : rocblas_operation_transpose;
              rocblas_operation trans_b = a_transposed
                  ? rocblas_operation_none
                  : rocblas_operation_transpose;

              float alpha_f = alpha, beta_f = beta;

              if (a.dtype() == float32) {
                rocblas_sgemm(
                    handle,
                    trans_a,
                    trans_b,
                    N,
                    M,
                    K,
                    &alpha_f,
                    b.data<float>() + b_offset,
                    b_transposed ? K : N,
                    a.data<float>() + a_offset,
                    a_transposed ? M : K,
                    &beta_f,
                    out.data<float>() + batch * M * N,
                    N);
              } else if (a.dtype() == float64) {
                double alpha_d = static_cast<double>(alpha);
                double beta_d = static_cast<double>(beta);
                rocblas_dgemm(
                    handle,
                    trans_a,
                    trans_b,
                    N,
                    M,
                    K,
                    &alpha_d,
                    b.data<double>() + b_offset,
                    b_transposed ? K : N,
                    a.data<double>() + a_offset,
                    a_transposed ? M : K,
                    &beta_d,
                    out.data<double>() + batch * M * N,
                    N);
              }
            });
      }
    } else {
      // Use naive GEMM for each batch when rocBLAS is not available
      // This is less efficient but provides correctness
      for (int64_t batch = 0; batch < batch_count; ++batch) {
        int64_t a_offset = 0, b_offset = 0;
        int64_t batch_idx = batch;
        for (int i = batch_shape.size() - 1; i >= 0; --i) {
          int64_t idx = batch_idx % batch_shape[i];
          batch_idx /= batch_shape[i];
          a_offset += idx * a_batch_strides[i];
          b_offset += idx * b_batch_strides[i];
        }

        // Use naive GEMM with explicit offsets
        rocm::naive_gemm_with_offset(
            encoder,
            a,
            b,
            out,
            M,
            N,
            K,
            a_transposed,
            lda,
            a_offset,
            b_transposed,
            ldb,
            b_offset,
            batch * M * N,
            alpha,
            beta);
      }
    }
  }
}

} // namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 2);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];

  // Return 0s if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero(0, a_pre.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  gemm_and_bias(
      encoder, M, N, K, a_transposed, lda, b_transposed, ldb, out, a, b);
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 3);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto c = inputs[2];

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  // Copy C into out first, then do GEMM with beta
  copy_gpu(c, out, CopyType::General, s);

  // Check if rocBLAS is available
  if (encoder.device().is_rocblas_available()) {
    // Do GEMM with alpha and beta
    gemm_rocblas(
        encoder,
        M,
        N,
        K,
        a_transposed,
        lda,
        b_transposed,
        ldb,
        out,
        a,
        b,
        alpha_,
        beta_);
  } else {
    // Use naive GEMM fallback
    rocm::naive_gemm(
        encoder,
        a,
        b,
        out,
        M,
        N,
        K,
        a_transposed,
        lda,
        b_transposed,
        ldb,
        alpha_,
        beta_);
  }
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 4);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& lhs_indices = inputs[2];
  auto& rhs_indices = inputs[3];

  // Return 0s if either input is empty.
  if (a.size() == 0 || b.size() == 0) {
    array zero(0, a.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  // Extract shapes from inputs.
  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);

  auto [transposed_a, lda, a_] = check_transpose(encoder, s, a);
  auto [transposed_b, ldb, b_] = check_transpose(encoder, s, b);

  auto use_gemv = rocm::can_use_gemv(M, N, K, transposed_a, transposed_b);

  if (M == 1 && use_gemv) {
    rocm::gather_mv(b_, a_, rhs_indices, lhs_indices, out, N, K, encoder);
    return;
  }

  if (N == 1 && use_gemv) {
    rocm::gather_mv(a_, b_, lhs_indices, rhs_indices, out, M, K, encoder);
    return;
  }

  // Check if rocBLAS is available
  bool use_rocblas = encoder.device().is_rocblas_available();

  // Fallback: loop over batches with individual GEMMs
  int batch_size = lhs_indices.size();

  // Get indices on CPU (this is not optimal but provides correctness)
  std::vector<uint32_t> lhs_idx(batch_size);
  std::vector<uint32_t> rhs_idx(batch_size);

  // Synchronize to get indices
  hipDeviceSynchronize();

  if (lhs_indices.dtype() == uint32) {
    std::memcpy(
        lhs_idx.data(),
        lhs_indices.data<uint32_t>(),
        batch_size * sizeof(uint32_t));
  }
  if (rhs_indices.dtype() == uint32) {
    std::memcpy(
        rhs_idx.data(),
        rhs_indices.data<uint32_t>(),
        batch_size * sizeof(uint32_t));
  }

  if (use_rocblas) {
    for (int i = 0; i < batch_size; ++i) {
      int64_t a_offset = lhs_idx[i] * M * K;
      int64_t b_offset = rhs_idx[i] * K * N;
      int64_t out_offset = i * M * N;

      encoder.launch_kernel([&, a_offset, b_offset, out_offset](
                                hipStream_t stream) {
        auto& device = encoder.device();
        rocblas_handle handle = device.get_rocblas_handle();
        rocblas_set_stream(handle, stream);

        rocblas_operation trans_a =
            transposed_b ? rocblas_operation_none : rocblas_operation_transpose;
        rocblas_operation trans_b =
            transposed_a ? rocblas_operation_none : rocblas_operation_transpose;

        float alpha = 1.0f, beta = 0.0f;

        if (a.dtype() == float32) {
          rocblas_sgemm(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha,
              b_.data<float>() + b_offset,
              transposed_b ? K : N,
              a_.data<float>() + a_offset,
              transposed_a ? M : K,
              &beta,
              out.data<float>() + out_offset,
              N);
        }
      });
    }
  } else {
    // Use naive GEMM for each batch
    for (int i = 0; i < batch_size; ++i) {
      int64_t a_offset = lhs_idx[i] * M * K;
      int64_t b_offset = rhs_idx[i] * K * N;
      int64_t out_offset = i * M * N;

      // Use naive GEMM with explicit offsets
      rocm::naive_gemm_with_offset(
          encoder,
          a_,
          b_,
          out,
          M,
          N,
          K,
          transposed_a,
          lda,
          a_offset,
          transposed_b,
          ldb,
          b_offset,
          out_offset,
          1.0f,
          0.0f);
    }
  }
}

} // namespace mlx::core
