// Copyright © 2026 Apple Inc.

#include "mlx/backend/hip/quantized/qqmm_impl.h"

namespace mlx::core {
void qqmm_impl(
    cu::CommandEncoder&,
    int,
    int,
    int,
    bool,
    int64_t,
    bool,
    int64_t,
    array&,
    const array&,
    const array&,
    const array&,
    const array&,
    QuantizationMode,
    const GemmScalars&) {
  throw std::runtime_error(
      "[QQMatmul::eval_gpu] QQMM is only supported with HIP 12.8 or higher.");
}
} // namespace mlx::core
