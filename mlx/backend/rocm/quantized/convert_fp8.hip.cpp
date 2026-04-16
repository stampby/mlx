#include "mlx/backend/rocm/rocm_utils.h"
// Copyright © 2025 Apple Inc.
#include "mlx/backend/rocm/unary/unary.hip.h"
#include "mlx/fast_primitives.h"

namespace mlx::core {
void fast::ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  
  auto& in = inputs[0];
  auto& out = outputs[0];
  auto& s = out.primitive().stream();
  if (to_fp8_) {
    unary_op_gpu<cu::ToFP8>(inputs, out, name(), s);
  } else {
    unary_op_gpu<cu::FromFP8>(inputs, out, name(), s);
  }
}
} // namespace mlx::core
