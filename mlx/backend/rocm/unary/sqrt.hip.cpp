// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/unary/unary.hip.h"

namespace mlx::core {
void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  
  auto& s = out.primitive().stream();
  if (recip_) {
    unary_op_gpu<cu::Rsqrt>(inputs, out, "Rsqrt", s);
  } else {
    unary_op_gpu<cu::Sqrt>(inputs, out, "Sqrt", s);
  }
}
} // namespace mlx::core
