#include "mlx/backend/rocm/rocm_utils.h"
// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/reduce/reduce.hip.h"
#include "mlx/backend/gpu/copy.h"

// NVTX not available on ROCm — profiling markers disabled

#include <cassert>

namespace mlx::core {

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here.
  assert(!axes_.empty());
  assert(out.size() != in.size());

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  if (in.size() == 0) {
    init_reduce(encoder, in, out, reduce_type_);
    return;
  }

  // Reduce.
  ReductionPlan plan = get_reduction_plan(in, axes_);

  // If it is a general reduce then copy the input to a contiguous array and
  // recompute the plan.
  //
  // TODO: Instead of copying we can use elem-to-loc to deal with broadcasting
  //       like we do in Metal. When it comes to broadcasted reduction axes
  //       some can be ignored eg for min/max.
  bool broadcasted = false;
  for (int i = 0, j = 0; i < in.ndim() && !broadcasted; i++) {
    if (j < axes_.size() && axes_[j] == i) {
      j++;
    } else {
      broadcasted = in.strides(i) == 0;
    }
  }
  if (plan.type == GeneralReduce || broadcasted || !in.flags().contiguous) {
    array in_copy = contiguous_copy_gpu(in, s);
    encoder.add_temporary(in_copy);
    in = in_copy;
    plan = get_reduction_plan(in, axes_);
  }

  if (plan.type == ContiguousAllReduce) {
    all_reduce(encoder, in, out, reduce_type_);
    return;
  }

  if (plan.type == ContiguousReduce || plan.type == GeneralContiguousReduce) {
    row_reduce(encoder, in, out, reduce_type_, axes_, plan);
    return;
  }

  if (plan.type == ContiguousStridedReduce ||
      plan.type == GeneralStridedReduce) {
    col_reduce(encoder, in, out, reduce_type_, axes_, plan);
    return;
  }

  throw std::runtime_error("No plan reached in reduce.");
}

} // namespace mlx::core
