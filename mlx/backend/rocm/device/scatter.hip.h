// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device/indexing.hip.h"
#include "mlx/backend/rocm/device/scatter_ops.hip.h"
#include "mlx/backend/rocm/device/utils.hip.h"

// cooperative_groups not available on HIP — use HIP equivalents
#include <hip/hip_cooperative_groups.h>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

template <
    typename T,
    typename IdxT,
    typename Op,
    int NIDX,
    int IDX_NDIM,
    typename LocT>
__global__ void scatter(
    const T* upd,
    T* out,
    LocT size,
    const  Shape upd_shape,
    const  Strides upd_strides,
    int32_t upd_ndim,
    LocT upd_post_idx_size,
    const  Shape out_shape,
    const  Strides out_strides,
    int32_t out_ndim,
    const  std::array<int32_t, NIDX> axes,
    const  std::array<IdxT*, NIDX> indices,
    const  std::array<int32_t, NIDX * IDX_NDIM>
        indices_shape,
    const  std::array<int64_t, NIDX * IDX_NDIM>
        indices_strides) {
  LocT upd_idx = cg::this_grid().thread_rank();
  if (upd_idx >= size) {
    return;
  }

  LocT out_elem = upd_idx % upd_post_idx_size;
  LocT idx_elem = upd_idx / upd_post_idx_size;

  LocT out_idx = elem_to_loc(
      out_elem, upd_shape.data() + IDX_NDIM, out_strides.data(), out_ndim);

#pragma unroll
  for (int i = 0; i < NIDX; ++i) {
    LocT idx_loc = elem_to_loc_nd<IDX_NDIM>(
        idx_elem,
        indices_shape.data() + i * IDX_NDIM,
        indices_strides.data() + i * IDX_NDIM);
    int32_t axis = axes[i];
    LocT idx_val = absolute_index(indices[i][idx_loc], out_shape[axis]);
    out_idx += idx_val * out_strides[axis];
  }

  LocT upd_loc = elem_to_loc(
      out_elem + idx_elem * upd_post_idx_size,
      upd_shape.data(),
      upd_strides.data(),
      upd_ndim);

  Op{}(out + out_idx, upd[upd_loc]);
}

template <typename T, bool SrcContiguous, bool DstContiguous, typename IdxT>
__global__ void masked_scatter(
    const T* dst,
    const bool* mask,
    const int32_t* scatter_offsets,
    const T* src,
    T* out,
    IdxT size,
    IdxT src_batch_size,
    IdxT mask_batch_size,
    const  Shape dst_shape,
    const  Strides dst_strides,
    int32_t dst_ndim,
    const  Shape src_shape,
    const  Strides src_strides,
    int32_t src_ndim) {
  IdxT index = cg::this_grid().thread_rank();
  if (index >= size) {
    return;
  }

  T dst_val;
  if constexpr (DstContiguous) {
    dst_val = dst[index];
  } else {
    IdxT dst_loc =
        elem_to_loc(index, dst_shape.data(), dst_strides.data(), dst_ndim);
    dst_val = dst[dst_loc];
  }

  if (mask[index]) {
    IdxT src_index = static_cast<IdxT>(scatter_offsets[index]);
    if (src_index < src_batch_size) {
      IdxT batch_idx = index / mask_batch_size;
      if constexpr (SrcContiguous) {
        out[index] = src[batch_idx * src_batch_size + src_index];
      } else {
        IdxT src_elem = batch_idx * src_batch_size + src_index;
        IdxT src_loc = elem_to_loc(
            src_elem, src_shape.data(), src_strides.data(), src_ndim);
        out[index] = src[src_loc];
      }
      return;
    }
  }

  out[index] = dst_val;
}

template <typename T, typename IdxT, int N_READS>
__global__ void masked_scatter_vec_contiguous(
    const T* dst,
    const bool* mask,
    const int32_t* scatter_offsets,
    const T* src,
    T* out,
    IdxT size,
    IdxT src_batch_size,
    IdxT mask_batch_size) {
  IdxT vec_index = cg::this_grid().thread_rank();
  IdxT base = vec_index * N_READS;
  if (base >= size) {
    return;
  }

  auto out_vec = load_vector<N_READS>(dst, vec_index, size, static_cast<T>(0));
  auto mask_vec = load_vector<N_READS>(mask, vec_index, size, false);
  auto offset_vec = load_vector<N_READS>(scatter_offsets, vec_index, size, 0);

#pragma unroll
  for (int i = 0; i < N_READS; ++i) {
    IdxT index = base + i;
    if (index >= size) {
      break;
    }
    if (mask_vec[i]) {
      IdxT src_index = static_cast<IdxT>(offset_vec[i]);
      if (src_index < src_batch_size) {
        IdxT batch_idx = index / mask_batch_size;
        out_vec[i] = src[batch_idx * src_batch_size + src_index];
      }
    }
  }

  store_vector<N_READS>(out, vec_index, out_vec, size);
}

} // namespace mlx::core::cu
