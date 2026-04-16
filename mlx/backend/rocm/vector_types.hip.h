// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace mlx::core::cu {

template <typename T>
struct Vector2;

template <>
struct Vector2<double> {
  using type = double2;
};

template <>
struct Vector2<float> {
  using type = float2;
};

template <>
struct Vector2<__half> {
  using type = __half2;
};

template <>
struct Vector2<__hip_bfloat16> {
  using type = __hip_bfloat162;
};

template <typename T>
using Vector2_t = typename Vector2<T>::type;

template <typename T>
struct Vector4 {
  T x, y, z, w;
};

template <typename T>
using Vector4_t = Vector4<T>;

using bf16x4 = Vector4_t<__hip_bfloat16>;
using fp16x4 = Vector4_t<__half>;
using fp32x4 = Vector4_t<float>;

} // namespace mlx::core::cu
