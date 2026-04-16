// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/fp16_math.hip.h"
#include "mlx/backend/rocm/device/utils.hip.h"

// cuda_fp8.h not available on HIP — fp8 ops disabled for now
#include <hip/hip_math_constants.h>
#include <cmath>

namespace mlx::core::cu {

struct Abs {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_unsigned_v<T>) {
      return x;
    } else {
      return std::abs(x);
    }
  }
};

struct ArcCos {
  template <typename T>
  __device__ T operator()(T x) {
    return std::acos(x);
  }
};

struct ArcCosh {
  template <typename T>
  __device__ T operator()(T x) {
    return std::acosh(x);
  }
};

struct ArcSin {
  template <typename T>
  __device__ T operator()(T x) {
    return std::asin(x);
  }
};

struct ArcSinh {
  template <typename T>
  __device__ T operator()(T x) {
    return std::asinh(x);
  }
};

struct ArcTan {
  template <typename T>
  __device__ T operator()(T x) {
    return std::atan(x);
  }
};

struct ArcTanh {
  template <typename T>
  __device__ T operator()(T x) {
    return std::atanh(x);
  }
};

struct BitwiseInvert {
  template <typename T>
  __device__ T operator()(T x) {
    return ~x;
  }
};

struct Ceil {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_integral_v<T>) {
      return x;
    } else if constexpr (is_complex_v<T>) {
      return T{std::ceil(x.real()), std::ceil(x.imag())};
    } else {
      return std::ceil(x);
    }
  }
};

struct Conjugate {
  template <typename T>
  __device__ complex_t<T> operator()(complex_t<T> x) {
    return std::conj(x);
  }
};

struct Cos {
  template <typename T>
  __device__ T operator()(T x) {
    return std::cos(x);
  }
};

struct Cosh {
  template <typename T>
  __device__ T operator()(T x) {
    return std::cosh(x);
  }
};

struct Erf {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, __half>) {
      return erf(__half2float(x));
    } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
      return erf(__bfloat162float(x));
    } else {
      return erf(x);
    }
  }
};

struct ErfInv {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, __half>) {
      return erfinv(__half2float(x));
    } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
      return erfinv(__bfloat162float(x));
    } else {
      return erfinv(x);
    }
  }
};

struct Exp {
  template <typename T>
  __device__ T operator()(T x) {
    return std::exp(x);
  }
};

struct Expm1 {
  template <typename T>
  __device__ T operator()(T x) {
    return std::expm1(x);
  }
};

struct Floor {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_integral_v<T>) {
      return x;
    } else if constexpr (is_complex_v<T>) {
      return T{std::floor(x.real()), std::floor(x.imag())};
    } else {
      return std::floor(x);
    }
  }
};

struct Imag {
  template <typename T>
  __device__ auto operator()(complex_t<T> x) {
    return x.imag();
  }
};

struct Log {
  template <typename T>
  __device__ T operator()(T x) {
    return std::log(x);
  }
};

struct Log2 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      auto y = Log{}(x);
      return {y.real() / 0.6931471805599453f, y.imag() / 0.6931471805599453f};
    } else {
      return std::log2(x);
    }
  }
};

struct Log10 {
  template <typename T>
  __device__ T operator()(T x) {
    return std::log10(x);
  }
};

struct Log1p {
  template <typename T>
  __device__ T operator()(T z) {
    if constexpr (is_complex_v<T>) {
      float x = z.real();
      float y = z.imag();
      float zabs = Abs{}(z).real();
      float theta = atan2f(y, x + 1);
      if (zabs < 0.5f) {
        float r = x * (2 + x) + y * y;
        if (r == 0) { // handle underflow
          return {x, theta};
        }
        return {0.5f * log1pf(r), theta};
      } else {
        float z0 = hypotf(x + 1, y);
        return {logf(z0), theta};
      }
    } else {
      return std::log1p(z);
    }
  }
};

struct LogicalNot {
  __device__ bool operator()(bool x) {
    return !x;
  }
};

struct Negative {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return T{0, 0} - x;
    } else {
      return -x;
    }
  }
};

struct Real {
  template <typename T>
  __device__ auto operator()(complex_t<T> x) {
    return x.real();
  }
};

struct Round {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return {std::rint(x.real()), std::rint(x.imag())};
    } else {
      return std::rint(x);
    }
  }
};

struct Sigmoid {
  template <typename T>
  __device__ T operator()(T x) {
    T y = 1 / (1 + std::exp(std::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};

struct Sign {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_unsigned_v<T>) {
      return x != 0;
    } else if constexpr (is_complex_v<T>) {
      if (x.real() == 0 && x.imag() == 0) {
        return x;
      } else {
        return x / Abs()(x);
      }
    } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
      return static_cast<float>((x > T(0.f)) - (x < T(0.f)));
    } else {
      return (x > T(0)) - (x < T(0));
    }
  }
};

struct Sin {
  template <typename T>
  __device__ T operator()(T x) {
    return std::sin(x);
  }
};

struct Sinh {
  template <typename T>
  __device__ T operator()(T x) {
    return std::sinh(x);
  }
};

struct Square {
  template <typename T>
  __device__ T operator()(T x) {
    return x * x;
  }
};

struct Sqrt {
  template <typename T>
  __device__ T operator()(T x) {
    return std::sqrt(x);
  }
};

struct Rsqrt {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return 1.0f / Sqrt{}(x);
    } else if constexpr (std::is_same_v<T, __half>) {
      return rsqrt(__half2float(x));
    } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
      return rsqrt(__bfloat162float(x));
    } else {
      return rsqrt(x);
    }
  }
};

struct Tan {
  template <typename T>
  __device__ T operator()(T x) {
    return std::tan(x);
  }
};

struct Tanh {
  template <typename T>
  __device__ T operator()(T x) {
    return std::tanh(x);
  }
};

struct ToFP8 {
  template <typename T>
  __device__ uint8_t operator()(T x) {
    return 0; // fp8 not available on HIP
  }
};

struct FromFP8 {
  __device__ float operator()(uint8_t x) {
    return 0.0f; // fp8 stub — not available on HIP
  }
};

} // namespace mlx::core::cu
