// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"

#include <sstream>

namespace mlx::core {

namespace rocm {

struct FusedKernelBuilder {
  std::string os;
  const std::string& kernel_name;
  const std::vector<array>& inputs;
  const std::vector<array>& outputs;
  const std::vector<array>& tape;
  const std::function<bool(size_t)>& is_constant;

  void build(const char* name, bool contiguous) {
    NodeNamer namer;

    // Function parameters.
    std::vector<std::string> params;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (is_constant(i)) {
        continue;
      }
      const auto& x = inputs[i];
      const std::string& xname = namer.get_name(x);
      params.push_back(
          std::string("const ") + dtype_to_hip_type(x.dtype()) + "* " + xname);
      if (!is_scalar(x) && !contiguous) {
        params.push_back(
            std::string("const hip::std::array<int64_t, NDIM> ") + xname +
            "_strides");
      }
    }
    for (const auto& x : outputs) {
      params.push_back(
          std::string(dtype_to_hip_type(x.dtype())) + "* " + namer.get_name(x));
    }
    if (!contiguous) {
      params.push_back("const hip::std::array<int32_t, NDIM> shape");
    }
    params.push_back("IdxT size");

    // Build function signature.
    if (contiguous) {
      os += "template <typename IdxT = uint32_t, int work_per_thread = 1>\n";
    } else {
      os +=
          "template <int NDIM, typename IdxT = uint32_t, int work_per_thread = 1>\n";
    }
    os += "__global__ void " + kernel_name + name + "(\n";
    for (size_t i = 0; i < params.size(); ++i) {
      os += "    ";
      os += params[i];
      if (i != params.size() - 1) {
        os += ",\n";
      }
    }
    os += ") {\n";

    // Index. For non contiguous kernels we create a separate index
    // variable per variable otherwise everyone uses `index`.
    os +=
        "  IdxT index = (blockIdx.x * blockDim.x + threadIdx.x) * work_per_thread;\n"
        "  if (index >= size) {\n"
        "    return;\n"
        "  }\n";
    if (!contiguous) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += "  IdxT " + xname + "_idx = 0;\n";
      }
      os += "  {\n";
      os += "    IdxT loc = index;\n";
      os +=
          "    #pragma unroll\n"
          "    for (int i = NDIM - 1; i >= 0; i--) {\n";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += "      " + xname + "_idx += (loc \% shape[i]) * IdxT(" + xname +
            "_strides[i]);\n";
      }
      os +=
          "      loc /= shape[i];\n"
          "    }\n"
          "  }\n";
    }

    // Work loop
    if (!contiguous) {
      os +=
          "\n"
          "  for (int i = 0; i < work_per_thread && index + i < size; i++) {\n";
    } else {
      os +=
          "\n"
          "  #pragma unroll\n"
          "  for (int i = 0; i < work_per_thread; i++) {\n"
          "    if (index + i >= size) break;\n";
    }

    // Read inputs.
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& x = inputs[i];
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_hip_type(x.dtype());
      std::string value;
      if (is_constant(i)) {
        std::ostringstream ss;
        print_constant(ss, x);
        value = std::string("static_cast<") + type + ">(" + ss.str() + ")";
      } else if (is_scalar(x)) {
        value = xname + "[0]";
      } else if (contiguous) {
        value = xname + "[index + i]";
      } else {
        value = xname + "[" + xname + "_idx]";
      }
      os +=
          std::string("    ") + type + " tmp_" + xname + " = " + value + ";\n";
    }

    // Write tape.
    for (const auto& x : tape) {
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_hip_type(x.dtype());
      std::string value;
      if (is_static_cast(x.primitive())) {
        value = std::string("static_cast<") + type + ">(tmp_" +
            namer.get_name(x.inputs()[0]) + ")";
      } else {
        value = x.primitive().name();
        value += "{}(";
        for (size_t i = 0; i < x.inputs().size() - 1; ++i) {
          value += "tmp_" + namer.get_name(x.inputs()[i]) + ", ";
        }
        value += "tmp_" + namer.get_name(x.inputs().back()) + ")";
      }
      os +=
          std::string("    ") + type + " tmp_" + xname + " = " + value + ";\n";
    }

    // Write output.
    for (const auto& x : outputs) {
      std::string xname = namer.get_name(x);
      if (contiguous) {
        os +=
            std::string("    ") + xname + "[index + i] = tmp_" + xname + ";\n";
      } else {
        os +=
            std::string("    ") + xname + "[index + i] = tmp_" + xname + ";\n";
      }
    }

    // End of work loop
    if (!contiguous) {
      os += "\n";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += std::string("    ") + xname + "_idx += " + xname +
            "_strides[NDIM - 1];\n";
      }
    }
    os += "  }\n";

    os += "}\n";
  }
};

} // namespace rocm

constexpr const char* g_jit_includes = R"(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// Standard type definitions for JIT compilation
using uint32_t = unsigned int;
using int32_t = signed int;
using uint64_t = unsigned long long;
using int64_t = signed long long;
using uint16_t = unsigned short;
using int16_t = signed short;
using uint8_t = unsigned char;
using int8_t = signed char;
using size_t = unsigned long;

// Simple array type for JIT compilation (hip/std/array not available in hiprtc)
namespace hip {
namespace std {
template <typename T, int N>
struct array {
  T data_[N];
  __device__ T& operator[](int i) { return data_[i]; }
  __device__ const T& operator[](int i) const { return data_[i]; }
};

template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<float> {
  __device__ static float infinity() { return __int_as_float(0x7f800000); }
};
} // namespace std
} // namespace hip

// Include device operations
namespace mlx::core::rocm {

// Binary ops — promote half/bfloat16 through float to avoid precision loss
// that compounds across 28-36 transformer layers in LLM inference.
struct Add {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(static_cast<float>(x) + static_cast<float>(y));
  }
};

struct Subtract {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(static_cast<float>(x) - static_cast<float>(y));
  }
};

struct Multiply {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(static_cast<float>(x) * static_cast<float>(y));
  }
};

struct Divide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(static_cast<float>(x) / static_cast<float>(y));
  }
};

struct Maximum {
  template <typename T>
  __device__ T operator()(T x, T y) { return x > y ? x : y; }
};

struct Minimum {
  template <typename T>
  __device__ T operator()(T x, T y) { return x < y ? x : y; }
};

struct Power {
  template <typename T>
  __device__ T operator()(T base, T exp) { return powf(base, exp); }
};

struct Equal {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x == y; }
};

struct NotEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x != y; }
};

struct Greater {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x > y; }
};

struct GreaterEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x >= y; }
};

struct Less {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x < y; }
};

struct LessEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x <= y; }
};

struct LogicalAnd {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x && y; }
};

struct LogicalOr {
  template <typename T>
  __device__ bool operator()(T x, T y) { return x || y; }
};

struct ArcTan2 {
  template <typename T>
  __device__ T operator()(T y, T x) {
    return T(atan2f(static_cast<float>(y), static_cast<float>(x)));
  }
};

struct Remainder {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(fmodf(static_cast<float>(x), static_cast<float>(y)));
  }
};

struct FloorDivide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return T(truncf(static_cast<float>(x) / static_cast<float>(y)));
  }
};

struct LogAddExp {
  template <typename T>
  __device__ T operator()(T x, T y) {
    float fx = static_cast<float>(x);
    float fy = static_cast<float>(y);
    float maxval = fx > fy ? fx : fy;
    float minval = fx > fy ? fy : fx;
    return T(maxval + log1pf(expf(minval - maxval)));
  }
};

struct BitwiseAnd {
  template <typename T>
  __device__ T operator()(T x, T y) { return x & y; }
};

struct BitwiseOr {
  template <typename T>
  __device__ T operator()(T x, T y) { return x | y; }
};

struct BitwiseXor {
  template <typename T>
  __device__ T operator()(T x, T y) { return x ^ y; }
};

struct LeftShift {
  template <typename T>
  __device__ T operator()(T x, T y) { return x << y; }
};

struct RightShift {
  template <typename T>
  __device__ T operator()(T x, T y) { return x >> y; }
};

// All unary math ops promote through float to support half/bfloat16.
// For float inputs the static_cast is a no-op.
#define UNARY_FLOAT_OP(name, op) \
struct name { \
  template <typename T> \
  __device__ T operator()(T x) { \
    return T(op(static_cast<float>(x))); \
  } \
};

// Unary ops
UNARY_FLOAT_OP(Abs, fabsf)
UNARY_FLOAT_OP(Exp, expf)
UNARY_FLOAT_OP(Log, logf)
UNARY_FLOAT_OP(Sqrt, sqrtf)

struct Negative {
  template <typename T>
  __device__ T operator()(T x) { return -x; }
};

struct Square {
  template <typename T>
  __device__ T operator()(T x) { return x * x; }
};

struct Sigmoid {
  template <typename T>
  __device__ T operator()(T x) {
    float fx = static_cast<float>(x);
    float y = 1.0f / (1.0f + expf(-fabsf(fx)));
    return T((fx < 0.0f) ? 1.0f - y : y);
  }
};

UNARY_FLOAT_OP(Tanh, tanhf)
UNARY_FLOAT_OP(Sin, sinf)
UNARY_FLOAT_OP(Cos, cosf)
UNARY_FLOAT_OP(Tan, tanf)
UNARY_FLOAT_OP(Sinh, sinhf)
UNARY_FLOAT_OP(Cosh, coshf)
UNARY_FLOAT_OP(Erf, erff)
UNARY_FLOAT_OP(ErfInv, erfinvf)
UNARY_FLOAT_OP(Expm1, expm1f)
UNARY_FLOAT_OP(Log1p, log1pf)
UNARY_FLOAT_OP(Log2, log2f)
UNARY_FLOAT_OP(Log10, log10f)
UNARY_FLOAT_OP(Ceil, ceilf)
UNARY_FLOAT_OP(Floor, floorf)
UNARY_FLOAT_OP(Round, rintf)
UNARY_FLOAT_OP(Rsqrt, rsqrtf)

struct Sign {
  template <typename T>
  __device__ T operator()(T x) {
    float fx = static_cast<float>(x);
    return T((fx > 0.0f) - (fx < 0.0f));
  }
};

UNARY_FLOAT_OP(Asin, asinf)
UNARY_FLOAT_OP(Acos, acosf)
UNARY_FLOAT_OP(Atan, atanf)
UNARY_FLOAT_OP(Asinh, asinhf)
UNARY_FLOAT_OP(Acosh, acoshf)
UNARY_FLOAT_OP(Atanh, atanhf)

struct LogicalNot {
  template <typename T>
  __device__ bool operator()(T x) { return !x; }
};

struct BitwiseNot {
  template <typename T>
  __device__ T operator()(T x) { return ~x; }
};

#undef UNARY_FLOAT_OP

struct Reciprocal {
  template <typename T>
  __device__ T operator()(T x) { return T(1) / x; }
};

// Ternary ops
struct Select {
  template <typename T>
  __device__ T operator()(bool c, T x, T y) { return c ? x : y; }
};

// Broadcast is a no-op in fused kernels (handled by indexing)
struct Broadcast {
  template <typename T>
  __device__ T operator()(T x) { return x; }
};

} // namespace mlx::core::rocm

#define inf hip::std::numeric_limits<float>::infinity()
)";

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();

  // Determine the work per thread for the vectorized reads/writes.
  int max_size = 1;
  for (const auto& x : outputs) {
    max_size = (max_size > x.itemsize()) ? max_size : x.itemsize();
  }
  int work_per_thread = 16 / max_size;

  rocm::JitModule& mod = rocm::get_jit_module(s.device, lib_name(), [&]() {
    // Build source code.
    rocm::FusedKernelBuilder builder{
        g_jit_includes, lib_name(), inputs_, outputs_, tape_, is_constant_};
    builder.os += "namespace mlx::core::rocm {\n\n";
    builder.build("_contiguous", true);
    builder.os += "\n";
    builder.build("_strided", false);
    builder.os += "\n} // namespace mlx::core::rocm\n";

    // Build kernel names.
    std::vector<std::string> kernel_names;
    kernel_names.push_back(
        std::string("mlx::core::rocm::") + lib_name() +
        "_contiguous<uint32_t, " + std::to_string(work_per_thread) + ">");
    kernel_names.push_back(
        std::string("mlx::core::rocm::") + lib_name() +
        "_contiguous<int64_t, " + std::to_string(work_per_thread) + ">");
    for (auto wpt : std::array<int, 2>{1, work_per_thread}) {
      for (int i = 1; i <= MAX_NDIM; ++i) {
        kernel_names.push_back(
            std::string("mlx::core::rocm::") + lib_name() + "_strided<" +
            std::to_string(i) + ", uint32_t, " + std::to_string(wpt) + ">");
        kernel_names.push_back(
            std::string("mlx::core::rocm::") + lib_name() + "_strided<" +
            std::to_string(i) + ", int64_t, " + std::to_string(wpt) + ">");
      }
    }

    return std::make_tuple(
        false, std::move(builder.os), std::move(kernel_names));
  });

  // Collapse contiguous dims to route to a faster kernel if possible.
  auto [contiguous, shape, strides_vec] =
      compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);

  // Whether to use large index.
  bool large = compiled_use_large_index(inputs, outputs, contiguous);

  rocm::KernelArgs args;
  // Put inputs.
  int strides_index = 1;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i)) {
      continue;
    }
    const auto& x = inputs[i];
    args.append(x);
    if (!contiguous && !is_scalar(x)) {
      args.append_ptr(strides_vec[strides_index++].data());
    }
  }

  // Put outputs.
  compiled_allocate_outputs(inputs, outputs, is_constant_, contiguous);
  for (auto& x : outputs) {
    args.append(x);
  }

  // Put shape and size.
  if (!contiguous) {
    args.append_ptr(shape.data());
  }
  if (large) {
    args.append<int64_t>(outputs[0].data_size());
  } else {
    args.append<uint32_t>(outputs[0].data_size());
  }

  // Choose work per thread
  if (!contiguous && shape.back() % work_per_thread != 0) {
    work_per_thread = 1;
  }

  // Launch kernel.
  const char* index_type = large ? "int64_t" : "uint32_t";
  std::string kernel_name = std::string("mlx::core::rocm::") + lib_name();
  if (contiguous) {
    kernel_name += std::string("_contiguous<") + index_type + ", " +
        std::to_string(work_per_thread) + ">";
  } else {
    kernel_name += std::string("_strided<") + std::to_string(shape.size()) +
        ", " + index_type + ", " + std::to_string(work_per_thread) + ">";
  }

  auto& encoder = rocm::get_command_encoder(s);
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  for (const auto& out : outputs) {
    encoder.set_output_array(out);
  }

  auto kernel = mod.get_kernel(kernel_name);

  // Calculate launch configuration
  int block_size = 256;
  int64_t total_work =
      (outputs[0].data_size() + work_per_thread - 1) / work_per_thread;
  int num_blocks = (total_work + block_size - 1) / block_size;

  encoder.launch_kernel([&](hipStream_t stream) {
    (void)hipModuleLaunchKernel(
        kernel,
        num_blocks,
        1,
        1,
        block_size,
        1,
        1,
        0,
        stream,
        args.args(),
        nullptr);
  });
}

} // namespace mlx::core
