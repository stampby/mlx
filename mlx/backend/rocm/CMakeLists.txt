# Filename rules in rocm backend:
#
# * Use .hip.cpp/.hip.cpph if code contains device code, and .cpp/.h if not.
# * Device-only code should be put in device/ subdir.
# * Files in device/ subdir should not include files outside.
target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arange.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/binary_two.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_contiguous.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_dynamic.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_input.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/conv/gemm_conv.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/conv/gemm_grouped_conv.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/hipblas_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/miopen_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/distributed.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fft.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/block_mask.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/gemv.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cublas_gemm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/grouped_gemm_unaligned.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/hadamard.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/jit_module.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/kernel_utils.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/random.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/all_reduce.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/col_reduce.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/init_reduce.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/row_reduce.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rope.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/scaled_dot_product_attention.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/scaled_dot_product_attention.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/scan.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/sort.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ternary.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/affine_quantize.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/fp_quantize.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/quantized.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qqmm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qqmm_utils.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/convert_fp8.hip.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/binary)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/quantized/qmm)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/unary)

# fp4 is not available on < 12.8
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8.0)
  target_include_directories(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/quantized/)
  target_sources(mlx
                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/quantized/no_qqmm_impl.cpp)
else()
  target_sources(
    mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qqmm_impl.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/quantized/cublas_qqmm.cpp)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9.0)
  target_sources(
    mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cublas_gemm_batched_12_9.hip.cpp)
else()
  target_sources(
    mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cublas_gemm_batched_12_0.cpp)
endif()

# Embed kernel sources in binary for JIT compilation.
file(
  GLOB MLX_JIT_SOURCES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "${CMAKE_CURRENT_SOURCE_DIR}/device/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/device/*.hip.cpph")
string(JOIN ":" MLX_JIT_SOURCES_ARG ${MLX_JIT_SOURCES})
add_custom_command(
  OUTPUT gen/cuda_jit_sources.h
  COMMAND
    ${CMAKE_COMMAND} -DMLX_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
    -DMLX_JIT_SOURCES=${MLX_JIT_SOURCES_ARG} -P
    "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake"
  DEPENDS bin2h.cmake ${MLX_JIT_SOURCES})
add_custom_target(rocm_jit_sources DEPENDS gen/cuda_jit_sources.h)
add_dependencies(mlx rocm_jit_sources)
target_include_directories(mlx PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen")

# ------------------------ ROCm/HIP Compilation configs ------------------------

target_compile_definitions(mlx PRIVATE MLX_USE_HIP)
target_compile_definitions(mlx PRIVATE __HIP_PLATFORM_AMD__)

# HIP files use .hip.cpp extension — set their language to HIP
file(GLOB_RECURSE HIP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.hip.cpp")
set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)

# GPU architecture target
set(GPU_TARGETS "gfx1151" CACHE STRING "AMD GPU targets")
set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})

# HIP compile options
target_compile_options(mlx PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:-fno-strict-aliasing>")
target_compile_options(mlx PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>")

message(STATUS "ROCm GPU targets: ${GPU_TARGETS}")

# No CUTLASS on ROCm — skip Hopper-specific kernels entirely
# ROCm equivalent optimizations use hipBLASLt

# ------------------------ ROCm Dependencies ------------------------

# Make HIP runtime APIs available in all files
target_include_directories(mlx PRIVATE ${hip_INCLUDE_DIRS} /opt/rocm/include)

# Link ROCm libraries
target_link_libraries(mlx PRIVATE hip::device)

# hipBLAS / hipBLASLt (replaces cuBLAS)
if(hipblas_FOUND)
  target_link_libraries(mlx PRIVATE roc::hipblas)
endif()
if(hipblaslt_FOUND)
  target_link_libraries(mlx PRIVATE roc::hipblaslt)
endif()
if(rocblas_FOUND)
  target_link_libraries(mlx PRIVATE roc::rocblas)
endif()

# rocFFT (replaces cuFFT)
find_package(rocfft QUIET)
if(rocfft_FOUND)
  target_link_libraries(mlx PRIVATE roc::rocfft)
  message(STATUS "rocFFT found")
endif()

# MIOpen (replaces cuDNN)
if(miopen_FOUND)
  target_link_libraries(mlx PRIVATE MIOpen)
endif()

# RCCL (replaces NCCL) — for distributed
find_package(rccl QUIET)
if(rccl_FOUND)
  target_link_libraries(mlx PRIVATE rccl)
  message(STATUS "RCCL found")
endif()

# ROCm RPATH
set_property(
  TARGET mlx
  APPEND
  PROPERTY INSTALL_RPATH "/opt/rocm/lib")
