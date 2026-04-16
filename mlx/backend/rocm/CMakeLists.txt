# ============================================================
# MLX ROCm/HIP Backend
# Hipified from CUDA backend for AMD gfx1151 (Strix Halo)
#
# Strategy: HIP device code compiled by hipcc into a separate
# static library (mlx_rocm_kernels), then linked into mlx.
# This prevents HIP compile flags from leaking to CXX sources.
# ============================================================

# Collect all .hip.cpp files — these need hipcc
file(GLOB_RECURSE ROCM_HIP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.hip.cpp")

# Collect all pure .cpp files — these use standard CXX
set(ROCM_CXX_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hipblas_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/miopen_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cublas_gemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cublas_gemm_batched_12_0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scaled_dot_product_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized/quantized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized/qqmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized/no_qqmm_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp
)

# Add pure CXX sources directly to mlx target
target_sources(mlx PRIVATE ${ROCM_CXX_SOURCES})
target_include_directories(mlx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /opt/rocm/include
    /opt/rocm/include/hip
    /opt/rocm/include/hipblas
    /opt/rocm/include/rocblas
)

# Build HIP kernels as a separate static library using hipcc
add_library(mlx_rocm_kernels STATIC ${ROCM_HIP_SOURCES})

# HIP kernel compile settings
set_target_properties(mlx_rocm_kernels PROPERTIES
    LINKER_LANGUAGE CXX
)

# Use hipcc for .hip.cpp files via custom compile rules
target_compile_options(mlx_rocm_kernels PRIVATE
    -fPIC
    -O3
    --offload-arch=${GPU_TARGETS}
    -D__HIP_PLATFORM_AMD__
    -DMLX_USE_HIP
)

target_include_directories(mlx_rocm_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_SOURCE_DIR}
    /opt/rocm/include
    /opt/rocm/include/hip
    /opt/rocm/include/hipblas
    /opt/rocm/include/rocblas
)

# Set hipcc as the compiler for this target
set_target_properties(mlx_rocm_kernels PROPERTIES
    CXX_COMPILER_LAUNCHER ""
    CXX_STANDARD 20
)

# Force hipcc for all sources in this target
set_source_files_properties(${ROCM_HIP_SOURCES} PROPERTIES
    COMPILE_OPTIONS "--offload-arch=${GPU_TARGETS};-D__HIP_PLATFORM_AMD__"
    LANGUAGE CXX
)

# Link the HIP kernel library into mlx
target_link_libraries(mlx PRIVATE mlx_rocm_kernels)

# Link ROCm libraries to both targets
find_package(hip REQUIRED)
target_link_libraries(mlx_rocm_kernels PUBLIC hip::device)
target_link_libraries(mlx PRIVATE hip::device)

find_package(hipblas QUIET)
if(hipblas_FOUND)
    target_link_libraries(mlx PRIVATE roc::hipblas)
endif()

find_package(hipblaslt QUIET)
if(hipblaslt_FOUND)
    target_link_libraries(mlx PRIVATE roc::hipblaslt)
endif()

find_package(rocblas QUIET)
if(rocblas_FOUND)
    target_link_libraries(mlx PRIVATE roc::rocblas)
endif()

find_package(rocfft QUIET)
if(rocfft_FOUND)
    target_link_libraries(mlx PRIVATE roc::rocfft)
endif()

find_package(miopen QUIET)
if(miopen_FOUND)
    target_link_libraries(mlx PRIVATE MIOpen)
endif()

find_package(rccl QUIET)
if(rccl_FOUND)
    target_link_libraries(mlx PRIVATE rccl)
endif()

# JIT sources for runtime compilation
file(GLOB MLX_JIT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/device/*.hip.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/device/*.h")
string(JOIN ":" MLX_JIT_SOURCES_ARG ${MLX_JIT_SOURCES})
add_custom_command(
    OUTPUT gen/cuda_jit_sources.h
    COMMAND ${CMAKE_COMMAND}
        -DMLX_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
        -DMLX_JIT_SOURCES=${MLX_JIT_SOURCES_ARG}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake"
    DEPENDS bin2h.cmake ${MLX_JIT_SOURCES})
add_custom_target(rocm_jit_sources DEPENDS gen/cuda_jit_sources.h)
add_dependencies(mlx rocm_jit_sources)
target_include_directories(mlx PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen")

# Compile definitions
target_compile_definitions(mlx PRIVATE MLX_USE_HIP)

# RPATH
set_property(TARGET mlx APPEND PROPERTY INSTALL_RPATH "/opt/rocm/lib")
