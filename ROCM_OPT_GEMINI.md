# ROCm Optimizations to Match llama.cpp Performance

Based on the benchmark results, the MLX ROCm backend underperforms `llama.cpp`. Here are the key areas for optimization:

### 1. Enable and Optimize Fused Flash Attention (SDPA)
- **Prefill (Flash Attention):** Implement a proper Triton-like Flash Attention kernel for ROCm (e.g., ported from AMD's Flash Attention or ROCm Composable Kernel) to handle large sequences efficiently during prompt processing.
- **Decode (Vector Attention):** Fix the stability issues in the existing `sdpa_vector` kernel so it can be enabled for autoregressive decoding (M=1). Currently, `ScaledDotProductAttention::use_fallback` unconditionally returns `true` because the ROCm kernel is marked as unstable for GQA and causal masking.

### 2. Fix QMM Prefill (Matrix-Matrix) Memory Thrashing
- **Dequantize-to-rocBLAS:** Fix the memory access bugs in the disabled `use_rocblas_dequant_path()` (gated by `MLX_ROCM_QMM_DEQUANT_GEMM`). Fusing a fast block-dequantization into a temporary FP16 buffer, followed by `rocblas_hgemm`, is exactly how `llama.cpp` achieves fast prefill.
- **Shared Memory Tiling:** Alternatively, implement a proper quantized GEMM kernel that loads blocks of X and W into shared memory (LDS) to reuse the weight matrix elements across the M dimension.

### 3. Hardware-Accelerated QMV Decode (Dot Products)
- **DP4A Instructions:** Replace the sequential software FMA with AMD's 4-byte packed dot product instructions (e.g., `__builtin_amdgcn_sdot4` or `__builtin_amdgcn_sdot8`). Grouping reads into `uint32` and using integer dot-products before scaling will double the decoding throughput.
- **Software FP8/FP4 Emulation:** The custom `fp8_e4m3_to_float` and `fp4_e2m1_to_float` functions use expensive bitwise operations and branching. These should be replaced with hardware conversion intrinsics (if using RDNA3/MI300) or optimized via fast shared-memory lookup tables.

### 4. Improve GEMV Bandwidth Utilization
- **Shared Memory Reduction:** Use `__shared__` memory for cross-warp and cross-block reductions instead of doing everything atomically or at the grid level.
- **Sub-Warp Tiling:** `llama.cpp` tunes wavefront/warp sizes and thread mapping per architecture (RDNA vs CDNA) to ensure 100% vector ALU utilization during `SGEMV` operations, preventing LDS bank conflicts and memory stalls. Ensure `gemv.hip` queries device wave sizes and tiles accordingly.
