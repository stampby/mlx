# ROCm QMV Comparison vs Metal and CUDA

## Scope

This note compares the ROCm quantized matrix-vector hot path (`qmv_warp_shared_kernel`) against the corresponding high-level and kernel strategies in Metal and CUDA backends, and proposes next steps focused on out-of-box performance.

## Current ROCm Path

- Main kernel: `mlx/backend/rocm/quantized/qmm.hip` (`qmv_warp_shared_kernel`, `qmv_warp_shared_batched_kernel`, `gather_qmv_warp_shared_kernel`)
- ROCm strategy today:
  - Stage `x` into shared memory chunks (`CHUNK_SIZE = 2048`)
  - Reuse shared tile across output columns in a warp-shared design
  - Dispatch controlled by QMV heuristics (`threads_per_col`, `cols_per_block`) and dequant+GEMM fallback policy

## CUDA Comparison

- Main kernel path: `mlx/backend/cuda/quantized/qmv.cu` (`fp_qmv_impl`, `fp_qmv_single`, `fp_qmv_batched`)
- CUDA design differences:
  - Uses per-thread vectorized loads and warp reduction (`cooperative_groups`), not shared-memory staging of `x` like ROCm
  - Chooses vectorization width (`n_per_thread` in `{1,2,4}`) from alignment checks at dispatch time
- Important caveat:
  - CUDA quantized matmul support here is not fully symmetric with ROCm affine flow (`mlx/backend/cuda/quantized/quantized.cpp` has Hopper-only affine path and otherwise `QMM NYI`)

## Metal Comparison

- Main kernel families:
  - `mlx/backend/metal/kernels/quantized.h`: `qmv_quad_impl`, `qmv_fast_impl`, `qmv_impl`
  - Dispatch in `mlx/backend/metal/quantized.cpp`
- Metal design differences:
  - Multiple specialized QMV kernels selected by shape
  - Explicit architecture-aware crossover from QMV to QMM via `get_qmv_batch_limit(...)`
  - Gather path optimization (`gather_qmm_rhs`) when expert/rhs indices are sorted and batch pattern is favorable

## High-Level Gap Summary

Compared with Metal (and partially CUDA), ROCm gaps are mostly scheduling/dispatch-level, not just arithmetic micro-kernel details:

1. No Metal-style sorted-index gather optimization path in ROCm GatherQMM scheduler.
2. Less explicit architecture-tiered QMV vs QMM crossover policy.
3. No tiny-K specialized QMV path analogous to Metal's `qmv_quad` route.
4. No CUDA-like alignment-driven vectorization mode selection at ROCm dispatch level.

## Next Steps (Priority Order)

1. **[DONE] Add ROCm GatherQMM sorted-rhs scheduling fast path**
   - Mirror Metal `gather_qmm_rhs` style batching/reuse logic for expert-ordered workloads.
   - Target file: `mlx/backend/rocm/quantized/qmm.hip` (GatherQMM dispatch section).

2. **[DONE] Introduce explicit ROCm QMV/QMM crossover table**
   - Build architecture- and shape-aware thresholds (e.g., `K`, `N`, batch, transpose mode).
   - Keep OOB defaults only; no required runtime knobs.
   - Target file: `mlx/backend/rocm/quantized/qmm.hip`.

3. **[DONE] Add tiny-K specialized QMV dispatch path**
   - Fast route for common decode small-inner-dimension cases to reduce overhead.
   - Target file: `mlx/backend/rocm/quantized/qmm.hip`.

4. **Add alignment-aware ROCm QMV variant selection**
   - Select specialized variants based on pointer alignment and packed layout compatibility.
   - Target file: `mlx/backend/rocm/quantized/qmm.hip`.

5. **Validate with profile gates**
   - Use `rocprof` kernel-trace runs for decode and prefill.
   - Track hotspot share changes for QMV, gather, and copy kernels.

## Success Criteria

- Improve out-of-box decode throughput without requiring user tuning knobs.
- Reduce share of time in generic gather/copy overhead for MoE-like routing patterns.
- Preserve or improve 9B decode while not regressing smaller 2B workloads.
