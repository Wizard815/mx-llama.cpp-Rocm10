#pragma once

// Q8_0 repacked GEMM tile config on gfx906 (prefill), see mul-mat.cu (dense, >= 128
// tokens) and mul-mat-id.cu (MoE, 32-wide path). TN = columns per thread (the tile
// is CW*TN tokens wide), NRL = row lanes per block (each thread owns 64/NRL rows;
// must leave 64/NRL a multiple of 4), MINB = __launch_bounds__ second argument, which
// HIP turns into amdgpu_waves_per_eu: minimum waves per SIMD, so the VGPR budget is
// 256/MINB (2 -> 128, 3 -> 84). Defaults are the previous hard-wired values;
// scripts/gfx906-q8-gemm-sweep.sh overrides them per candidate.
#ifndef GGML_RP_Q8_DENSE_TN
#define GGML_RP_Q8_DENSE_TN   MMQ_RP_Q8_TN          // 2
#endif
#ifndef GGML_RP_Q8_DENSE_NRL
#define GGML_RP_Q8_DENSE_NRL  MMQ_RP_Q8_NROW_LANES  // 4
#endif
// MoE lane width: the tile is CW*TN tokens and a thread owns 64/NRL rows x TN tokens.
// gfx906 default CW 8 / TN 4 / NRL 8: the 32-token tile (~ tokens per expert at
// -ub 1024, 256 experts) with each LDS weight read feeding 4 tokens instead of 1.
// Against CW 32 / TN 1 (the old hard-wired tile) on Tiel-Coder-35B-A3B Q8_0, 2x MI50:
// LDS wait / VALU active 1.27 -> 0.01, MoE up -6.5%, down -3.2%, pp2048 +2.1%,
// pp8192 +1.9%, PPL identical. One-wave blocks at 14 KiB LDS run 1 wave/SIMD
// whatever the VGPR count, so the compiler drops MINB here (196 VGPR, no spills).
#ifndef GGML_RP_Q8_MOE_W32_CW
#define GGML_RP_Q8_MOE_W32_CW  8
#endif
#ifndef GGML_RP_Q8_MOE_W32_TN
#define GGML_RP_Q8_MOE_W32_TN  4
#endif
#ifndef GGML_RP_Q8_MOE_W32_NRL
#define GGML_RP_Q8_MOE_W32_NRL 8
#endif
// MoE tail tiles: each expert's tokens past its last full CW*TN tile run in a second
// launch with a narrower tile (16: CW 8 TN 2 NRL 8, 8: CW 8 TN 1 NRL 8; 0 = off, one
// launch as before). GGML_CUDA_REPACK_MOE_TAIL=0/8/16 overrides it at run time.
#ifndef GGML_RP_Q8_MOE_TAIL_BN
#define GGML_RP_Q8_MOE_TAIL_BN 0
#endif
// MoE: use the 32-wide tile while n_assign < FACTOR * 128 * n_expert (64-wide otherwise)
#ifndef GGML_RP_Q8_MOE_W32_FACTOR
#define GGML_RP_Q8_MOE_W32_FACTOR 2
#endif
#ifndef GGML_RP_Q8_DENSE_MINB
#define GGML_RP_Q8_DENSE_MINB 2
#endif
#ifndef GGML_RP_Q8_MOE_W32_MINB
#define GGML_RP_Q8_MOE_W32_MINB 2
#endif
