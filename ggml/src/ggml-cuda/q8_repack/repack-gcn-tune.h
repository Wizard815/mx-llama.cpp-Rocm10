#pragma once

// Q8_0 repacked GEMM tile config on gfx906 (prefill), see mul-mat.cu (dense, >= 128
// tokens) and mul-mat-id.cu (MoE, 32-wide path). TN = columns per thread (the tile
// is CW*TN tokens wide), NRL = row lanes per block (each thread owns 64/NRL rows;
// must leave 64/NRL a multiple of 4). Defaults are the previous hard-wired values;
// scripts/gfx906-q8-gemm-sweep.sh overrides them per candidate.
#ifndef GGML_RP_Q8_DENSE_TN
#define GGML_RP_Q8_DENSE_TN   MMQ_RP_Q8_TN          // 2
#endif
#ifndef GGML_RP_Q8_DENSE_NRL
#define GGML_RP_Q8_DENSE_NRL  MMQ_RP_Q8_NROW_LANES  // 4
#endif
#ifndef GGML_RP_Q8_MOE_W32_TN
#define GGML_RP_Q8_MOE_W32_TN  1
#endif
#ifndef GGML_RP_Q8_MOE_W32_NRL
#define GGML_RP_Q8_MOE_W32_NRL (MMQ_RP_Q8_NROW_LANES * 2)  // 8
#endif
// MoE: use the 32-wide tile while n_assign < FACTOR * 128 * n_expert (64-wide otherwise)
#ifndef GGML_RP_Q8_MOE_W32_FACTOR
#define GGML_RP_Q8_MOE_W32_FACTOR 2
#endif
