#pragma once

// gfx906 (GCN) config of the chunked Gated DeltaNet kernel at head size 128, see
// launch_gated_delta_net_chunk in gated_delta_net_chunk.cu; scripts/gfx906-gdn-sweep.sh
// overrides these per candidate. Sweep on MI50 (16 key / 32 value heads, us/run at
// 1024 / 2048 tokens): previous 2 warps, 2 cols, no unroll = 3598 / 7631;
// 4 warps, 2 cols, unroll = 3353 / 6798 (-7% / -11%). Unrolling won in every pairing;
// 1 column per warp (+52%) and 8 (+11..16%) were slower.
#ifndef GGML_GDN_GCN_128_NWARPS
#define GGML_GDN_GCN_128_NWARPS 4 // warps per block (each 64-lane warp owns Tc state columns)
#endif
#ifndef GGML_GDN_GCN_128_TC
#define GGML_GDN_GCN_128_TC     2 // state columns per warp
#endif
#ifndef GGML_GDN_GCN_128_UNROLL
#define GGML_GDN_GCN_128_UNROLL 1 // 1: unroll the column loop so reductions interleave
#endif
