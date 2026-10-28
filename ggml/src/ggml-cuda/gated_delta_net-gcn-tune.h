#pragma once

// gfx906 (GCN) config of the chunked Gated DeltaNet kernel at head size 128, see
// launch_gated_delta_net_chunk in gated_delta_net_chunk.cu. The defaults below match the
// previous hard-wired values; scripts/gfx906-gdn-sweep.sh overrides them per candidate.
#ifndef GGML_GDN_GCN_128_NWARPS
#define GGML_GDN_GCN_128_NWARPS 2 // warps per block (each 64-lane warp owns Tc state columns)
#endif
#ifndef GGML_GDN_GCN_128_TC
#define GGML_GDN_GCN_128_TC     2 // state columns per warp
#endif
#ifndef GGML_GDN_GCN_128_UNROLL
#define GGML_GDN_GCN_128_UNROLL 0 // 1: unroll the column loop so reductions interleave
#endif
