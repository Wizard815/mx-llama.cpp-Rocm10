#pragma once

// Optional flash_attn_tile config overrides for GCN (gfx906), see
// ggml_cuda_fattn_tile_get_config_amd_gcn in fattn-tile.cuh. Empty by default,
// so GCN uses its built-in values there. scripts/gfx906-fattn-tile-sweep.sh
// rewrites this file per candidate and restores it afterwards. Example:
//
//   #define GGML_FATTN_TILE_GCN_256_32 256, 2, 32, 128
//   #define GGML_FATTN_TILE_GCN_256_16 256, 2, 64, 32   // also 128_32, 64_32
