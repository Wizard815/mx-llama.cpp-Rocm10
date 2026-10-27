#pragma once

// Optional flash_attn_tile config overrides for GCN (gfx906), see
// ggml_cuda_fattn_tile_get_config_amd_gcn in fattn-tile.cuh. Empty by default,
// so GCN uses the generic AMD table. scripts/gfx906-fattn-tile-sweep.sh
// rewrites this file per candidate and restores it afterwards. Example:
//
//   #define GGML_FATTN_TILE_GCN_256_32 256, 2, 64, 64
