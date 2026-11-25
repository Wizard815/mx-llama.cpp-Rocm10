# gfx906 + ROCm 10 combo build

This fork ports gfx906 (AMD Instinct MI50/MI60)-specific fixes and optimizations from
several community forks on top of [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp),
built against a custom ROCm 10 distribution from
[TheRock-gfx906](https://github.com/Wizard815/TheRock-gfx906) (gfx906 isn't an officially
supported target upstream — official support starts at gfx908).

## Upstream base

Merged up to mxxm-t/mx-llama.cpp `c4d12e2c0` (Oct 15 2026, includes llama.cpp tag b11553),
plus these later ggml-org/llama.cpp commits cherry-picked for HIP/GCN:

- `62bacfa01` conv2d: SIMT implicit GEMM on AMD without WMMA (gfx906 has no WMMA/MFMA)
- `9d1878a43` HIP: `__builtin_amdgcn_perm` for Q1_0 unpacking
- `2744e48d1` faster MMVQ for low-bit types, incl. GCN nwarps tuning for Q3_K/Q6_K
- `5aeac37a3` fattn-tile: skip OOB checks for the last full KV tile
- `99db719bd` fattn-vec: fix shared-memory write-after-read hazard on KQ
- `b68449857` mmf: use the device warp size (wave64) in the MUL_MAT_ID MMF check
- `623c16393` rms_norm float4 variants (merged with mx-llama.cpp's clamped-grid kernel flag)
- `13786fbb3` drop captured CUDA/HIP graphs when the legacy pool frees memory (gfx906 uses the
  legacy pool with `GGML_HIP_NO_VMM=ON` and HIP graphs)

## What's ported here that isn't in mxxm-t/mx-llama.cpp

- `ggml/src/ggml-cuda/rope.cu` - `__sincosf` fusion in `rope_yarn` (from iacopPBK/llama.cpp-gfx906)
- `ggml/src/ggml-cuda/gfx906-sgemm.cuh` - custom tiled SGEMM kernel for gfx906 (from iacopPBK)
- `ggml/src/ggml-cuda/mmq-config-gfx906.cuh` - one MI50-measured Q5_K J=64 tile override
  (I=64 vs the GCN table's I=128, +15-35% measured) from mixa3607/ML-gfx906's
  `mxxm-gfx906-kcase.patch`
- `mmq-config-gcn.cuh` - GCN MMQ retune from llama.cpp draft PR #30021 (8-wave tiles and J=24/40/48
  rows at small batch sizes, dp4a only; also used by gfx906 under the overrides above). Only matters
  for weights MMQ sees: types the repack buffer does not take (Q4_0, Q4_1, Q5_0, Q2_K, Q3_K, IQ1-3,
  IQ4_XS, ...) and every type under `--no-repack`. `-DGGML_HIP_GCN_MMQ_LEGACY=ON` builds the previous
  table (`mmq-config-gcn-legacy.cuh`) for A/B. Plus the #30837 static_assert (DP4A tile height must
  be a multiple of the warp size).
- `-DGGML_FATTN_TILE_GCN_UPSTREAM_30344` (in `CMAKE_HIP_FLAGS`) opts into the two FA tile rows of
  llama.cpp draft PR #30344 the gfx906 default lacks: (256,256,16) and (64,64,32) = 256,2,64,32.
  Off by default, not measured on MI50.
- `GGML_OP_SOLVE_TRI` crash fix - gfx906's rocBLAS `strsm` has no gfx906 kernel, so for
  n > 64 or k > 32 the op is reported unsupported on gfx906 and runs on the CPU instead of
  crashing (affects Gated Delta Network models: Qwen3.5, Qwen3-Next, Kimi Linear)
- Turbo KV cache compression (`--cache-type-k/v turbo2/turbo3/turbo4`) - ~3.3x KV cache
  capacity at a generation-speed cost, via a rotate/quantize/dequantize scheme requiring
  flash attention; ported from moriyasujapan/llamacpp-gfx-906-turbo-gemma4 (a bugfixed
  version of arte-fact/llamacpp-gfx-906-turbo's original). The turbo FA vec instances are
  always compiled; the f16-q8_0 instance is no longer forced in, add it with
  `-DGGML_CUDA_FA_QUANTS=f16-q8_0` if you run f16 K with q8_0 V (otherwise that pair is
  converted to f16 at run time). **Multi-GPU layer-split is unverified** - the shared
  rotation-matrix tensor is allocated on whichever GPU's context builds first. Test this
  first on real hardware.
- turbo5 / turbo6 KV cache types (5.125 / 6.125 bits per value, 32/64-entry codebooks, ported
  from TheTom/llama-cpp-turboquant PR #381). They are meant as V types with a q8_0 K
  (`-ctk q8_0 -ctv turbo5|turbo6`); as K types they lose quality. Head size must be 128 or 256.
  FA vec instances: q8_0/f16 K with turbo5/turbo6 V, and turbo5/turbo5, turbo6/turbo6. Mixed
  turbo/plain K/V pairs are rotated correctly: Q is rotated iff K is turbo, the attention
  output is inverse-rotated (with V's rotation) iff V is turbo.
- Tensor-split helpers: meta-backend split rules for the turbo WHT op and for elementwise ops
  between a MIRRORED and a split operand; backend sampling under `-sm tensor` when the output
  is mirrored (also enabled for MTP)
- Custom all-reduce peer access: let the driver decide, clear the sticky HIP error on refusal
- `GGML_HIP_RCCL` defaults to ON
- `-mllvm -amdgpu-sched-strategy=max-ilp` compiler flag recommended at build time (see below)

## Now upstream (dropped from this fork)

- `mmq-config-gcn.cuh` (wave64 GCN MMQ table, llama.cpp PR #27841), the gfx906 MMQ dispatch,
  and the gfx906 J>64 occupancy gate are mx-llama.cpp's own now (the gate lives in the host
  tile-width loop in `mmq.cu`). This fork keeps only the Q5_K override above.
- `gcn_repack/` (Q3_K..Q6_K three-plane repack from sixvolts/llamacpp-gfx906-furnace) was
  already dropped for mx-llama.cpp's native K-quant repack in the q8_0 repack buffer type.
- The llama.cpp cherry-picks this fork carried (#29941, #27694, #29903, #28498, #29675, #29096,
  #29393, #29633, #28751, #27663, #29309, #27530, #29217) and their local fixups are all in
  the new base.

Flash-attention on gfx906 already uses the native `v_dot2_f32_f16` instruction upstream
(`common.cuh`'s `V_DOT2_F32_F16_AVAILABLE` gate includes `__gfx906__`) - no separate port
needed there.

## Runtime switches

Every fork feature that changes speed has an environment switch, so an A/B needs
no rebuild. Defaults are what a plain launch gets.

| Switch | Default | Effect |
|---|---|---|
| `GGML_CUDA_DISABLE_FUSION=1` | off | all CUDA/HIP graph fusions off (upstream switch) |
| `GGML_CUDA_NO_ADD_NORM_FUSION=1` | off | residual ADD -> RMS_NORM -> MUL fusion off |
| `GGML_CUDA_NO_GDN_GATE_FUSION=1` | off | GDN gate ADD -> SOFTPLUS -> MUL (+ SIGMOID) fusion off |
| `GGML_CUDA_NO_GDN_L2_NORM_PAIR_FUSION=1` | off | GDN q/k l2 norms in one launch off |
| `GGML_CUDA_FEWROW_F32=0/512/1024` | 1 (auto) | few-row F32 mat-vec on GCN: off, or a forced block size |
| `GGML_GDN_NO_COLTHREAD=1` | off | skip mx's column-thread GDN kernel; the chunked kernel and its gfx906 tuning run instead |
| `GGML_CUDA_GDN_DECODE_MC=1` | off | GDN decode multi-column layout (measured slower; only reached with `GGML_GDN_NO_COLTHREAD=1`) |
| `GGML_CUDA_REPACK_MOE_OPT=0/1` | 1 | LDS skew in the repacked Q8_0 MoE GEMM |
| `--no-repack` | repack on | plain weight layout, no gfx906 repacked GEMM (the mixa Q5_K MMQ tile only matters then) |
| `GGML_CUDA_MMQ_GCN_STREAM_K=0` | on | GCN MMQ without stream-k (as before upstream #30022) |
| `GGML_ENABLE_CUSTOM_AR=0` | on | tensor-split custom P2P AllReduce off |
| `GGML_TP_AR_BF16_MIN_NE=N` | 0 (off) | AllReduce messages of N+ elements travel as BF16 |
| `GGML_SCHED_BARRIER_ALWAYS=1` | off | the scheduler waits before every split again (old behaviour) |
| `LLAMA_MTP_DEFERRED_PREFILL=0` | on | MTP: build the draft KV per prompt chunk instead of once after prefill |
| `LLAMA_MTP_NO_KV_ONLY=1` | off | MTP: the prefill replay runs the full draft head |
| `LLAMA_MTP_DRAFT_PIPELINE=1` | off | MTP: give the draft context a pipeline ring |
| `LLAMA_MTP_TENSOR_MIRROR=1` | off | MTP under `-sm tensor`: mirror the output head so the GPU sampler can run |
| `LLAMA_MTP_DRAFT_VOCAB=K` | off | MTP draft scores only the first K token ids (layer split) |
| `LLAMA_MOE_RANDOM_ROUTING=1` | off | benchmark only: random expert choice, for MoE cache tests |
| `LLAMA_LOAD_BUFFER_MB=N` | 64 | read chunk for `-lm none` loads (was 1 MB); `-lm none` reloads from the page cache after a swap, `dio` always reads the disk |
| `GGML_CUDA_FA_GQA_PAD=0` | on | tile FA: no padding of GQA groups 3 and 5..7 to 4/8 heads per block (1..8 tokens) |
| `GGML_CUDA_RP_NC_SPLIT=0` | on | repacked dense mat-vec: 9..16 tokens go to the tiled GEMM again instead of two narrow passes |
| `-lm dio-auto` | off (`-lm dio`) | direct I/O, but chunks already in the page cache are read buffered from RAM; checked per chunk with `preadv2(RWF_NOWAIT)`. Linux only, elsewhere it is `-lm none` |
| `LLAMA_DIO_AUTO_FILL=0/1` | auto | `-lm dio-auto` cache misses: 1 = read buffered so the next load is from RAM, 0 = read with O_DIRECT (no cache fill). Auto fills when the file fits in MemAvailable with 1/8 spare |
| `GGML_FA_TILE_GFX906=0` | on | tile FA head 256, 2 Q heads per block: rayakame's MI50 rows for 1..8 tokens (128 threads, nbatch_K 32); used for 1 token always and for 2..8 tokens from 1024 K/V rows on. Off = the GCN table. 2 heads per block means GQA 2 (or 10, 14, ...); GQA 6 gets there only with `GGML_CUDA_FA_GQA_PAD=0`, since by default 1..8 tokens of GQA 3 and 5..7 are padded to 4/8 heads per block. Quantized K/V at 1..2 tokens use the vec kernel instead |
| `GGML_FA_KV_PREFETCH=0` or a mask | all (2,4,8,16,32) | tile FA head 256, 2 Q heads per block (same GQA limits as above; GQA 6 from 9 tokens on uses it with the 32-column GCN row): load the next K/V chunk into registers while the current one is computed. Same math; results can differ in the last bits where a variant's occupancy changes the number of parallel blocks. The value is a bit mask of the column counts that prefetch; 1 token from 32K K/V rows on never prefetches |
| `GGML_HIP_GRAPH_REINSTANTIATE=1` | off | re-instantiate HIP graphs instead of `hipGraphExecUpdate`, which leaks ~1 MiB per request on ROCm (issue 10713) |

Build-time A/B options (not on by default until measured on MI50; Docker: `GFX906_HIP_FLAGS` / `GFX906_HIP_MAX_ILP` in `.env`):

| Build option | Default | Effect |
|---|---|---|
| `-DGGML_HIP_FA_VEC_MIN_BLOCKS=2` or `4` | 1 | FA vec kernel occupancy (llama-of-vega measured 4 faster; with ROCm 6.4 clang this fork's q8_0/turbo vec kernels then spill 200-1100 bytes per lane) |
| `-DGGML_HIP_DPP_REDUCE=1` | off | warp sums with DPP/swizzle instead of LDS shuffles (upstream PR #32215, +5% tg on MI100; not bit-identical) |
| `-DGGML_HIP_MAX_ILP=ON` | OFF | max-ilp per file except `mmvq.cu`; drop the global `-amdgpu-sched-strategy=max-ilp` from the HIP flags then |

Checking the FA tile rows and K/V prefetch on MI50: `scripts/gfx906-fa-prefetch-ab.sh` builds
`build-fa-ab` with tests, reports VGPR spills of the head-256 tile kernels, runs test-backend-ops
FLASH_ATTN_EXT against the CPU and times the head-256 cases with the switches above off and on
(including `GGML_CUDA_FA_GQA_PAD=0`), and runs llama-bench when `MODEL` is set. Results go to
`fa-ab-<date>/summary.md`. `DEVICES="0 1"` runs each card.

## Build

Requires a gfx906 ROCm 10 install from
[TheRock-gfx906](https://github.com/Wizard815/TheRock-gfx906) at `/opt/rocm` (or `$ROCM_PATH`).

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build -GNinja \
    -DGGML_HIP=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_RCCL=ON \
    -DAMDGPU_TARGETS=gfx906 \
    -DCMAKE_BUILD_TYPE=Release \
    -DHIP_COMPILER=clang \
    -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument" \
    -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
    -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release -j"$(nproc)"
```

A companion Docker setup (multi-stage build against the ROCm 10 tarball, with
`/dev/kfd`/`/dev/dri` passthrough for container-based testing) exists locally but is not
yet published.
