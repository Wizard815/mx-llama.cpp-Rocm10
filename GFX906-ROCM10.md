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
