# Building the gfx906 llama.cpp image yourself

Replaces `zenth815/mx-llama-rocm10-gfx906:latest` from Docker Hub with an image
you compile from `Wizard815/mx-llama.cpp-Rocm10` on the box.

## The chain, before and after

**Before**

```
zenth815/mx-llama-rocm10-gfx906:latest     <- Docker Hub, someone else rebuilds it
        │  Dockerfile.mx-llamacpp-ssh
        ▼
mx-llamacpp-mx-llamacpp-ssh:latest
        │  docker/llama-swap-source.Containerfile  (BASE_IMAGE default, unchanged)
        ▼
llama-swap-rescan:local
```

**After**

```
mx-llama.cpp-Rocm10 checkout
        │  new .devops/mx-llama-rocm10-gfx906.Dockerfile
        ▼
mx-llama-rocm10-gfx906:local               <- you build this, whenever you want
        │  Dockerfile.mx-llamacpp-ssh   (one line changed: the FROM)
        ▼
mx-llamacpp-mx-llamacpp-ssh:latest
        │  docker/llama-swap-source.Containerfile  (unchanged)
        ▼
llama-swap-rescan:local
```

Stage 3 needs no edit at all — its `BASE_IMAGE`/`BASE_TAG` args already default
to `mx-llamacpp-mx-llamacpp-ssh:latest`.

## Commands

```bash
# 1. update the checkout (see UPSTREAM-MERGE.md -- the merge is already applied
#    locally in your workspace clone, just not pushed)
cd /path/to/mx-llama.cpp-Rocm10
git pull

# 2. build llama.cpp for gfx906
cp /path/to/self-build/Dockerfile.mx-llama-rocm10-gfx906 \
   .devops/mx-llama-rocm10-gfx906.Dockerfile
docker build -f .devops/mx-llama-rocm10-gfx906.Dockerfile \
  -t mx-llama-rocm10-gfx906:local .

# 3. prove it is what you think it is before anything else uses it
docker run --rm mx-llama-rocm10-gfx906:local cat /app/versions.txt
docker run --rm --entrypoint llama-server mx-llama-rocm10-gfx906:local \
  --list-devices

# 4. rebuild the ssh layer (drop in the new Dockerfile first)
cd /mnt/user/appdata/mx-llamacpp
cp /path/to/self-build/Dockerfile.mx-llamacpp-ssh .
docker compose up -d --build

# 5. rebuild llama-swap on top
cd /mnt/user/appdata/llamaswap
docker build -f docker/llama-swap-source.Containerfile -t llama-swap-rescan:local .
cd deploy && docker compose up -d
```

Step 3 is the important one and it is cheap: `--list-devices` must print both
MI50s and a ROCm backend. If it prints only CPU, stop — the build produced a
CPU-only server, which is the standard failure when `AMDGPU_TARGETS` or the ROCm
version is wrong.

## Corrected: the base is your TheRock build, not a `rocm/dev-ubuntu` tag

Originally I set `ARG ROCM_VERSION=10.0.0` and a
`rocm/dev-ubuntu-24.04:10.0.0-complete` base. Both were wrong. Your ROCm comes
from **`Wizard815/TheRock-gfx906`** — TheRock, the build system ROCm has used
since 7.14 — and that fork's `version.json` says:

```json
{ "rocm-version": "10.1.0", "release-metadata": { "base-date": "" } }
```

with `GFX906.md` describing the output as *"a self-contained ROCm 10 install tree
for gfx906"* at `build/dist/rocm/`. And there is no stock tag to point at: your
box reports `rocm-smi 4.0.0+97ab4d1f88` against `ROCM-SMI-LIB version: 7.8.0`,
which is a cherry-picked tree — ROCm 7 tooling with bits taken forward from 10,
exactly as you described — not a release of anything.

So the Dockerfile now takes `ARG ROCM_BASE=zenth815/mx-llama-rocm10-gfx906:latest`
— your own image, which already carries that tree — and compiles llama.cpp inside
it. That also makes the compiler and the runtime the same ROCm by construction,
which is the property that actually matters here.

The build stage prints the ROCm tree it found and exits with a clear message if
`hipcc` is missing, so a runtime-only base fails in ten seconds rather than
twenty minutes into a compile. If that happens, the recipe for producing a
TheRock build tree is at the end of the Dockerfile, taken from `GFX906.md` —
including the `meson==1.11.2` pin, the `CMAKE_BUILD_PARALLEL_LEVEL=4` note (the
`-j` flag does not propagate to nested per-subproject `cmake --build` calls), and
why `THEROCK_TEST_AMDGPU_TARGETS=gfx906` matters (the default builds RCCL for ~24
architectures).

`versions.txt` now records both numbers, so `docker run --rm <image> cat
/app/versions.txt` answers "which llama.cpp, which ROCm tree, which TheRock
version?" — the question the Hub image could not answer at all.

## What the new Dockerfile does differently

- **`ARG AMDGPU_TARGETS=gfx906`** — upstream's arch list
  (`gfx908;gfx90a;gfx942;gfx1030;gfx1100;...`) omits gfx906 entirely, so it must
  be passed explicitly.
- **The fork's exact cmake recipe** from `FEATURES.md`: `GGML_HIP=ON`,
  `GGML_HIP_GRAPHS=ON`, `GGML_HIP_RCCL=ON`, `HIP_COMPILER=clang`,
  `-O3 -Wno-unused-command-line-argument`. `HIPCXX`/`HIP_PATH` are set the way
  upstream's `.devops/rocm.Dockerfile` does, because a bare `-DGGML_HIP=ON` does
  not always select the right clang.
- **`-complete` base.** The plain `rocm/dev-ubuntu` image has no rocBLAS/RCCL;
  `GGML_HIP_RCCL=ON` and the repack kernels need them.
- **A build-time guard** that fails the build if no `ggml-hip` backend was
  produced, instead of shipping a silent CPU-only image.
- **`/app/versions.txt`** recording the git commit, git describe, ROCm version,
  gfx906 target and build timestamp — so "which llama.cpp is this?" is always
  answerable, which it was not with the Hub image.
- **The three recommended env vars baked in** (`HSA_FORCE_FINE_GRAIN_PCIE`,
  `GPU_MAX_HW_QUEUES`, `LLAMA_ENABLE_MTP_OPT`) so they apply to every container
  built on this image. Note `HSA_OVERRIDE_GFX_VERSION` is deliberately *not*
  set — a build targeting gfx906 already knows the part.
- **No UI stage.** Upstream's Dockerfile builds `tools/ui` for llama-server's
  built-in web UI; you get the UI from llama-swap, so that build time is wasted.

## What I could not verify

I have no ROCm toolchain here, so **neither of these Dockerfiles has been built.**
The cmake flags are transcribed from your fork's `FEATURES.md` and the structure
from upstream's `.devops/rocm.Dockerfile`, but the first real signal is step 3
above. If `ROCM_VERSION` is wrong you will see it there rather than three
rebuilds later.
