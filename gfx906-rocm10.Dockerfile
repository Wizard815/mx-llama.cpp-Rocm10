# Base: any image with a gfx906 ROCm at /opt/rocm. Default is the Docker Hub
# image this setup started from; set GFX906_BASE_IMAGE in .env to build
# against your own TheRock-gfx906 SDK instead, e.g. rocm10-gfx906:10.2-sdk
# (see TheRock-gfx906/dockerfiles/gfx906-sdk.Dockerfile).
ARG BASE_IMAGE=zenth815/mx-llama-rocm10-gfx906:latest
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git libssl-dev libgomp1 curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build -GNinja \
      -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=ON \
      -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release -DHIP_COMPILER=clang \
      -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument" \
      -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
      -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --config Release -j"$(nproc)" \
    && mkdir -p /app \
    && rm -f /app/libggml*.so* /app/libllama*.so* /app/libmtmd*.so* \
    && cp build/bin/llama-server build/bin/llama-cli build/bin/llama-bench /app/ \
    && cp -P build/bin/*.so* /app/ \
    && ls -l /app/llama-server /app/libggml-hip.so

# The base image ships its own (older) libggml-*.so / libllama*.so in /app,
# next to the binaries. Copying only the executables let llama-server load
# the base image's libggml-hip.so against this build's libggml-base.so, so
# GPU fixes in this tree never ran. The rm above drops every old copy first.
