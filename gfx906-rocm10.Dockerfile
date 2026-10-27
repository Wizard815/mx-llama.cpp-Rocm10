FROM zenth815/mx-llama-rocm10-gfx906:latest

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
    && rm -f /app/libggml*.so* /app/libllama*.so* /app/libmtmd*.so* \
    && cp build/bin/llama-server build/bin/llama-cli build/bin/llama-bench /app/ \
    && cp -P build/bin/*.so* /app/ \
    && ls -l /app/llama-server /app/libggml-hip.so

# The base image ships its own (older) libggml-*.so / libllama*.so in /app,
# next to the binaries. Copying only the executables let llama-server load
# the base image's libggml-hip.so against this build's libggml-base.so, so
# GPU fixes in this tree never ran. The rm above drops every old copy first.
