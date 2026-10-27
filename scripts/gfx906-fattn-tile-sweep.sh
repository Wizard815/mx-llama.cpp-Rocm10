#!/usr/bin/env bash
# Sweep flash_attn_tile configs for GCN (gfx906) at head size 256 / 32 columns,
# the prefill kernel for Qwen3.5/3.6-35B-A3B-style models (GQA 8 -> ncols 4x8).
#
# For each candidate "nthreads occupancy nbatch_fa nbatch_K" it writes
# ggml/src/ggml-cuda/fattn-tile-gcn-tune.h, rebuilds test-backend-ops (only the
# flash-attention objects recompile), checks FLASH_ATTN_EXT hsk=256 against the
# CPU backend, then times the two prefill perf cases. The tune header is
# restored on exit. Configs that violate the kernel's static_asserts fail to
# compile and are reported as such.
#
#   cmake -S . -B build -DLLAMA_BUILD_TESTS=ON ...   # once, with the HIP flags
#   scripts/gfx906-fattn-tile-sweep.sh                # default candidate list
#   scripts/gfx906-fattn-tile-sweep.sh "256 2 64 64" "512 1 32 128"
#
# Env: BUILD_DIR (default build), BACKEND (default ROCm0), SKIP_CHECK=1.
set -uo pipefail

cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build}
BACKEND=${BACKEND:-ROCm0}
TUNE=ggml/src/ggml-cuda/fattn-tile-gcn-tune.h
TBO="$BUILD_DIR/bin/test-backend-ops"
PERF_FILTER='hsk=256,hsv=256,nh=2,nr23=\[8,1\],kv=(8192|32768),nb=1024'

if [ $# -gt 0 ]; then
    CANDIDATES=("$@")
else
    # First entry is the generic AMD table's value (the baseline).
    CANDIDATES=(
        "256 2 32 128"
        "256 2 32 64"
        "256 2 64 64"
        "256 2 64 32"
        "256 1 64 128"
        "256 1 128 64"
        "256 1 32 256"
        "128 2 32 128"
        "512 1 32 128"
        "512 1 64 64"
    )
fi

backup=$(mktemp)
cp "$TUNE" "$backup"
restore() { cp "$backup" "$TUNE"; rm -f "$backup"; }
trap restore EXIT

results=()
for c in "${CANDIDATES[@]}"; do
    read -r nt occ fa nk <<< "$c"
    label="nthreads=$nt occ=$occ nbatch_fa=$fa nbatch_K=$nk"
    echo "=== $label"
    printf '#pragma once\n#define GGML_FATTN_TILE_GCN_256_32 %s, %s, %s, %s\n' "$nt" "$occ" "$fa" "$nk" > "$TUNE"

    if ! cmake --build "$BUILD_DIR" --target test-backend-ops -j"$(nproc)" > "/tmp/fattn-sweep-build.log" 2>&1; then
        echo "    compile failed (see /tmp/fattn-sweep-build.log)"
        results+=("$label | compile failed")
        continue
    fi

    check="skipped"
    if [ "${SKIP_CHECK:-0}" != 1 ]; then
        if out=$("$TBO" -o FLASH_ATTN_EXT -b "$BACKEND" -p 'hsk=256' 2>&1); then
            check="ok"
        else
            check="FAILED"
            echo "$out" | grep -E 'FAIL' | head -5
        fi
        echo "    correctness: $check ($(echo "$out" | grep -Eo '[0-9]+/[0-9]+ tests passed' | tail -1))"
    fi

    perf=$("$TBO" perf -o FLASH_ATTN_EXT -b "$BACKEND" -p "$PERF_FILTER" 2>&1 | grep -E 'us/run')
    echo "$perf" | sed 's/^/    /'
    times=$(echo "$perf" | grep -Eo '[0-9.]+ us/run' | awk '{printf "%s ", $1}')
    results+=("$label | check=$check | us/run (kv 8192, 32768): $times")
done

echo
echo "=== summary (lower us/run is faster)"
printf '%s\n' "${results[@]}"
