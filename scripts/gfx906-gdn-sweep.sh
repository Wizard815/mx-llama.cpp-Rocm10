#!/usr/bin/env bash
# Sweep the chunked Gated DeltaNet kernel config for gfx906 at head size 128, the
# linear-attention prefill kernel of Qwen3.5/3.6-35B-A3B-style models
# (gated_delta_net_chunked_cuda<128,false,64,Tc>, ~11% of prefill GPU time on 2x MI50).
#
# For each candidate "warps_per_block cols_per_warp unroll" it writes
# ggml/src/ggml-cuda/gated_delta_net-gcn-tune.h, rebuilds test-backend-ops, checks
# GATED_DELTA_NET head_size=128 against the CPU backend, then times the two prefill
# perf cases. The header is restored (and the build redone) on exit.
#
#   scripts/gfx906-gdn-sweep.sh                    # default candidate list
#   scripts/gfx906-gdn-sweep.sh "2 4 1" "4 2 1"
#
# Env: BUILD_DIR (default build), BACKEND (default ROCm0), SKIP_CHECK=1.
set -uo pipefail

cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build}
BACKEND=${BACKEND:-ROCm0}
TUNE=ggml/src/ggml-cuda/gated_delta_net-gcn-tune.h
TBO="$BUILD_DIR/bin/test-backend-ops"
PERF_FILTER='head_count=16,head_size=128,n_seq_tokens=(1024|2048),n_seqs=1,v_repeat=2'

if [ $# -gt 0 ]; then
    CANDIDATES=("$@")
else
    # First entry is the previous default (2 warps, 2 columns, no unroll); the gfx906
    # default is now 4 2 1.
    # Warps per block is capped at 4 by the kernel's __launch_bounds__.
    CANDIDATES=(
        "2 2 0"
        "2 2 1"
        "2 1 0"
        "2 4 0"
        "2 4 1"
        "2 8 1"
        "4 1 0"
        "4 2 1"
        "4 4 1"
        "1 4 1"
        "1 8 1"
    )
fi

backup=$(mktemp)
cp "$TUNE" "$backup"
# Rebuild after restoring: binaries built in-tree load these libs (RUNPATH), so a
# leftover candidate build would otherwise end up in whatever runs from this tree.
restore() {
    cp "$backup" "$TUNE"; rm -f "$backup"
    echo "=== restoring default build"
    cmake --build "$BUILD_DIR" --target test-backend-ops -j"$(nproc)" > /dev/null 2>&1 || echo "    rebuild failed, rebuild $BUILD_DIR manually"
}
trap restore EXIT

results=()
for c in "${CANDIDATES[@]}"; do
    read -r nw tc un <<< "$c"
    label="warps=$nw cols=$tc unroll=$un"
    echo "=== $label"
    printf '#pragma once\n#define GGML_GDN_GCN_128_NWARPS %s\n#define GGML_GDN_GCN_128_TC %s\n#define GGML_GDN_GCN_128_UNROLL %s\n' \
        "$nw" "$tc" "$un" > "$TUNE"

    if ! cmake --build "$BUILD_DIR" --target test-backend-ops -j"$(nproc)" > /tmp/gdn-sweep-build.log 2>&1; then
        echo "    compile failed (see /tmp/gdn-sweep-build.log)"
        results+=("$label | compile failed")
        continue
    fi

    check="skipped"
    if [ "${SKIP_CHECK:-0}" != 1 ]; then
        if out=$("$TBO" -o GATED_DELTA_NET -b "$BACKEND" -p 'head_size=128' 2>&1); then
            check="ok"
        else
            check="FAILED"
            echo "$out" | grep -E 'FAIL' | head -5
        fi
        echo "    correctness: $check ($(echo "$out" | grep -Eo '[0-9]+/[0-9]+ tests passed' | tail -1))"
    fi

    perf=$("$TBO" perf -o GATED_DELTA_NET -b "$BACKEND" -p "$PERF_FILTER" 2>&1 | grep -E 'us/run')
    echo "$perf" | sed 's/^/    /'
    times=$(echo "$perf" | grep -Eo '[0-9.]+ us/run' | awk '{printf "%s ", $1}')
    results+=("$label | check=$check | us/run (1024, 2048 tokens): $times")
done

echo
echo "=== summary (lower us/run is faster)"
printf '%s\n' "${results[@]}"
