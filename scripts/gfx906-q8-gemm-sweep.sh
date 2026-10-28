#!/usr/bin/env bash
# Sweep the repacked Q8_0 prefill GEMM tiles on gfx906 (ggml/src/ggml-cuda/q8_repack):
#   dense  mmq_gemm_repacked<false,TN,NRL,Q8_0>      (>= 128 tokens; attention/shared projections)
#   MoE    mmq_gemm_repacked_w32<true,TN,NRL,Q8_0>   (expert GEMM, 32-wide path)
# Together ~30-35% of prefill GPU time for Tiel-Coder-35B-A3B Q8 on 2x MI50.
#
# Per candidate:
#   1. test-backend-ops MUL_MAT_REPACK: the weights go in the repacked buffer type, so
#      these are the kernels under test; checked against the CPU backend, then timed
#      at the model's shapes with -ub 1024 (n=1024).
#   2. with MODEL set: llama-perplexity on a fixed text (must match the first
#      candidate's PPL) and llama-bench prefill speed on the real model.
#
# Candidate = "dense_TN dense_NRL moe_TN moe_NRL moe_w32_factor [dense_MINB moe_MINB [moe_CW]]"
#   TN  = columns per thread (tile width = 64*TN dense, moe_CW*TN MoE)
#   NRL = row lanes per block (64/NRL rows per thread; 64/NRL must be a multiple of 4)
#   factor: MoE uses the 32-wide tile while n_assign < factor*128*n_expert
#           (inert for 256 experts below ~4096-token ubatches)
#   MINB: minimum waves per SIMD (__launch_bounds__ -> amdgpu_waves_per_eu), VGPR
#         budget 256/MINB (default 2 dense = 128, 3 MoE = 84).
#   moe_CW: MoE lanes per row of the block (default 32). CW 16 TN 2 or CW 8 TN 4
#         keep the 32-token tile (~ tokens per expert at -ub 1024 with 256
#         experts) but reuse each LDS weight read across 2 / 4 tokens.
#
#   scripts/gfx906-q8-gemm-sweep.sh                       # kernel stage only
#   MODEL=/app/models/.../Tiel-Coder-...-noBF16.gguf scripts/gfx906-q8-gemm-sweep.sh
#   MODEL=... scripts/gfx906-q8-gemm-sweep.sh "2 4 1 8 2" "1 8 1 8 2"
#
# Env: MODEL (optional), BUILD_DIR (default build), BACKEND (default ROCm0),
#      GPUS (default 0,1), SKIP_CHECK=1,
#      PPL_TOL (relative PPL tolerance vs the first candidate, default 0.001).
set -uo pipefail

cd "$(dirname "$0")/.."
MODEL=${MODEL:-}
BUILD_DIR=${BUILD_DIR:-build}
BACKEND=${BACKEND:-ROCm0}
GPUS=${GPUS:-0,1}
PPL_TOL=${PPL_TOL:-0.001}
TUNE=ggml/src/ggml-cuda/q8_repack/repack-gcn-tune.h
BIN="$BUILD_DIR/bin"
TBO="$BIN/test-backend-ops"
TARGETS=(test-backend-ops)
[ -n "$MODEL" ] && TARGETS+=(llama-bench llama-perplexity)
COMMON=(-m "$MODEL" -ngl 99 -sm layer -lm dio)

if [ $# -gt 0 ]; then
    CANDIDATES=("$@")
else
    # First and last entry are the current default (same-GPU drift check). TN/NRL
    # and MINB at CW 32 were swept already (all slower); these keep the 32-token
    # MoE tile and move work from lanes to tokens per thread.
    CANDIDATES=(
        "2 4 1 8 2 2 3 32"
        "2 4 2 8 2 2 3 16"
        "2 4 2 8 2 2 2 16"
        "2 4 2 16 2 2 3 16"
        "2 4 2 4 2 2 2 16"
        "2 4 4 8 2 2 2 8"
        "2 4 4 16 2 2 2 8"
        "2 4 1 8 2 2 3 32"
    )
fi

# Fixed perplexity text: the repo's own Markdown, so every candidate sees the same input.
PPL_FILE=$(mktemp)
find . -name '*.md' -not -path './build*' -not -path './.git/*' | sort | xargs cat > "$PPL_FILE" 2>/dev/null

backup=$(mktemp)
cp "$TUNE" "$backup"
# Rebuild after restoring: binaries built in-tree load these libs (RUNPATH), so a
# leftover candidate build would otherwise end up in whatever runs from this tree.
restore() {
    cp "$backup" "$TUNE"; rm -f "$backup" "$PPL_FILE"
    echo "=== restoring default build"
    cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j"$(nproc)" > /dev/null 2>&1 || echo "    rebuild failed, rebuild $BUILD_DIR manually"
}
trap restore EXIT

ppl_ref=""
results=()
for c in "${CANDIDATES[@]}"; do
    read -r dtn dnrl mtn mnrl fac dmb mmb mcw <<< "$c"
    dmb=${dmb:-2}; mmb=${mmb:-3}; mcw=${mcw:-32}
    label="dense TN=$dtn NRL=$dnrl MINB=$dmb | moe CW=$mcw TN=$mtn NRL=$mnrl MINB=$mmb factor=$fac"
    echo "=== $label"
    printf '#pragma once\n#define GGML_RP_Q8_DENSE_TN %s\n#define GGML_RP_Q8_DENSE_NRL %s\n#define GGML_RP_Q8_MOE_W32_TN %s\n#define GGML_RP_Q8_MOE_W32_NRL %s\n#define GGML_RP_Q8_MOE_W32_FACTOR %s\n#define GGML_RP_Q8_DENSE_MINB %s\n#define GGML_RP_Q8_MOE_W32_MINB %s\n#define GGML_RP_Q8_MOE_W32_CW %s\n' \
        "$dtn" "$dnrl" "$mtn" "$mnrl" "$fac" "$dmb" "$mmb" "$mcw" > "$TUNE"

    if ! cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j"$(nproc)" > /tmp/q8-sweep-build.log 2>&1; then
        echo "    compile failed (see /tmp/q8-sweep-build.log)"
        results+=("$label | compile failed")
        continue
    fi

    check="skipped"
    if [ "${SKIP_CHECK:-0}" != 1 ]; then
        if out=$("$TBO" -o MUL_MAT_REPACK -b "$BACKEND" 2>&1); then
            check="ok"
        else
            check="FAILED"
            echo "$out" | grep -E 'FAIL|not supported' | head -5
        fi
        echo "    kernel check: $check ($(echo "$out" | grep -Eo '[0-9]+/[0-9]+ tests passed' | tail -1))"
    fi
    perf=$("$TBO" perf -o MUL_MAT_REPACK -b "$BACKEND" -p ',n=1024,' 2>&1 | grep -E 'us/run')
    echo "$perf" | sed -E 's/^ *MUL_MAT_REPACK\(type_a=q8_0,/    /'
    # dense rows first, then MoE up/gate and down (order of the perf cases)
    times=$(echo "$perf" | grep -Eo '[0-9.]+ us/run' | awk '{printf "%s ", $1}')
    total=$(echo "$perf" | grep -Eo '[0-9.]+ us/run' | awk '{s+=$1} END {printf "%.0f", s}')
    label="$label | check=$check | us/run: $times| sum=$total"

    if [ -z "$MODEL" ]; then
        results+=("$label")
        continue
    fi

    ppl=$(HIP_VISIBLE_DEVICES=$GPUS "$BIN/llama-perplexity" "${COMMON[@]}" -fa on -f "$PPL_FILE" \
            -c 512 --chunks 16 -b 512 -ub 512 2>&1 | grep -Eo 'Final estimate: PPL = [0-9.]+' | grep -Eo '[0-9.]+$')
    if [ -z "$ppl" ]; then
        echo "    perplexity run failed"
        results+=("$label | perplexity failed")
        continue
    fi
    pcheck="ok"
    if [ -z "$ppl_ref" ]; then
        ppl_ref=$ppl
        pcheck="reference"
    elif ! python3 -c "import sys; r,p,t=map(float,sys.argv[1:]); sys.exit(0 if abs(p-r)/r <= t else 1)" "$ppl_ref" "$ppl" "$PPL_TOL"; then
        pcheck="MISMATCH"
    fi
    echo "    PPL $ppl ($pcheck)"

    bench=$(HIP_VISIBLE_DEVICES=$GPUS "$BIN/llama-bench" "${COMMON[@]}" -fa 1 -b 2048 -ub 1024 \
            -p 2048,8192 -n 0 -r 3 -o csv 2>/dev/null)
    pp=$(echo "$bench" | python3 -c '
import csv, sys
rows = list(csv.DictReader(sys.stdin))
print(" ".join(f"pp{r[\"n_prompt\"]}={float(r[\"avg_ts\"]):.1f}" for r in rows))')
    echo "    $pp t/s"
    results+=("$label | PPL $ppl ($pcheck) | $pp")
done

echo
echo "=== summary (lower us/run and higher t/s are faster; check=ok and PPL must match the first candidate)"
printf '%s\n' "${results[@]}"
