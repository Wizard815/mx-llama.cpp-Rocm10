#!/usr/bin/env bash
# Sweep the repacked Q8_0 prefill GEMM tiles on gfx906 (ggml/src/ggml-cuda/q8_repack):
#   dense  mmq_gemm_repacked<false,TN,NRL,Q8_0>      (>= 128 tokens; attention/shared projections)
#   MoE    mmq_gemm_repacked_w32<true,TN,NRL,Q8_0>   (expert GEMM, 32-wide path)
# Together ~30-35% of prefill GPU time for Tiel-Coder-35B-A3B Q8 on 2x MI50.
#
# These kernels only run on weights repacked at model load, which test-backend-ops
# never creates, so this sweep uses the real model: llama-perplexity on a fixed text
# as the correctness check (must match the first candidate's PPL) and llama-bench
# for prefill speed.
#
# Candidate = "dense_TN dense_NRL moe_TN moe_NRL moe_w32_factor"
#   TN  = columns per thread (tile width = 64*TN dense, 32*TN MoE)
#   NRL = row lanes per block (64/NRL rows per thread; 64/NRL must be a multiple of 4)
#   factor: MoE uses the 32-wide tile while n_assign < factor*128*n_expert
#
#   MODEL=/app/models/.../Tiel-Coder-...-noBF16.gguf scripts/gfx906-q8-gemm-sweep.sh
#   MODEL=... scripts/gfx906-q8-gemm-sweep.sh "2 4 1 8 2" "1 8 1 8 2"
#
# Env: MODEL (required), BUILD_DIR (default build), GPUS (default 0,1),
#      PPL_TOL (relative PPL tolerance vs the first candidate, default 0.001).
set -uo pipefail

cd "$(dirname "$0")/.."
: "${MODEL:?set MODEL to the .gguf to benchmark}"
BUILD_DIR=${BUILD_DIR:-build}
GPUS=${GPUS:-0,1}
PPL_TOL=${PPL_TOL:-0.001}
TUNE=ggml/src/ggml-cuda/q8_repack/repack-gcn-tune.h
BIN="$BUILD_DIR/bin"
COMMON=(-m "$MODEL" -ngl 99 -sm layer -lm dio)

if [ $# -gt 0 ]; then
    CANDIDATES=("$@")
else
    # First entry is the current default.
    CANDIDATES=(
        "2 4 1 8 2"
        "1 4 1 8 2"
        "4 4 1 8 2"
        "2 8 1 8 2"
        "1 8 1 8 2"
        "2 4 2 8 2"
        "2 4 1 4 2"
        "2 4 1 16 2"
        "2 4 2 4 2"
        "2 4 1 8 1"
        "2 4 1 8 4"
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
    cmake --build "$BUILD_DIR" --target llama-bench llama-perplexity -j"$(nproc)" > /dev/null 2>&1 || echo "    rebuild failed, rebuild $BUILD_DIR manually"
}
trap restore EXIT

ppl_ref=""
results=()
for c in "${CANDIDATES[@]}"; do
    read -r dtn dnrl mtn mnrl fac <<< "$c"
    label="dense TN=$dtn NRL=$dnrl | moe TN=$mtn NRL=$mnrl factor=$fac"
    echo "=== $label"
    printf '#pragma once\n#define GGML_RP_Q8_DENSE_TN %s\n#define GGML_RP_Q8_DENSE_NRL %s\n#define GGML_RP_Q8_MOE_W32_TN %s\n#define GGML_RP_Q8_MOE_W32_NRL %s\n#define GGML_RP_Q8_MOE_W32_FACTOR %s\n' \
        "$dtn" "$dnrl" "$mtn" "$mnrl" "$fac" > "$TUNE"

    if ! cmake --build "$BUILD_DIR" --target llama-bench llama-perplexity -j"$(nproc)" > /tmp/q8-sweep-build.log 2>&1; then
        echo "    compile failed (see /tmp/q8-sweep-build.log)"
        results+=("$label | compile failed")
        continue
    fi

    ppl=$(HIP_VISIBLE_DEVICES=$GPUS "$BIN/llama-perplexity" "${COMMON[@]}" -fa on -f "$PPL_FILE" \
            -c 512 --chunks 16 -b 512 -ub 512 2>&1 | grep -Eo 'Final estimate: PPL = [0-9.]+' | grep -Eo '[0-9.]+$')
    if [ -z "$ppl" ]; then
        echo "    perplexity run failed"
        results+=("$label | perplexity failed")
        continue
    fi
    check="ok"
    if [ -z "$ppl_ref" ]; then
        ppl_ref=$ppl
        check="reference"
    elif ! python3 -c "import sys; r,p,t=map(float,sys.argv[1:]); sys.exit(0 if abs(p-r)/r <= t else 1)" "$ppl_ref" "$ppl" "$PPL_TOL"; then
        check="MISMATCH"
    fi
    echo "    PPL $ppl ($check)"

    bench=$(HIP_VISIBLE_DEVICES=$GPUS "$BIN/llama-bench" "${COMMON[@]}" -fa 1 -b 2048 -ub 1024 \
            -p 2048,8192 -n 0 -r 3 -o csv 2>/dev/null)
    pp=$(echo "$bench" | python3 -c '
import csv, sys
rows = list(csv.DictReader(sys.stdin))
print(" ".join(f"pp{r[\"n_prompt\"]}={float(r[\"avg_ts\"]):.1f}" for r in rows))')
    echo "    $pp t/s"
    results+=("$label | PPL $ppl ($check) | $pp")
done

echo
echo "=== summary (higher t/s is faster; PPL must match the first candidate)"
printf '%s\n' "${results[@]}"
