#!/usr/bin/env bash
# A/B the gfx906 flash-attention tile rows and K/V register prefetch (fattn-tile.cuh, head 256)
# on MI50 cards: correctness with test-backend-ops, kernel time per op, llama-bench, and a
# register spill report of the tile kernels as your compiler built them.
#
# "Before" is the same build with GGML_FA_TILE_GFX906=0 GGML_FA_KV_PREFETCH=0, which takes the
# dispatch and kernels the code had before the port. BASE_REF=<commit> also builds that commit
# (e.g. BASE_REF=2295f0c78^) and times it through the same test binary, for a check of that claim.
#
#   scripts/gfx906-fa-prefetch-ab.sh                         # GPU 0, kernels only
#   DEVICES="0 1" MODEL=/models/qwen.gguf scripts/gfx906-fa-prefetch-ab.sh
#
# Env:
#   BUILD_DIR   build of this checkout (default build-fa-ab); configured with tests if it has no
#               CMakeCache.txt, built for test-backend-ops and llama-bench unless SKIP_BUILD=1
#   CMAKE_ARGS  configure flags for a new build dir (default: the GFX906-ROCM10.md build + tests)
#   BACKEND     test-backend-ops backend (default ROCm0; HIP_VISIBLE_DEVICES picks the card)
#   DEVICES     HIP device indices, one run each (default 0); e.g. "0 1" for a 16GB and a 32GB card
#   MODEL       GGUF for llama-bench; without it llama-bench is skipped
#   DEPTHS      llama-bench depths (default 0,1024,4096,16384,32768)
#   CACHE_TYPES llama-bench K/V cache types (default "f16 q8_0")
#   REPS        llama-bench repetitions (default 3)
#   TEST_FILTER / PERF_FILTER  test-backend-ops -p regexes (defaults below)
#   QUICK=1     fewer perf cases (4096 and 32768 K/V rows) and depths 0,4096,32768
#   SKIP_TEST=1 / SKIP_PERF=1 / SKIP_SPILL=1 skip a stage
#   BASE_REF    also build and time this commit (git worktree in $BUILD_DIR-base)
#   OUT         results directory (default fa-ab-<date>)
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
BUILD_DIR=${BUILD_DIR:-build-fa-ab}
DEVICES=${DEVICES:-0}
MODEL=${MODEL:-}
DEPTHS=${DEPTHS:-0,1024,4096,16384,32768}
CACHE_TYPES=${CACHE_TYPES:-f16 q8_0}
REPS=${REPS:-3}
QUICK=${QUICK:-0}
BACKEND=${BACKEND:-ROCm0}
BASE_REF=${BASE_REF:-}
OUT=${OUT:-fa-ab-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
SUMMARY="$OUT/summary.md"

DEFAULT_CMAKE_ARGS="-GNinja -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
-DHIP_COMPILER=clang -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_HIP_FLAGS=-mllvm\ -amdgpu-sched-strategy=max-ilp \
-DLLAMA_BUILD_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
CMAKE_ARGS=${CMAKE_ARGS:-$DEFAULT_CMAKE_ARGS}

# name|env. "before" first: the perf table compares every other column with it.
CONFIGS=(
    "before|GGML_FA_TILE_GFX906=0 GGML_FA_KV_PREFETCH=0"
    "rows|GGML_FA_KV_PREFETCH=0"
    "prefetch|GGML_FA_TILE_GFX906=0"
    "rows+prefetch|"
    "nopad:before|GGML_CUDA_FA_GQA_PAD=0 GGML_FA_TILE_GFX906=0 GGML_FA_KV_PREFETCH=0"
    "nopad:rows+prefetch|GGML_CUDA_FA_GQA_PAD=0"
)
# llama-bench runs fewer configs, each one loads the model again
BENCH_CONFIGS=("before" "rows+prefetch" "nopad:rows+prefetch")

TEST_FILTER=${TEST_FILTER:-'hsk=256,hsv=256'}
if [ -n "${PERF_FILTER:-}" ]; then
    :
elif [ "$QUICK" = 1 ]; then
    PERF_FILTER='hsk=256,hsv=256,nh=2,nr23=\[(2|6),1\],kv=(4096|32768),'
    DEPTHS=${DEPTHS_QUICK:-0,4096,32768}
else
    PERF_FILTER='hsk=256,hsv=256,nh=2,nr23=\[(2|6),1\],kv=[0-9]+,nb=[0-9]+,mask=1,sinks=0,max_bias=0'
fi

log() { echo "== $*" | tee -a "$OUT/run.log"; }

cfg_env() { local c; for c in "${CONFIGS[@]}"; do [ "${c%%|*}" = "$1" ] && { echo "${c#*|}"; return; }; done; }

build() { # dir src
    local dir=$1 src=$2
    if [ ! -f "$dir/CMakeCache.txt" ]; then
        log "configure $dir"
        # shellcheck disable=SC2086
        (cd "$src" && export HIPCXX="${HIPCXX:-$(hipconfig -l)/clang}" HIP_PATH="${HIP_PATH:-$(hipconfig -R)}" &&
            eval cmake -S . -B "$dir" $CMAKE_ARGS) > "$OUT/configure-$(basename "$dir").log" 2>&1 \
            || { echo "configure failed, see $OUT/configure-$(basename "$dir").log"; exit 1; }
    fi
    log "build $dir"
    cmake --build "$dir" --target test-backend-ops llama-bench -j"$(nproc)" > "$OUT/build-$(basename "$dir").log" 2>&1 \
        || { echo "build failed, see $OUT/build-$(basename "$dir").log"; exit 1; }
}

case "$BUILD_DIR" in /*) ;; *) BUILD_DIR="$ROOT/$BUILD_DIR" ;; esac
TBO="$BUILD_DIR/bin/test-backend-ops"
BENCH="$BUILD_DIR/bin/llama-bench"

if [ "${SKIP_BUILD:-0}" != 1 ]; then
    build "$BUILD_DIR" "$ROOT"
    if [ -n "$BASE_REF" ]; then
        BASE_SRC="$BUILD_DIR-base-src"
        if [ ! -d "$BASE_SRC" ]; then
            git worktree add --detach "$BASE_SRC" "$BASE_REF" > /dev/null || exit 1
        else
            git -C "$BASE_SRC" checkout --detach "$BASE_REF" > /dev/null 2>&1 || exit 1
        fi
        build "$BUILD_DIR-base" "$BASE_SRC"
    fi
fi
[ -x "$TBO" ] || { echo "missing $TBO (configure with -DLLAMA_BUILD_TESTS=ON)"; exit 1; }
if [ -n "$BASE_REF" ]; then
    # the new test binary carries the new cases; LD_LIBRARY_PATH beats its RUNPATH, so it loads the base ggml
    CONFIGS+=("base:$BASE_REF|LD_LIBRARY_PATH=$BUILD_DIR-base/bin")
fi

{
    echo "# gfx906 FA tile rows + K/V prefetch A/B"
    echo
    echo "- commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
    echo "- build: $BUILD_DIR"
    echo "- date: $(date -Is)"
    command -v rocm-smi > /dev/null && echo "- GPUs: $(rocm-smi --showproductname 2>/dev/null | grep -Eo 'Card (series|SKU).*' | sort -u | tr '\n' ' ')"
    echo
    echo "Configs (env on top of the defaults):"
    echo
    for c in "${CONFIGS[@]}"; do echo "- \`${c%%|*}\`: \`${c#*|}\`"; done
    echo
} > "$SUMMARY"

# --- register spills of the head-256 tile kernels, as this compiler built them ---------------
if [ "${SKIP_SPILL:-0}" != 1 ]; then
    log "spill report"
    {
        echo "## Tile kernels, head 256, 2 Q heads per block"
        echo
        python3 - "$BUILD_DIR/compile_commands.json" <<'EOF' 2>&1 || echo "(spill report failed; configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)"
import json, re, shlex, subprocess, sys
entries = [e for e in json.load(open(sys.argv[1])) if e['file'].endswith('fattn-tile-instance-dkq256-dv256.cu')]
if not entries:
    sys.exit('fattn-tile-instance-dkq256-dv256.cu not in compile_commands.json')
e = entries[0]
args = e['arguments'] if 'arguments' in e else shlex.split(e['command'])
if '-o' in args:
    i = args.index('-o'); args[i + 1] = '/dev/null'
args = [a for a in args if not a.startswith(('-MD', '-MF', '-MT')) or a == '-MD']
args = [a for a in args if a != '-MD']
args += ['-Rpass-analysis=kernel-resource-usage', '-Wno-everything']
r = subprocess.run(args, cwd=e['directory'], capture_output=True, text=True)
rows, cur = [], None
for l in r.stderr.splitlines():
    m = re.search(r'Function Name: (\S+)', l)
    if m:
        t = re.search(r'flash_attn_tileILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELb([01])ELb([01])ELb([01])E', m.group(1))
        cur = {'t': tuple(int(x) for x in t.groups())} if t else None
        if cur: rows.append(cur)
        continue
    m = cur is not None and re.search(r'remark:\s+(VGPRs|VGPRs Spill|ScratchSize \[bytes/lane\]|Occupancy \[waves/SIMD\]): (\d+)', l)
    if m: cur[m.group(1)] = int(m.group(2))
if not rows:
    sys.exit('no resource remarks (compiler output follows)\n' + r.stderr[-2000:])
print('| columns | softcap | gfx906 rows | prefetch | VGPRs | VGPR spill | scratch B/lane | waves/SIMD |')
print('|---|---|---|---|---|---|---|---|')
for x in sorted(rows, key=lambda x: (x['t'][4], x['t'][2]*x['t'][3], x['t'][5], x['t'][6])):
    d, dv, c1, c2, sc, g, p = x['t']
    if c2 != 2: continue
    print(f"| {c1*c2} | {sc} | {g} | {p} | {x.get('VGPRs')} | {x.get('VGPRs Spill')} | {x.get('ScratchSize [bytes/lane]')} | {x.get('Occupancy [waves/SIMD]')} |")
EOF
        echo
    } >> "$SUMMARY"
fi

perf_table() { # file with "config<TAB>params<TAB>us"; prints a markdown table relative to "before"
    awk -F'\t' -v order="$1" '
        BEGIN { n = split(order, cols, " ") }
        { t[$2, $1] = $3; if (!($2 in seen)) { seen[$2] = 1; keys[++nk] = $2 } }
        END {
            printf "| case |"; for (i = 1; i <= n; i++) printf " %s |", cols[i]; print ""
            printf "|---|";    for (i = 1; i <= n; i++) printf "---|";           print ""
            for (k = 1; k <= nk; k++) {
                p = keys[k]; c = p
                gsub(/hsk=256,hsv=256,|,mask=1|,sinks=0|,max_bias=0\.0+|,logit_softcap=0\.0+|,prec=f32|,permute=\[0,1,2,3\]|,kv_view=1|,v_is_view_of_k=0|,n_kv_max=0|,q_head_padding=0|,mask_heads=1/, "", c)
                sub(/,type_V=[a-z0-9_]+/, "", c)
                printf "| %s |", c
                b = t[p, cols[1]]
                for (i = 1; i <= n; i++) {
                    v = t[p, cols[i]]
                    if (v == "") { printf " - |"; continue }
                    if (i > 1 && b > 0) printf " %.1f (%+.1f%%) |", v, 100.0*(v - b)/b
                    else printf " %.1f |", v
                }
                print ""
            }
        }' "$2"
}

for dev in $DEVICES; do
    export HIP_VISIBLE_DEVICES=$dev
    name=$(rocm-smi -d "$dev" --showproductname 2>/dev/null | grep -Eo 'Card (series|SKU):.*' | head -1 | sed 's/.*: *//')
    vram=$(rocm-smi -d "$dev" --showmeminfo vram 2>/dev/null | grep -Eo 'Total Memory \(B\): *[0-9]+' | grep -Eo '[0-9]+$')
    [ -n "$vram" ] && vram="$((vram / 1024 / 1024 / 1024)) GiB"
    echo "## GPU $dev ${name:-} ${vram:-}" >> "$SUMMARY"
    echo >> "$SUMMARY"

    # --- correctness: every config against the CPU backend -------------------------------------
    if [ "${SKIP_TEST:-0}" != 1 ]; then
        echo "### test-backend-ops FLASH_ATTN_EXT ($TEST_FILTER)" >> "$SUMMARY"
        echo >> "$SUMMARY"
        echo "| config | result |" >> "$SUMMARY"
        echo "|---|---|" >> "$SUMMARY"
        for c in "${CONFIGS[@]}"; do
            cname=${c%%|*}; cenv=${c#*|}
            log "GPU $dev test $cname"
            f="$OUT/test-gpu$dev-${cname//[:+]/_}.log"
            env $cenv "$TBO" test -o FLASH_ATTN_EXT -b "$BACKEND" -p "$TEST_FILTER" > "$f" 2>&1
            rc=$?
            res=$(grep -Eo '[0-9]+/[0-9]+ tests passed' "$f" | tail -1)
            nfail=$(grep -c '\[FLASH_ATTN_EXT\] .*FAIL\|  FAIL$' "$f")
            [ $rc -eq 0 ] && st="ok" || st="**FAILED** (exit $rc, $nfail failing cases, see $(basename "$f"))"
            echo "| $cname | $st ${res:-} |" >> "$SUMMARY"
            [ $rc -eq 0 ] || grep -E 'FAIL' "$f" | head -5 | sed 's/^/    /'
        done
        echo >> "$SUMMARY"
    fi

    # --- kernel time per op -------------------------------------------------------------------
    if [ "${SKIP_PERF:-0}" != 1 ]; then
        tsv="$OUT/perf-gpu$dev.tsv"; : > "$tsv"
        order=""
        for c in "${CONFIGS[@]}"; do
            cname=${c%%|*}; cenv=${c#*|}
            order="$order $cname"
            log "GPU $dev perf $cname"
            f="$OUT/perf-gpu$dev-${cname//[:+]/_}.log"
            env $cenv "$TBO" perf -o FLASH_ATTN_EXT -b "$BACKEND" -p "$PERF_FILTER" > "$f" 2>&1
            sed -nE 's/^ *FLASH_ATTN_EXT\(([^)]*)\): .* ([0-9.]+) us\/run.*/\1\t\2/p' "$f" | \
                awk -F'\t' -v c="$cname" '{ print c "\t" $1 "\t" $2 }' >> "$tsv"
        done
        {
            echo "### Kernel time, us/run (change vs before; lower is better)"
            echo
            echo "nh = K/V heads, nr23 = [GQA ratio, sequences], kv = K/V rows, nb = tokens."
            echo "GQA 6 at 1..8 tokens only reaches the gfx906 rows and the prefetch with GGML_CUDA_FA_GQA_PAD=0 (nopad: columns)."
            echo
            perf_table "$order" "$tsv"
            echo
        } >> "$SUMMARY"
    fi

    # --- end to end ---------------------------------------------------------------------------
    if [ -n "$MODEL" ] && [ -x "$BENCH" ]; then
        echo "### llama-bench $(basename "$MODEL")" >> "$SUMMARY"
        echo >> "$SUMMARY"
        for ct in $CACHE_TYPES; do
            for cname in "${BENCH_CONFIGS[@]}"; do
                cenv=$(cfg_env "$cname")
                log "GPU $dev llama-bench $cname cache $ct"
                f="$OUT/bench-gpu$dev-$ct-${cname//[:+]/_}.md"
                env $cenv "$BENCH" -m "$MODEL" -ngl 999 -fa 1 -ctk "$ct" -ctv "$ct" \
                    -p 1,2,4,8 -n 32 -d "$DEPTHS" -r "$REPS" -o md > "$f" 2> "${f%.md}.err"
                { echo "#### $cname, cache $ct"; echo; grep '^|' "$f" || echo "(failed, see $(basename "${f%.md}.err"))"; echo; } >> "$SUMMARY"
            done
        done
    fi
done

log "done: $SUMMARY"
