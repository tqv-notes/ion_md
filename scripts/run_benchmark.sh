#!/bin/bash
# Performance benchmark: time per step vs number of ions for the OpenMP CPU
# backend (1, 2, 4, 8 threads) and the CUDA backend (1 and 2 GPUs).
#
#   scripts/run_benchmark.sh [OUT_DIR]
#
# Environment:
#   BUILD_DIR      (default: build)
#   SIZES          ion numbers   (default: "100 300 1000 3000 10000 30000")
#   THREAD_LIST    CPU threads   (default: "1 2 4 8")
#   GPU_LIST       GPU counts    (default: "1 2", ignored without CUDA)
#   MAX_CPU_PAIRS  skip CPU runs with N^2/threads above this (default 2e8)
#   TARGET_SECONDS approximate wall time per measurement (default 5)
#
# Writes OUT_DIR/benchmark.csv and OUT_DIR/benchmark.png
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
OUT="${1:-$ROOT/results/benchmark_$(date +%Y%m%d_%H%M%S)}"
SIZES="${SIZES:-100 300 1000 3000 10000 30000}"
THREAD_LIST="${THREAD_LIST:-1 2 4 8}"
GPU_LIST="${GPU_LIST:-1 2}"
MAX_CPU_PAIRS="${MAX_CPU_PAIRS:-200000000}"
TARGET_SECONDS="${TARGET_SECONDS:-5}"
IONMD="$BUILD_DIR/app/ionmd"
CFG="$ROOT/tests/configs/benchmark.json"
PYTHON="${PYTHON:-python3}"

export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

mkdir -p "$OUT/runs"
CSV="$OUT/benchmark.csv"
echo "n_ions,backend,threads,gpus,steps,production_time,steps_per_second,time_per_step,pair_rate" > "$CSV"

HAVE_CUDA=0
if "$IONMD" --info | grep -q "^  [1-9][0-9]* CUDA device"; then HAVE_CUDA=1; fi
"$IONMD" --info

# Steps such that each measurement takes roughly TARGET_SECONDS, assuming
# ~2e8 pair interactions / s / thread (adjusted by the first measurement).
steps_for() {  # N workers rate
    "$PYTHON" -c "
import sys
N, w, rate = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
s = int($TARGET_SECONDS * rate * w / max(N * N, 1))
print(max(20, min(s, 20000)))" "$@"
}

measure() {  # N backend threads gpus steps
    local N=$1 be=$2 th=$3 g=$4 st=$5
    local dir="$OUT/runs/${be}_N${N}_t${th}_g${g}"
    OMP_NUM_THREADS=$th "$IONMD" "$CFG" -N "$N" -b "$be" -t "$th" -g "$g" -n "$st" \
        -o "$dir" -q > "$dir.log" 2>&1 || { echo "  failed: $dir.log"; return; }
    "$PYTHON" - "$dir/run_info.json" "$N" "$be" "$th" "$g" "$st" >> "$CSV" <<'EOF'
import json, sys
info = json.load(open(sys.argv[1]))
N, be, th, g, st = sys.argv[2:7]
N = int(N)
tp = info["production_time"]
sps = info["steps_per_second"]
print("%d,%s,%s,%s,%s,%.6f,%.3f,%.6e,%.4e" % (N, be, th, g, st, tp, sps, 1.0 / sps,
                                               sps * N * (N - 1)))
EOF
    tail -1 "$CSV"
}

for N in $SIZES; do
    for th in $THREAD_LIST; do
        if [ $(( N * N / th )) -gt "$MAX_CPU_PAIRS" ] && [ "$th" -lt 8 ]; then
            echo "  skip cpu N=$N threads=$th (too slow)"; continue
        fi
        st=$(steps_for "$N" "$th" 2e8)
        measure "$N" cpu "$th" 0 "$st"
    done
    if [ "$HAVE_CUDA" = 1 ]; then
        for g in $GPU_LIST; do
            st=$(steps_for "$N" 1 2e10)
            measure "$N" cuda "${THREAD_LIST##* }" "$g" "$st"
        done
    fi
done

"$PYTHON" "$ROOT/python/plot_benchmark.py" "$CSV"
echo "Results: $CSV"
