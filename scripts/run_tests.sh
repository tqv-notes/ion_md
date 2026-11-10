#!/bin/bash
# Run the IonMD physics test suite and analyse the results.
#
#   scripts/run_tests.sh [RESULTS_DIR]
#
# Environment:
#   BUILD_DIR   build directory containing app/ionmd   (default: build)
#   THREADS     OpenMP threads        (default: OMP_NUM_THREADS or 8)
#   BACKEND     cpu | cuda | auto     (default: auto = GPU for N >= 256 if built with CUDA)
#   GPUS        GPUs to use, 0 = all  (default: 0)
#   TESTS       space separated subset, e.g. "01 04 06" (default: all)
#   SKIP_UNIT   set to 1 to skip the C++ unit tests
#
# Results: RESULTS_DIR/<test>/ (raw output), RESULTS_DIR/figures/<test>/*.png,
#          RESULTS_DIR/summary.md, RESULTS_DIR/logs/*.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
RESULTS="${1:-$ROOT/results/tests_$(date +%Y%m%d_%H%M%S)}"
THREADS="${THREADS:-${OMP_NUM_THREADS:-8}}"
BACKEND="${BACKEND:-auto}"
GPUS="${GPUS:-0}"
IONMD="$BUILD_DIR/app/ionmd"
CONFIGS="$ROOT/tests/configs"
PYTHON="${PYTHON:-python3}"

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

if [ ! -x "$IONMD" ] && [ ! -x "$IONMD.exe" ]; then
    echo "ionmd executable not found in $BUILD_DIR/app - run scripts/build.sh first" >&2
    exit 1
fi

mkdir -p "$RESULTS/logs"
echo "IonMD test suite"
echo "  results: $RESULTS"
echo "  threads: $THREADS, backend: $BACKEND, gpus: $GPUS"
"$IONMD" --info | tee "$RESULTS/logs/build_info.log"
HAVE_CUDA=0
if "$IONMD" --info | grep -q "CUDA: yes"; then
    if "$IONMD" --info | grep -q "^  [1-9][0-9]* CUDA device"; then
        HAVE_CUDA=1
    fi
fi

FAILED=0

if [ "${SKIP_UNIT:-0}" != "1" ]; then
    echo "== C++ unit tests"
    if (cd "$BUILD_DIR/tests" && ./tests) > "$RESULTS/logs/unit_tests.log" 2>&1; then
        grep -E "All tests passed|test cases" "$RESULTS/logs/unit_tests.log"
    else
        echo "  UNIT TESTS FAILED (see logs/unit_tests.log)"
        tail -20 "$RESULTS/logs/unit_tests.log"
        FAILED=1
    fi
fi

selected() {
    [ -z "${TESTS:-}" ] && return 0
    for t in $TESTS; do
        case "$1" in "$t"*) return 0 ;; esac
    done
    return 1
}

run_case() {   # name config [extra args...]
    local name="$1"; shift
    local cfg="$1"; shift
    local log="$RESULTS/logs/$name.log"
    local t0=$(date +%s)
    printf "== %-34s" "$name"
    if "$IONMD" "$cfg" -o "$RESULTS/$name" -t "$THREADS" "$@" > "$log" 2>&1; then
        printf "done in %4ds  (%s)\n" "$(( $(date +%s) - t0 ))" \
            "$(grep -m1 'Coulomb backend' "$log" | sed 's/.*backend: //')"
    else
        printf "FAILED after %ds (see %s)\n" "$(( $(date +%s) - t0 ))" "$log"
        tail -5 "$log"
        FAILED=1
    fi
}

for cfg in "$CONFIGS"/0[1-8]_*.json; do
    name="$(basename "$cfg" .json)"
    selected "$name" || continue
    run_case "$name" "$cfg" -b "$BACKEND" -g "$GPUS"
done

if selected 09_backend_consistency; then
    run_case 09_backend_consistency_cpu "$CONFIGS/09_backend_consistency.json" -b cpu
    if [ "$HAVE_CUDA" = 1 ]; then
        run_case 09_backend_consistency_cuda "$CONFIGS/09_backend_consistency.json" -b cuda -g "$GPUS"
    else
        echo "== 09_backend_consistency_cuda      skipped (no CUDA build or no GPU)"
    fi
fi

echo "== Analysis"
if "$PYTHON" "$ROOT/python/analyze_tests.py" "$RESULTS"; then
    :
else
    FAILED=1
fi

echo
echo "Figures: $RESULTS/figures"
echo "Summary: $RESULTS/summary.md"
exit $FAILED
