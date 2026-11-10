#!/bin/bash
# Configure, build and unit-test IonMD.
#
#   scripts/build.sh            # CUDA if nvcc is found, otherwise CPU only
#   scripts/build.sh --cpu      # force CPU-only (OpenMP) build
#   scripts/build.sh --cuda     # require CUDA
#
# Environment: BUILD_DIR (default: build), JOBS (default: 8),
#              CUDA_ARCH (default: 70 = Tesla V100), EXTRA_CMAKE_ARGS
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
JOBS="${JOBS:-8}"
CUDA_ARCH="${CUDA_ARCH:-70}"

MODE=auto
for arg in "$@"; do
    case "$arg" in
        --cpu) MODE=cpu ;;
        --cuda) MODE=cuda ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option $arg"; exit 1 ;;
    esac
done

USE_CUDA=OFF
if [ "$MODE" = cuda ]; then
    USE_CUDA=ON
elif [ "$MODE" = auto ] && command -v nvcc >/dev/null 2>&1; then
    USE_CUDA=ON
fi

echo "== Configuring (CUDA=$USE_CUDA) in $BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DIONMD_USE_OPENMP=ON \
    -DIONMD_USE_CUDA="$USE_CUDA" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DIONMD_NATIVE=ON \
    ${EXTRA_CMAKE_ARGS:-}

echo "== Building"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo "== Build information"
"$BUILD_DIR/app/ionmd" --info

echo "== Unit tests"
(cd "$BUILD_DIR/tests" && ./tests)
