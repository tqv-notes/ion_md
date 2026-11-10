#!/bin/bash
# Source this file on the HPC to load the toolchain:
#     source scripts/hpc_env.sh
#
# Module names come from hpc_module_avail.txt. Notes:
#  * GCC 13.1 is the host compiler. CUDA 12.2 and older do NOT officially
#    support GCC 13 as nvcc host compiler; CUDA 12.4+ does, so cuda12.8 is used.
#  * The driver (575.57, CUDA 12.9) supports every toolkit up to 12.9.
#  * Armadillo / Boost are NOT needed any more.
#
# Override any choice by exporting the variable before sourcing, e.g.
#     IONMD_CUDA_MODULE=cuda12.4/toolkit/12.4.1 source scripts/hpc_env.sh

: "${IONMD_GCC_MODULE:=gcc/13.1.0}"
: "${IONMD_CMAKE_MODULE:=cmake/3.27.7-gcc-13.1.0}"
: "${IONMD_CUDA_MODULE:=cuda12.8/toolkit/12.8.1}"
: "${IONMD_PYTHON_MODULES:=python/3.11.6-gcc-13.1.0 py-numpy/1.26.1-gcc-13.1.0 py-matplotlib/3.8.1-gcc-13.1.0}"

# Batch jobs run non-interactive shells, where the `module` command may not
# be defined yet: initialise Environment Modules / Lmod if needed.
if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
             /usr/share/Modules/init/bash "${MODULESHOME:-/nonexistent}/init/bash"; do
        if [ -f "$f" ]; then
            # shellcheck disable=SC1090
            source "$f"
            break
        fi
    done
fi

_ionmd_load() {
    if module load "$1" 2>/dev/null; then
        echo "  loaded $1"
    else
        echo "  WARNING: could not load module $1" >&2
    fi
}

echo "Loading IonMD build environment..."
_ionmd_load "$IONMD_GCC_MODULE"
_ionmd_load "$IONMD_CMAKE_MODULE"
if [ "${IONMD_NO_CUDA:-0}" != "1" ]; then
    _ionmd_load "$IONMD_CUDA_MODULE"
fi
for m in $IONMD_PYTHON_MODULES; do
    _ionmd_load "$m"
done

# Make sure CMake/nvcc use the GCC we just loaded.
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
if command -v nvcc >/dev/null 2>&1; then
    export CUDAHOSTCXX="$CXX"
fi

# OpenMP defaults: one thread per allocated core (PBS: NCPUS, SLURM:
# SLURM_CPUS_PER_TASK, otherwise 8 unless already set), threads pinned to cores.
if [ -n "${NCPUS:-}" ]; then
    export OMP_NUM_THREADS="$NCPUS"
elif [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
elif [ -z "${OMP_NUM_THREADS:-}" ]; then
    export OMP_NUM_THREADS=8
fi
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "  gcc:   $($CXX --version | head -1)"
echo "  cmake: $(cmake --version | head -1)"
if command -v nvcc >/dev/null 2>&1; then
    echo "  nvcc:  $(nvcc --version | tail -1)"
else
    echo "  nvcc:  not available (CPU-only build)"
fi
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
