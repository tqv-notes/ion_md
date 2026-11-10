#!/bin/bash
# Batch job: build IonMD (OpenMP + CUDA), run the unit tests, the physics test
# suite and the benchmark on one node with 8 CPU cores and 2 GPUs.
#
# Submit from the repository root:
#   PBS:    qsub scripts/job_tests.sh
#   SLURM:  sbatch scripts/job_tests.sh
#   none:   bash scripts/job_tests.sh      (e.g. in an interactive session on a GPU node)
#
# Each scheduler only reads its own directives (#PBS or #SBATCH) and treats
# the others as comments. Uncomment / adapt the queue (PBS -q) or partition
# (SLURM) line to your cluster if a GPU queue has to be named explicitly.
#
#PBS -N ionmd-tests
#PBS -l select=1:ncpus=8:ngpus=2:mem=30gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -S /bin/bash
##PBS -q gpu
#
#SBATCH --job-name=ionmd-tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=ionmd-tests-%j.out
##SBATCH --partition=gpu

set -uo pipefail
# Batch systems run a copy of this script, so locate the repository via the
# submission directory (submit from the repository root).
ROOT="${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT" || exit 1
[ -f scripts/hpc_env.sh ] || { echo "Submit this job from the IonMD repository root"; exit 1; }

source scripts/hpc_env.sh
nvidia-smi || true

STAMP="$(date +%Y%m%d_%H%M%S)"

# 1. build (+ unit tests, including the CPU/GPU agreement test)
bash scripts/build.sh --cuda || { echo "build failed"; exit 1; }

# 2. physics tests (GPU used automatically for N >= gpu_min_ions)
THREADS="$OMP_NUM_THREADS" BACKEND=auto SKIP_UNIT=1 \
    bash scripts/run_tests.sh "results/tests_$STAMP"
TEST_STATUS=$?

# 3. benchmark: OpenMP 1-8 threads vs 1-2 GPUs
bash scripts/run_benchmark.sh "results/benchmark_$STAMP"

echo "Test suite exit status: $TEST_STATUS"
exit $TEST_STATUS
