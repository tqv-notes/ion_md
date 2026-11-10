#!/bin/bash
# Batch job for one production simulation (8 CPU cores, 2 GPUs, 30 GB).
# Results go to results/<config name>_<job id>/ (with figures).
#
# Submit from the repository root. CONFIG is the path of a JSON configuration
# file (e.g. tests/configs/06_sympathetic_cooling.json); ARGS are optional
# extra ionmd options (see README), e.g. "-N 5000 -n 500000".
#
#   PBS:    qsub -v CONFIG=tests/configs/06_sympathetic_cooling.json scripts/job_run.sh
#           qsub -v CONFIG=my.json,ARGS="-N 5000 -n 500000" scripts/job_run.sh
#   SLURM:  sbatch scripts/job_run.sh my.json -N 5000 -n 500000
#   none:   bash scripts/job_run.sh my.json -N 5000 -n 500000
#
# (qsub cannot pass command-line arguments to a job script, hence -v.)
#
#PBS -N ionmd
#PBS -l select=1:ncpus=8:ngpus=2:mem=30gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -S /bin/bash
##PBS -q gpu
#
#SBATCH --job-name=ionmd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --output=ionmd-%j.out
##SBATCH --partition=gpu

set -euo pipefail
ROOT="${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"
[ -f scripts/hpc_env.sh ] || { echo "Submit this job from the IonMD repository root"; exit 1; }

# Configuration: first argument, or the CONFIG variable (qsub -v).
if [ $# -ge 1 ]; then
    CONFIG="$1"
    shift
else
    CONFIG="${CONFIG:-}"
    # shellcheck disable=SC2086
    set -- ${ARGS:-}
fi
if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
    echo "Configuration file '${CONFIG}' not found."
    echo "Usage: bash scripts/job_run.sh path/to/config.json [ionmd options]"
    echo "       qsub -v CONFIG=path/to/config.json[,ARGS=\"...\"] scripts/job_run.sh"
    exit 1
fi

source scripts/hpc_env.sh

JOB_ID="${PBS_JOBID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
JOB_ID="${JOB_ID%%.*}"   # PBS ids look like 1234.server
OUT="results/$(basename "$CONFIG" .json)_${JOB_ID}"

./build/app/ionmd "$CONFIG" -o "$OUT" -t "$OMP_NUM_THREADS" "$@"
python3 python/plot_run.py "$OUT"
