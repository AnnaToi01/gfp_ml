#!/bin/bash
#SBATCH --job-name=gfp-stage1
#SBATCH --output=logs/stage1_%A_%a.out
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=08:00:00

# Stage 1 -- validate the measurement apparatus before running the real matrix.
#
#   S1-Z  same condition, same seed, twice  -> is the pipeline deterministic?
#   S1-R  random acquisition, 5 seeds       -> the baseline everything is judged against
#   S1-S  spectral clustering, timed        -> is the eigensolver affordable at all?
#   S1-K  kmeans clustering, same seed      -> does it match spectral?
#
# Run:  mkdir -p logs && sbatch slurm/stage1.sh

set -euo pipefail

module load cuda 2>/dev/null || true
source "${GFP_VENV:-.venv}/bin/activate" 2>/dev/null || true

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUT=results/stage1.csv
POOL=cgre1338-06
COMMON="--test-peak ${POOL} --num-workers 6 --out ${OUT}"

case ${SLURM_ARRAY_TASK_ID} in
  0) python3 -u run_experiment.py --condition C1 --seed 0 ${COMMON} --tag S1-Z-a ;;
  1) python3 -u run_experiment.py --condition C1 --seed 0 ${COMMON} --tag S1-Z-b ;;

  2) python3 -u run_experiment.py --condition C2 --seed 0 ${COMMON} --tag S1-R ;;
  3) python3 -u run_experiment.py --condition C2 --seed 1 ${COMMON} --tag S1-R ;;
  4) python3 -u run_experiment.py --condition C2 --seed 2 ${COMMON} --tag S1-R ;;
  5) python3 -u run_experiment.py --condition C2 --seed 3 ${COMMON} --tag S1-R ;;
  6) python3 -u run_experiment.py --condition C2 --seed 4 ${COMMON} --tag S1-R ;;

  # C0 is the only condition using SpectralClustering. Give it its own task so a
  # slow eigensolver cannot stall the rest of the array.
  7) python3 -u run_experiment.py --condition C0 --seed 0 ${COMMON} --tag S1-S ;;
esac
