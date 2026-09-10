#!/bin/bash
#SBATCH --job-name=gfp-stage2
#SBATCH --output=logs/stage2_%A_%a.out
#SBATCH --array=0-74%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=24:00:00

# Stage 2 -- the ablation matrix: 15 conditions x 5 seeds = 75 tasks.
#
# %3 runs three tasks concurrently on the one GPU. The model is 4.7M parameters
# (~19 MB), so three copies fit trivially in 80 GB and the GPU would otherwise
# idle during clustering and acquisition.
#
# Run:  mkdir -p logs && sbatch slurm/stage2.sh
#
# Only run this after Stage 1 confirms determinism and settles whether KMeans
# stands in for spectral -- every condition except C0 assumes it does.

set -euo pipefail

module load cuda 2>/dev/null || true
source "${GFP_VENV:-.venv}/bin/activate" 2>/dev/null || true

# 3 concurrent tasks over 12 physical cores.
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONDITIONS=(C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14)
SEEDS=(0 1 2 3 4)

N_SEEDS=${#SEEDS[@]}
COND=${CONDITIONS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}

echo "task ${SLURM_ARRAY_TASK_ID}: condition ${COND} seed ${SEED}"

python3 -u run_experiment.py \
  --condition "${COND}" \
  --seed "${SEED}" \
  --test-peak cgre1338-06 \
  --num-workers 6 \
  --out "results/stage2.csv" \
  --tag stage2
