#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --gres=gpu:l40s:1


mkdir -p logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1

echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_JOB_NUM_NODES"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Start time:     $(date)"
conda activate desc

# Baseline Taylor (simsopt J/dJ). Optional probes for the ~2% grad gap:
#   --force-kt-adjoint   # (1) adjoint uses Kᵀ instead of reused K
#   --drop-winkler-wa    # (2) drop grounded k_attachment*w_a Winkler term
#   --simsopt-free       # (3) also Taylor jax.grad(fem.objective)
# Pass extra flags via: sbatch --export=ALL,TAYLOR_FLAGS="--force-kt-adjoint" ...
python -u ./taylor.py ${TAYLOR_FLAGS:-}

echo "End time:       $(date)"
