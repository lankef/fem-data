#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# PETSc solver is CPU-based; no GPU requested.
# JAX body-force computations will fall back to CPU as well.

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

python -u ./beam_dolfinx.py 

echo "End time:       $(date)"
