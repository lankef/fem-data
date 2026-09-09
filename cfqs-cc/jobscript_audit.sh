#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err

# CPU-only no-FEM audit.  Optional: DEBUG_DROP_JBCD=1 DEBUG_INTERIOR_THETA=1

mkdir -p logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_JOB_NUM_NODES"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Start time:     $(date)"
echo "Debug flags:    DEBUG_DROP_JBCD=${DEBUG_DROP_JBCD:-0} DEBUG_INTERIOR_THETA=${DEBUG_INTERIOR_THETA:-0}"
conda activate desc
python -u ./audit_init_constraints.py

echo "End time:       $(date)"
