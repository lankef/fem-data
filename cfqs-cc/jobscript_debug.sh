#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err
#SBATCH --gres=gpu:l40s:1

# Instrumented copy of jobscript.sh.  Original is unchanged.
# Optional flags (sbatch --export=ALL,DEBUG_DROP_JBCD=1 or export below):
#   DEBUG_DUMMY_OBJ DEBUG_SKIP_FEM DEBUG_NO_NL_CONS
#   DEBUG_DROP_JBCD DEBUG_FIX_THETAS DEBUG_INTERIOR_THETA

mkdir -p logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_JOB_NUM_NODES"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Start time:     $(date)"
echo "Debug flags:    DEBUG_DUMMY_OBJ=${DEBUG_DUMMY_OBJ:-0} DEBUG_SKIP_FEM=${DEBUG_SKIP_FEM:-0} DEBUG_NO_NL_CONS=${DEBUG_NO_NL_CONS:-0} DEBUG_DROP_JBCD=${DEBUG_DROP_JBCD:-0} DEBUG_FIX_THETAS=${DEBUG_FIX_THETAS:-0} DEBUG_INTERIOR_THETA=${DEBUG_INTERIOR_THETA:-0}"
conda activate desc
python -u ./optimize_beams_debug.py

echo "End time:       $(date)"
