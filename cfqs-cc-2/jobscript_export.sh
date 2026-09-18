#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# Array job: each task runs run_export.py against one of the two cases
# (init_auglag, fin_auglag), reading Jstress_init_auglag.json /
# Jstress_fin_auglag.json and writing init_auglag_dolfinx/, fin_auglag_dolfinx/.
#   sbatch jobscript_export.sh

mkdir -p logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1

echo "Job ID:         $SLURM_JOB_ID"
echo "Array task ID:  $SLURM_ARRAY_TASK_ID"
echo "Nodes:          $SLURM_JOB_NUM_NODES"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Start time:     $(date)"
conda activate desc

CASES=(init_auglag fin_auglag)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"

echo "Case:           $CASE"

python -u ./run_export.py "$CASE"

status=$?
echo "Exit code:      $status"
echo "End time:       $(date)"
exit $status
