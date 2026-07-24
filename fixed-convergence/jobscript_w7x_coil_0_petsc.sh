#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=w7x_coil_0_petsc_logs/slurm_%A_%a.out
#SBATCH --error=w7x_coil_0_petsc_logs/slurm_%A_%a.err

# PETSc solver is CPU-based; no GPU requested.
# JAX body-force computations will fall back to CPU as well.

mkdir -p w7x_coil_0_petsc_logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1

echo "Job ID:         $SLURM_JOB_ID"
echo "Task:           $TASK_ID / $NUM_TASKS"
echo "Nodes:          $SLURM_JOB_NUM_NODES"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Start time:     $(date)"
conda activate desc

python -u ./w7x_coil_0_petsc.py \
    --task_id   $TASK_ID \
    --num_tasks $NUM_TASKS

echo "End time:       $(date)"
