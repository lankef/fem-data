#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=w7x_coil_0_logs/slurm_%A_%a.out   # stdout log (%A=job id, %a=array index)
#SBATCH --error=w7x_coil_0_logs/slurm_%A_%a.err    # stderr log
###SBATCH --gres=gpu:1             # one GPU per task

# Create w7x_coil_0_logs directory if it doesn't exist
mkdir -p w7x_coil_0_logs

# Load modules and activate conda environment
module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
# Debugging petsc error
export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1
# Print job information
echo "Job ID:         $SLURM_JOB_ID"
echo "Task:           $TASK_ID / $NUM_TASKS"
echo "Nodes:          $SLURM_JOB_NUM_NODES"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Start time:     $(date)"
conda activate desc
nvidia-smi
# Run the single-coil benchmark for this task's slice of N_QUADPOINTS_LIST.
# SLURM_ARRAY_TASK_ID  → --task_id   (0-based index of this job)
# SLURM_ARRAY_TASK_COUNT → --num_tasks (total number of array jobs = 11)
python -u ./w7x_coil_0.py \
    --task_id   $TASK_ID \
    --num_tasks $NUM_TASKS

echo "End time:       $(date)"

