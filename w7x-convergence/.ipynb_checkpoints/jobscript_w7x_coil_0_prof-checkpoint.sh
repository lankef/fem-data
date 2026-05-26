#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000
#SBATCH --time=00:20:00
#SBATCH --array=0-8:3             # 11 parallel tasks, one per entry in N_QUADPOINTS_LIST
#SBATCH --output=logs/slurm_%A_%a.out   # stdout log (%A=job id, %a=array index)
#SBATCH --error=logs/slurm_%A_%a.err    # stderr log
#SBATCH --gres=gpu:1             # one GPU per task

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules and activate conda environment
module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh

# Print job information
echo "Job ID:         $SLURM_JOB_ID"
echo "Array task:     $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Nodes:          $SLURM_JOB_NUM_NODES"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Start time:     $(date)"
conda activate desc
nvidia-smi
export PYTHONNOUSERSITE=1
# Run the single-coil benchmark for this task's slice of N_QUADPOINTS_LIST.
# SLURM_ARRAY_TASK_ID  → --task_id   (0-based index of this job)
# SLURM_ARRAY_TASK_COUNT → --num_tasks (total number of array jobs = 11)
python -u ./w7x_coil_0_prof.py \
    --task_id   $SLURM_ARRAY_TASK_ID \
    --num_tasks $SLURM_ARRAY_TASK_COUNT

echo "End time:       $(date)"
echo "Job completed!"
nsys --version
