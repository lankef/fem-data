#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --time=07:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --mem=50G
#SBATCH --array=0                       # 4 tasks

# Pick the script for this array index
SCRIPTS=(optimize_force.py)
SCRIPT=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh

export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1
conda activate desc

python -u ./$SCRIPT \
    --task_id   $TASK_ID \
    --num_tasks $NUM_TASKS