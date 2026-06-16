#!/bin/bash -l
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --mem=20G
#SBATCH --array=0-2                       # 3 tasks: indices 0,1,2

# Pick the script for this array index
SCRIPTS=(fixed.py force.py movable.py)
SCRIPT=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh

export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1
conda activate desc

python -u ./$SCRIPT \
    --task_id   $TASK_ID \
    --num_tasks $NUM_TASKS