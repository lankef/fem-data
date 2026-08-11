#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-15


mkdir -p logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

n_clamp_vals=(0 1 2 3)
n_beam_cc_vals=(0 2 4 6)
i=$((SLURM_ARRAY_TASK_ID / 4))
j=$((SLURM_ARRAY_TASK_ID % 4))
export N_CLAMP=${n_clamp_vals[$i]}
export N_BEAM_CC=${n_beam_cc_vals[$j]}
export SAVE_DIR="data/job_${SLURM_ARRAY_TASK_ID}/"
mkdir -p "$SAVE_DIR"

echo "Job ID:         $SLURM_JOB_ID"
echo "Array task ID:  $SLURM_ARRAY_TASK_ID"
echo "N_CLAMP:        $N_CLAMP"
echo "N_BEAM_CC:      $N_BEAM_CC"
echo "SAVE_DIR:       $SAVE_DIR"
echo "Nodes:          $SLURM_JOB_NUM_NODES"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Start time:     $(date)"
conda activate desc
python -u ./optimize_beams.py 

echo "End time:       $(date)"
