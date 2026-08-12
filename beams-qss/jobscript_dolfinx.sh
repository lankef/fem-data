#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# Array job: each task runs beam_dolfinx.py against one of the two exported
# cases (fin_dolfinx/, init_dolfinx/), produced earlier by run_export.py from
# Jstress_fin.json / Jstress_init.json.
#   sbatch jobscript_dolfinx.sh

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

CASES=(fin init)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="${CASE}_dolfinx"

echo "Case:           $CASE"
echo "Run dir:        $RUN_DIR"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "Missing $RUN_DIR (expected output of run_export.py)" >&2
    exit 1
fi

cd "$RUN_DIR"
ln -sf "../Jstress_${CASE}.json" Jstress.json

python -u ../beam_dolfinx.py

status=$?
echo "Exit code:      $status"
echo "End time:       $(date)"
exit $status
