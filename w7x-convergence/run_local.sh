#!/bin/bash
# Run the first case (TASK_ID=0, n_quadpoints=100) locally without SLURM.
set -euo pipefail

cd "$(dirname "$0")"

export TASK_ID=1
export NUM_TASKS=8
export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1
export PYTHONPATH="/home/lf2869/Documents/Codes/coilforce/src:${PYTHONPATH:-}"

mkdir -p w7x_coil_0_logs w7x_coil_0_data w7x_coil_0_dolfinx

echo "Task:       $TASK_ID / $NUM_TASKS"
echo "Start time: $(date)"

conda run -n rod --no-capture-output \
    python -u ./w7x_coil_0_coilfem_only.py \
        --task_id   "$TASK_ID" \
        --num_tasks "$NUM_TASKS" \
    2>&1 | tee "w7x_coil_0_logs/local_task${TASK_ID}.out"

echo "End time: $(date)"
