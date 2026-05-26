#!/bin/bash
NUM_TASKS=8
mkdir -p w7x_coil_0_logs
mkdir -p w7x_coil_0_data
mkdir -p w7x_coil_0_dolfinx
for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do
    MEM_GB=$(awk "BEGIN {printf \"%d\", 20 + ($TASK_ID * 0.57)^3 / 100 * 120}")
    echo "Task $TASK_ID: ${MEM_GB}G"
    sbatch --mem=${MEM_GB}G \
           --output="w7x_coil_0_logs/slurm_%j_${TASK_ID}.out" \
           --error="w7x_coil_0_logs/slurm_%j_${TASK_ID}.err" \
           --export=ALL,TASK_ID=$TASK_ID,NUM_TASKS=$NUM_TASKS \
           jobscript_w7x_coil_0.sh

done