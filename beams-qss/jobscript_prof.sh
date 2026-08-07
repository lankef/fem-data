#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --gres=gpu:l40s:1

mkdir -p logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate desc

# python -u ./proto_prof.py
export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1

# Leave headroom for cuDSS, which allocates outside XLA's pool
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# NOTE: do NOT set JAX_PLATFORMS. coil_fem/magnetic.py repairs simsopt's
# jax_platform_name='cpu' pin only when JAX_PLATFORMS is unset.

REPORT="coilfem_${SLURM_JOB_ID}"

echo "Job ID:         $SLURM_JOB_ID"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Start time:     $(date)"
nsys --version

nsys profile \
  -t cuda,nvtx,osrt \
  --capture-range=nvtx --nvtx-capture="capture" \
  --capture-range-end=stop \
  -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
  --cuda-memory-usage=true \
  --trace-fork-before-exec=true \
  -o "$REPORT" \
  python -u ./proto_prof.py


echo "----- nsys stats -----"
for r in nvtx_gpu_proj_sum cuda_gpu_kern_sum cuda_api_sum cuda_gpu_mem_size_sum; do
  echo "===== $r ====="
  nsys stats --report "$r" --force-export=true "${REPORT}.nsys-rep"
done

echo "End time:       $(date)"