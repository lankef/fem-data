#!/bin/bash -l
#SBATCH --account=torch_pr_292_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --gres=gpu:l40s:1

# Preferred (flags after script name):
#   sbatch jobscript_taylor.sh
#   sbatch jobscript_taylor.sh --diagnostics
#   sbatch jobscript_taylor.sh --force-kt-adjoint
#   sbatch jobscript_taylor.sh --drop-winkler-wa
#   sbatch jobscript_taylor.sh --simsopt-free
#   sbatch jobscript_taylor.sh --compare-wa-slice
#   sbatch jobscript_taylor.sh --vjp-ablation freeze_k
#   sbatch jobscript_taylor.sh --vjp-ablation freeze_sdofs_geom
#
# Deep diagnostics (one phase per job; sync coil-fem + taylor.py first):
#   sbatch jobscript_taylor.sh --diagnostics-deep groups
#   sbatch jobscript_taylor.sh --diagnostics-deep residual
#   sbatch jobscript_taylor.sh --diagnostics-deep ablation --vjp-ablation none
#   sbatch jobscript_taylor.sh --diagnostics-deep ablation --vjp-ablation freeze_k
#   sbatch jobscript_taylor.sh --diagnostics-deep ablation --vjp-ablation freeze_sdofs_geom
#   sbatch jobscript_taylor.sh --diagnostics-deep ablation-sum
#   sbatch jobscript_taylor.sh --diagnostics-deep surgical --vjp-ablation freeze_wa_in_k
#   sbatch jobscript_taylor.sh --diagnostics-deep surgical --vjp-ablation freeze_wa_in_coupling
#   sbatch jobscript_taylor.sh --diagnostics-deep ift
#   sbatch jobscript_taylor.sh --diagnostics-deep shared-wa
#   sbatch jobscript_taylor.sh --diagnostics-deep scale --n-coils 1
#   sbatch jobscript_taylor.sh --diagnostics-deep scale --n-coils 5
#
# Fallback if the cluster strips sbatch args:
#   sbatch --export=ALL,TAYLOR_FLAGS=--diagnostics jobscript_taylor.sh
#   sbatch --export=ALL,TAYLOR_FLAGS=--diagnostics-deep residual jobscript_taylor.sh
#   sbatch --export=ALL,TAYLOR_FLAGS=--force-kt-adjoint jobscript_taylor.sh
#   (no quotes around the flag value)

mkdir -p logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
export PETSC_OPTIONS="-no_signal_handler"
export PYTHONFAULTHANDLER=1

echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_JOB_NUM_NODES"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Start time:     $(date)"
echo "PWD:            $PWD"
echo "taylor.py:      $(ls -l ./taylor.py)"
echo "sbatch \$@:     $*"
echo "TAYLOR_FLAGS:   ${TAYLOR_FLAGS-}"

conda activate desc
# Prefer the env's libstdc++ over the module anaconda one (basix CXXABI).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "CONDA_PREFIX:   $CONDA_PREFIX"

# sbatch args take precedence; else TAYLOR_FLAGS.
if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
elif [[ -n "${TAYLOR_FLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($TAYLOR_FLAGS)
else
  EXTRA_ARGS=()
fi
echo "Running: python -u ./taylor.py ${EXTRA_ARGS[*]-}"

python -u ./taylor.py "${EXTRA_ARGS[@]}"
status=$?
echo "Exit code:      $status"
echo "End time:       $(date)"
exit $status
