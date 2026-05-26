# Benchmark – full W7-X coil set (5 base coils, nfp=5, stellsym=True),
# parallelised over quadpoints.
#
# Each SLURM array task handles the subset of N_QUADPOINTS_LIST assigned to
# it via stride slicing (task_id :: num_tasks).  Results are written to
# data_python/n_<N>/timings.npz and the corresponding VTU files.
#
# Usage (local, all quadpoints in one process):
#   python w7x_coil_set.py
#
# Usage (SLURM array, one GPU per task):
#   sbatch jobscript_w7x_coil_set.sh

from shared import (
    jax, jnp, np, plt, Path,
    N_QUADPOINTS_LIST,
    load_w7x_data, parse_task_args, get_task_items,
    run_one_n_quadpoints, load_results_from_disk,
)

print("✅ Imports successful")
print(f"JAX version: {jax.__version__}")

# ── Load W7-X data ────────────────────────────────────────────────────────────
curves, currents, axis, nfp_w7x, bs = load_w7x_data()

# ── 5 base coils, nfp=5, stellarator symmetry ────────────────────────────────
N_BASE = 5
base_curve_objs   = curves[:N_BASE]
base_current_objs = currents[:N_BASE]

save_dir = 'w7x_coil_set_data'

# ── Determine which quadpoints this task is responsible for ───────────────────
task_id, num_tasks = parse_task_args()
n_quadpoints_i = get_task_items(task_id)

print(f"Task {task_id}/{num_tasks}: running n_quadpoints = {n_quadpoints_i}")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")


# ── Run FEM for each assigned quadpoints value ────────────────────────────────
result, t_jit, t_run = run_one_n_quadpoints(
    n_quadpoints_i    = n_quadpoints_i,
    base_curve_objs   = base_curve_objs,
    base_current_objs = base_current_objs,
    nfp               = 5,
    stellsym          = True,
    save_dir          = save_dir,
)
print(f"[n={n_quadpoints_i}] Compile + run: {t_jit:.3f}s  |  Run only: {t_run:.3f}s")

