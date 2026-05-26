# Benchmark – single W7-X coil (coil 0), parallelised over quadpoints.
#
# Each SLURM array task handles the subset of N_QUADPOINTS_LIST assigned to
# it via stride slicing (task_id :: num_tasks).  Results are written to
# w7x_coil_0_data/n_<N>/timings.npz and the corresponding VTU files.
#
# Usage (local, all quadpoints in one process):
#   python w7x_coil_0.py
#
# Usage (SLURM array, one GPU per task):
#   sbatch jobscript_w7x_coil_0.sh

from shared import (
    jax, jnp, np, plt, Path,
    N_QUADPOINTS_LIST,
    load_w7x_data, parse_task_args, get_task_items,
    run_one_n_quadpoints, load_results_from_disk,
)

print("✅ Imports successful")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

# ── Load W7-X data ────────────────────────────────────────────────────────────
curves, currents, axis, nfp_w7x, bs = load_w7x_data()

# ── Single coil (coil 0) – no field-period symmetry ──────────────────────────
base_curve_objs   = [curves[0]]
base_current_objs = [currents[0]]

save_dir = 'w7x_coil_0_bench'

# ── Determine which quadpoints this task is responsible for ───────────────────
task_id, num_tasks = parse_task_args()
n_quadpoints_i = get_task_items(task_id)

print(f"Task {task_id}/{num_tasks}: running n_quadpoints = {n_quadpoints_i}")

# ── Run FEM for each assigned quadpoints value ────────────────────────────────
result, t_jit, t_run = run_one_n_quadpoints(
    n_quadpoints_i  = n_quadpoints_i,
    base_curve_objs = base_curve_objs,
    base_current_objs = base_current_objs,
    nfp             = 1,      # no field-period symmetry for a single coil
    stellsym        = False,
    save_dir        = save_dir,
)
print(f"[n={n_quadpoints_i}] Compile + run: {t_jit:.3f}s  |  Run only: {t_run:.3f}s")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")


