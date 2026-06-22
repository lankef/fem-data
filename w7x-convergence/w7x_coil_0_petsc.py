# Benchmark – single W7-X coil (coil 0), parallelised over quadpoints.
# PETSc linear-solver variant of w7x_coil_0.py.
#
# Identical to w7x_coil_0.py except:
#   * solver='petsc' is passed to run_one_n_quadpoints (CPU direct solver)
#   * results are written to w7x_coil_0_data_petsc/
#   * dolfinx validation goes to w7x_coil_0_petsc_dolfinx/
#
# Usage (local):
#   python w7x_coil_0_petsc.py
#
# Usage (SLURM array):
#   sbatch jobscript_w7x_coil_0_petsc.sh   (via submit_petsc.sh)

from shared import (
    jax, jnp, np, plt, Path,
    N_QUADPOINTS_LIST,
    load_w7x_data, parse_task_args, get_task_items,
    run_one_n_quadpoints, load_results_from_disk,
    validate_with_dolfinx,
)


# ── Load W7-X data ────────────────────────────────────────────────────────────
curves, currents, axis, nfp_w7x, bs = load_w7x_data()

# ── Single coil (coil 0) – no field-period symmetry ──────────────────────────
base_curve_objs   = [curves[0]]
base_current_objs = [currents[0]]

save_dir = 'w7x_coil_0_data_petsc'

# ── Determine which quadpoints this task is responsible for ───────────────────
task_id, num_tasks = parse_task_args()
n_quadpoints_i = get_task_items(task_id)

# ── Run FEM for each assigned quadpoints value (PETSc linear solver) ─────────

result, fem, t_jit, t_run = run_one_n_quadpoints(
    n_quadpoints_i  = n_quadpoints_i,
    base_curve_objs = base_curve_objs,
    base_current_objs = base_current_objs,
    nfp             = 1,      # no field-period symmetry for a single coil
    stellsym        = False,
    save_dir        = save_dir,
    solver          = 'petsc',
)
print(f"[n={n_quadpoints_i}] Compile + run: {t_jit:.3f}s  |  Run only: {t_run:.3f}s")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
validate_with_dolfinx(
    coil_fem=fem,
    out_dir = f"w7x_coil_0_petsc_dolfinx/n_{n_quadpoints_i}",
    run_A = True,
    run_B = True,
    result_coilfem = result,
)
