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

# ── PETSc solver options ──────────────────────────────────────────────────────
# coil-fem passes an empty dict for petsc_solver, so JAX-FEM would default to
# ksp_type='bcgsl' / pc_type='ilu'.  Patch petsc_solve to enforce the desired
# options.  Setting -ksp_type/-pc_type in the PETSc options database before the
# call ensures ksp.setFromOptions() initialises the PC to 'lu'; the subsequent
# ksp.pc.setType('lu') inside petsc_solve is then a no-op (type unchanged) so
# pc_factor_mat_solver_type='mumps' is preserved.
from petsc4py import PETSc as _PETSc
import jax_fem.solver as _jfem_solver

_orig_petsc_solve = _jfem_solver.petsc_solve

def _petsc_solve_preonly_lu(A, b, ksp_type, pc_type):
    _PETSc.Options()['-ksp_type'] = 'preonly'
    _PETSc.Options()['-pc_type']  = 'lu'
    _PETSc.Options()['-pc_factor_mat_solver_type'] = 'mumps'
    return _orig_petsc_solve(A, b, 'preonly', 'lu')

_jfem_solver.petsc_solve = _petsc_solve_preonly_lu


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
