# Instrumented copy of optimize_beams.py.  Original is unchanged.
# Physics / DOFs / constraints / minimize options match the original
# unless DEBUG_* env flags are set.  Artifacts go under logs/.

from coil_fem.simsopt import (
    CoilSupportBeamsSorted,
    BeamSurfaceDistance,
    BeamCurveAngle,
    BeamCurveDistance,
    CoilFEMObjective,
)
from simsopt.geo import CurveSurfaceDistance
from simsopt import save, load
import json
import numpy as np
import time
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from scipy.optimize import minimize, Bounds, LinearConstraint

from debug_hooks import (
    DebugSession,
    apply_interior_theta,
    build_preflight,
    constraint_from_optimizable_logged,
    env_flag,
    finite_stats,
    install_excepthook,
    logged_fun,
    preflight_is_clean,
    print_debug_flags,
    write_preflight,
)

print_debug_flags()

MAXITER = 1000

_ROOT = Path(__file__).resolve().parent.parent
cfqs_dict = load(str(_ROOT / "cfqs-data" / "cfqs_data.json"))
plasma_surface = cfqs_dict['plasma_surface']

# Setting beam parameters
w1_beam = 0.05
w2_beam = 0.1
fixed_dof_names = [
    # "thetas_orientation_cc",
    "w1_beam",
    "w2_beam",
]
if env_flag("DEBUG_FIX_THETAS") and "thetas_orientation_cc" not in fixed_dof_names:
    fixed_dof_names.append("thetas_orientation_cc")
    print("DEBUG_FIX_THETAS: added thetas_orientation_cc to fixed_dof_names")

# Loading the coils.
base_coils = cfqs_dict['base_coils']
base_curves = [c.curve for c in base_coils]
base_currents = [c.current for c in base_coils]

# ----- FEM / support options -----

_OPTIONS_PATH = _ROOT / "cfqs-options.json"
opts = json.load(open(_OPTIONS_PATH))
mesh_options = opts["mesh_options"]
material_options = opts["material_options"]
gravity_options = opts["gravity_options"]
problem_options = opts["problem_options"]
physics_options = opts["physics_options"]
beam_options = opts["beam_options"]
# beam_options["cross_section_type"] = "hollow_rectangle"
fixed_clamp_options = opts["fixed_clamp_options"]

# ----- Defining optimizable -----

# One support object covers the whole base coilset
coil_support = CoilSupportBeamsSorted(
    base_coils=base_coils,
    nfp=plasma_surface.nfp,
    stellsym=plasma_surface.stellsym,
    beam_options=beam_options,
    w1_beam=w1_beam,
    w2_beam=w2_beam,
    fixed_clamp_options=fixed_clamp_options,
    fixed_dof_names=fixed_dof_names,
)

skip_fem = env_flag("DEBUG_SKIP_FEM")
dummy_obj = env_flag("DEBUG_DUMMY_OBJ")
Jstress = None
if skip_fem:
    print("DEBUG_SKIP_FEM: skipping CoilFEMObjective")
else:
    # The Simsopt wrapper for a differentiable FEM problem.
    # It behaves like a simsopt objective.
    # max_von_mises_lse is strictly inferior to l2_von_mises (it can't make max lower)
    Jstress = CoilFEMObjective(
        coil_support,
        metrics          = ("sq_max_von_mises_lse",), # ("sq_max_von_mises_lse"), # ("l2_von_mises",),
        metric_weights   = (1.,),
        mesh_options     = mesh_options,
        material_options = material_options,
        gravity_options  = gravity_options,
        problem_options  = problem_options,
        physics_options  = physics_options,
        coupling         = "monolithic",
    )
    save([Jstress], "init_Jstress.json")
    Jstress.save_run_vtu("init_run")
    with open("init_summary.json", "w") as fp:
        summary = Jstress.summary()
        json.dump(summary, fp)
    print("# mesh node for all coils:", Jstress.n_nodes)
    print("# mesh cell for all coils:", Jstress.n_cells)

# ----- Beam-surface distance -----
# First, reading the coil plasma distance
min_csd = CurveSurfaceDistance(base_curves, plasma_surface, 0).shortest_distance()
Jbsd = BeamSurfaceDistance(coil_support, plasma_surface, min_csd * 0.9)

# ----- Beam-curve angle -----

Jbca = BeamCurveAngle(
    coil_support, minimum_angle=np.pi/6, mode="all"
)

# ----- Beam-curve distance

target_bcd = (
    np.sqrt(w1_beam**2 + w2_beam**2)
    + np.sqrt(mesh_options["w1"]**2 + mesh_options["w2"]**2)
)
Jbcd = BeamCurveDistance(
    coil_support,
    dead_length=target_bcd*2,
    minimum_distance=target_bcd*0.9,
)

# ----- Optimization -----

# Fix every coil degree of freedom (geometry + current) so only the free
# CoilSupportBeamsSorted dofs (the beam network) are optimized.
for c in base_curves:
    c.fix_all()
for cur in base_currents:
    cur.fix_all()

session = DebugSession()
install_excepthook(lambda: {
    **session.snapshot(),
    "script": "optimize_beams_debug.py",
})

opt = Jstress if Jstress is not None else coil_support
if env_flag("DEBUG_INTERIOR_THETA"):
    apply_interior_theta(opt)


def fun(dofs):
    Jstress.x = dofs
    J = Jstress.J()
    grad = Jstress.dJ()
    return J, grad


def _sum_dphis_constraint(dof_names):
    """Linear inequalities sum_j dphis*[i][j] <= 1 for each coil/group.

    Applies to free DOFs named ``dphis_start_cc`` and ``dphis_end_cc``
    (simsopt names like ``...:dphis_start_cc(i,j)``).
    """
    keys = ("dphis_start_cc", "dphis_end_cc")
    groups = defaultdict(list)
    for j, name in enumerate(dof_names):
        # simsopt: "CoilSupportBeamsSorted1:dphis_start_cc(0,3)"
        local = name.split(":", 1)[-1]
        key = local.split("(", 1)[0]
        if key not in keys:
            continue
        i_coil = int(local.split("(", 1)[1].split(",", 1)[0])
        groups[(key, i_coil)].append(j)
    n = len(dof_names)
    if not groups:
        return LinearConstraint(np.zeros((0, n)), -np.inf, np.zeros(0))
    A = np.zeros((len(groups), n))
    for row, idxs in enumerate(groups.values()):
        A[row, idxs] = 1.0
    return LinearConstraint(A, -np.inf, np.ones(A.shape[0]))


dofs = opt.x
lb, ub = opt.bounds
bounds = Bounds(lb, ub)
include_jbcd = not env_flag("DEBUG_DROP_JBCD")
linear = _sum_dphis_constraint(opt.dof_names)
constraints = [linear]
if env_flag("DEBUG_NO_NL_CONS"):
    print("DEBUG_NO_NL_CONS: bounds + sum_dphis only")
else:
    constraints.append(constraint_from_optimizable_logged(Jbsd, -np.inf, 0, "Jbsd", session))
    constraints.append(constraint_from_optimizable_logged(Jbca, -np.inf, 0, "Jbca", session))
    if include_jbcd:
        constraints.append(constraint_from_optimizable_logged(Jbcd, -np.inf, 0, "Jbcd", session))
    else:
        print("DEBUG_DROP_JBCD: omitting Jbcd constraint")

print("MAXITER =", MAXITER)
print("# free dofs =", len(dofs))
print("# linear inequality constraints =", linear.A.shape[0])

fun_x0 = None
if Jstress is not None:
    print("preflight: evaluating fun(x0)")
    J0, g0 = fun(np.asarray(dofs, dtype=float))
    fun_x0 = {"J": float(J0), **finite_stats(g0)}
    print("preflight fun(x0):", fun_x0)

no_nl = env_flag("DEBUG_NO_NL_CONS")
if no_nl:
    qr_hinges = []
    clean_keys = ()
elif include_jbcd:
    qr_hinges = [Jbsd, Jbca, Jbcd]
    clean_keys = ("Jbsd", "Jbca", "Jbcd")
else:
    qr_hinges = [Jbsd, Jbca]
    clean_keys = ("Jbsd", "Jbca")

payload = build_preflight(
    Jbsd=Jbsd,
    Jbca=Jbca,
    Jbcd=Jbcd,
    dofs=dofs,
    lb=lb,
    ub=ub,
    dof_names=opt.dof_names,
    linear_cons=linear,
    include_jbcd=include_jbcd,
    qr_hinges=qr_hinges,
    fun_x0=fun_x0,
    extra={
        "target_bcd": float(target_bcd),
        "script": "optimize_beams_debug.py",
        "skip_fem": skip_fem,
        "dummy_obj": dummy_obj,
    },
)
write_preflight(session.logs_dir / "preflight.json", payload)

if not preflight_is_clean(payload, hinge_keys=clean_keys):
    print("preflight found non-finite values or failed stacked QR; aborting", file=sys.stderr)
    sys.exit(2)

use_dummy = dummy_obj or skip_fem
if use_dummy:
    if skip_fem and not dummy_obj:
        print("DEBUG_SKIP_FEM without DEBUG_DUMMY_OBJ: using dummy objective")
    else:
        print("DEBUG_DUMMY_OBJ: using dummy objective in minimize")
    obj_fun = logged_fun(lambda x: (0.0, np.zeros_like(x)), session, tag="dummy")
else:
    obj_fun = logged_fun(fun, session, tag="fun")

time_filament_1 = time.time()
res = minimize(
    obj_fun, dofs, jac=True, method="trust-constr",
    bounds=bounds,
    constraints=constraints,
    options={
        "maxiter": MAXITER,
        "gtol": 1e-5,
        "xtol": 1e-5,
        "barrier_tol": 1e-5,
        "verbose": 2,
    },
)
time_filament_2 = time.time()
print("time", time_filament_2 - time_filament_1)
print("res ", res)

if Jstress is None:
    print("skipping fin_* FEM exports (no Jstress)")
    sys.exit(0)

save([Jstress], "fin_Jstress.json")
Jstress.save_run_vtu("fin_run")
with open("fin_results.pkl", "wb") as file:
    pickle.dump({
        "res": res,
        "time": time_filament_2 - time_filament_1,
    }, file)

with open("fin_summary.json", "w") as fp:
    summary = Jstress.summary()
    json.dump(summary, fp)
