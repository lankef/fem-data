# No-FEM constraint audit + dummy trust-constr (Phase 0).
# Rebuilds the same geometric objects as optimize_beams.py without
# CoilFEMObjective / cuDSS.  Artifacts go under logs/.

from collections import defaultdict
from pathlib import Path

import json
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from simsopt import load
from simsopt.geo import CurveSurfaceDistance

from coil_fem.simsopt import (
    BeamCurveAngle,
    BeamCurveDistance,
    BeamSurfaceDistance,
    CoilSupportBeamsSorted,
)
from debug_hooks import (
    DebugSession,
    apply_interior_theta,
    build_preflight,
    constraint_from_optimizable_logged,
    env_flag,
    install_excepthook,
    logged_fun,
    print_debug_flags,
    write_preflight,
)

AUDIT_FLAGS = ("DEBUG_DROP_JBCD", "DEBUG_INTERIOR_THETA")

print_debug_flags(AUDIT_FLAGS)

_ROOT = Path(__file__).resolve().parent.parent
cfqs_dict = load(str(_ROOT / "cfqs-data" / "cfqs_data.json"))
plasma_surface = cfqs_dict["plasma_surface"]

w1_beam = 0.05
w2_beam = 0.1
fixed_dof_names = [
    "w1_beam",
    "w2_beam",
]

base_coils = cfqs_dict["base_coils"]
base_curves = [c.curve for c in base_coils]
base_currents = [c.current for c in base_coils]

_OPTIONS_PATH = _ROOT / "cfqs-options.json"
opts = json.load(open(_OPTIONS_PATH))
mesh_options = opts["mesh_options"]
beam_options = opts["beam_options"]
fixed_clamp_options = opts["fixed_clamp_options"]

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

min_csd = CurveSurfaceDistance(base_curves, plasma_surface, 0).shortest_distance()
Jbsd = BeamSurfaceDistance(coil_support, plasma_surface, min_csd * 0.9)
Jbca = BeamCurveAngle(coil_support, minimum_angle=np.pi / 6, mode="all")
target_bcd = (
    np.sqrt(w1_beam**2 + w2_beam**2)
    + np.sqrt(mesh_options["w1"] ** 2 + mesh_options["w2"] ** 2)
)
Jbcd = BeamCurveDistance(
    coil_support,
    dead_length=target_bcd * 2,
    minimum_distance=target_bcd * 0.9,
)

for c in base_curves:
    c.fix_all()
for cur in base_currents:
    cur.fix_all()


def _sum_dphis_constraint(dof_names):
    keys = ("dphis_start_cc", "dphis_end_cc")
    groups = defaultdict(list)
    for j, name in enumerate(dof_names):
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


session = DebugSession()
install_excepthook(lambda: {
    **session.snapshot(),
    "script": "audit_init_constraints.py",
})

if env_flag("DEBUG_INTERIOR_THETA"):
    apply_interior_theta(coil_support)

include_jbcd = not env_flag("DEBUG_DROP_JBCD")
dofs = np.asarray(coil_support.x, dtype=float)
lb, ub = coil_support.bounds
bounds = Bounds(lb, ub)
linear = _sum_dphis_constraint(coil_support.dof_names)
constraints = [linear]
constraints.append(constraint_from_optimizable_logged(Jbsd, -np.inf, 0, "Jbsd", session))
constraints.append(constraint_from_optimizable_logged(Jbca, -np.inf, 0, "Jbca", session))
if include_jbcd:
    constraints.append(constraint_from_optimizable_logged(Jbcd, -np.inf, 0, "Jbcd", session))

print("target_bcd =", target_bcd)
print("dead_length =", target_bcd * 2)
print("# free dofs =", len(dofs))
print("# linear inequality constraints =", linear.A.shape[0])
print("include_jbcd =", include_jbcd)

payload = build_preflight(
    Jbsd=Jbsd,
    Jbca=Jbca,
    Jbcd=Jbcd,
    dofs=dofs,
    lb=lb,
    ub=ub,
    dof_names=coil_support.dof_names,
    linear_cons=linear,
    include_jbcd=include_jbcd,
    extra={"target_bcd": float(target_bcd), "script": "audit_init_constraints.py"},
)
write_preflight(session.logs_dir / "preflight.json", payload)

print("stacked_qr:", payload["stacked_qr"])
print("Jbcd_geometry n_active_spans =", payload["Jbcd_geometry"]["n_active_spans"])
print("Jbcd_shortest =", payload["Jbcd_shortest"])
print("n_thetas_on_lb =", payload["bounds"]["n_thetas_on_lb"])

print("Running dummy trust-constr (maxiter=2)")
res = minimize(
    logged_fun(lambda x: (0.0, np.zeros_like(x)), session, tag="dummy"),
    dofs,
    jac=True,
    method="trust-constr",
    bounds=bounds,
    constraints=constraints,
    options={
        "maxiter": 2,
        "gtol": 1e-5,
        "xtol": 1e-5,
        "barrier_tol": 1e-5,
        "verbose": 2,
    },
)
print("dummy res.message =", res.message)
print("dummy res.niter =", getattr(res, "niter", None))
