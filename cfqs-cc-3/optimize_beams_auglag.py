# This protototype py optimizes support only fixing 
# coil dofs. Its goal is to investigate how challenging 
# support optimization is. If support optimization is 
# not too bad, then we can handle support like an optimization
# subproblem, because handling support and coils together
# can cause coils to converge prematurely. 
#
# Constrained variant: CoilSupportBeamsSorted + NLopt AUGLAG
# (L-BFGS local) with box bounds from Jstress.bounds and
# inequalities sum_j dphis*[i][j] <= 1 per coil/group for
# dphis_start_cc and dphis_end_cc. Beam–plasma distance is
# scaled 10x so its initial AL penalty is stronger than the
# other constraints.

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
from collections import defaultdict
from pathlib import Path
import nlopt

MAXITER = 1000
BSD_PENALTY_WEIGHT = 10.0  # others stay 1

# NLopt result codes (nlopt.h). Printed because set_exceptions_enabled(False)
# swallows NLOPT_FAILURE instead of raising.
_NLOPT_RESULT = {
    -5: "FORCED_STOP",
    -4: "ROUNDOFF_LIMITED",
    -3: "OUT_OF_MEMORY",
    -2: "INVALID_ARGS",
    -1: "FAILURE",
    1: "SUCCESS",
    2: "STOPVAL_REACHED",
    3: "FTOL_REACHED",
    4: "XTOL_REACHED",
    5: "MAXEVAL_REACHED",
    6: "MAXTIME_REACHED",
}
_EVAL_LOG = Path("auglag_evals.jsonl")

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
fixed_clamp_options = opts["fixed_clamp_options"]

# ----- Adding stellarator-symmetric beams -----

beam_options['i_beam_cs'] = [(2, 2), (3, 3), (3, 3)]
beam_options['s_beam_cs'] = [True, True, False]

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

# The Simsopt wrapper for a differentiable FEM problem.
# It behaves like a simsopt objective.
# max_von_mises_lse is strictly inferior to l2_von_mises (it can't make max lower)
Jstress = CoilFEMObjective(
    coil_support,
    metrics          = ("l2_von_mises",), # ("sq_max_von_mises_lse"), # ("l2_von_mises",),
    metric_weights   = (1.,),
    mesh_options     = mesh_options,
    material_options = material_options,
    gravity_options  = gravity_options,
    problem_options  = problem_options,
    physics_options  = physics_options,
    coupling         = "monolithic",
)
save([Jstress], "init_Jstress_auglag.json")
Jstress_init = float(Jstress.J())
Jstress.save_run_vtu("init_run_auglag")
with open("init_summary_auglag.json", "w") as fp:
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

# ----- Beam-curve distance -----

# The ratio of w and coil-coil distance of CFQS
# seems to often cause this dead zone to cover
# the full beams.
# target_bcd = (
#     np.sqrt(w1_beam**2 + w2_beam**2)
#     + np.sqrt(mesh_options["w1"]**2 + mesh_options["w2"]**2)
# )
# Jbcd = BeamCurveDistance(
#     coil_support, 
#     dead_length=target_bcd*2,
#     minimum_distance=target_bcd*0.9,
# )

# ----- Optimization -----

# Fix every coil degree of freedom (geometry + current) so only the free
# CoilSupportBeamsSorted dofs (the beam network) are optimized.
for c in base_curves:
    c.fix_all()
for cur in base_currents:
    cur.fix_all()


_counts = defaultdict(int)
_t0 = None
_x0 = None


def _log(record):
    """Append one JSON line and echo it. Survives a killed slurm job."""
    record["t"] = time.time() - _t0
    line = json.dumps(record, default=float)
    print(line, flush=True)
    with open(_EVAL_LOG, "a") as fp:
        fp.write(line + "\n")


def _arr_stats(name, a):
    a = np.asarray(a, dtype=float)
    return {
        f"{name}_norm": float(np.linalg.norm(a)),
        f"{name}_maxabs": float(np.max(np.abs(a))) if a.size else 0.0,
        f"{name}_finite": bool(np.all(np.isfinite(a))),
    }


def nlopt_fun(x, grad):
    want_grad = grad.size > 0
    _counts["f"] += 1
    _counts["f_grad"] += int(want_grad)
    Jstress.x = x
    g = None
    if want_grad:
        g = np.asarray(Jstress.dJ() / Jstress_init, dtype=float)
        grad[:] = g
    val = float(Jstress.J()) / Jstress_init
    rec = {
        "kind": "f",
        "n": _counts["f"],
        "want_grad": want_grad,
        "J": val,
        "dx": float(np.linalg.norm(np.asarray(x) - _x0)),
        **_arr_stats("x", x),
    }
    if g is not None:
        rec.update(_arr_stats("g", g))
    _log(rec)
    return val


def nlopt_ineq_from_optimizable(obj, weight=1.0, tag="c"):
    """Same as constraint_from_optimizable(obj, -inf, 0), plus a scale."""
    def fc(x, grad):
        want_grad = grad.size > 0
        _counts[tag] += 1
        obj.x = x
        raw = float(obj.J())
        val = float(weight * raw)
        if want_grad:
            g = weight * np.asarray(obj.dJ(), dtype=float)
            grad[:] = g
            g_finite = bool(np.all(np.isfinite(g)))
        else:
            g_finite = True
        # Constraints are cheap; log residuals every call so we see AUGLAG's
        # first rho-setup eval and any NaN/Inf.
        _log({
            "kind": tag,
            "n": _counts[tag],
            "want_grad": want_grad,
            "c": val,
            "c_raw": raw,
            "c_finite": bool(np.isfinite(val)),
            "g_finite": g_finite,
            "dx": float(np.linalg.norm(np.asarray(x) - _x0)),
        })
        return val
    return fc


def _sum_dphis_A(dof_names):
    """Build A for inequalities sum_j dphis*[i][j] <= 1 per coil/group.

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
        return np.zeros((0, n))
    A = np.zeros((len(groups), n))
    for row, idxs in enumerate(groups.values()):
        A[row, idxs] = 1.0
    return A


def nlopt_dphis_ineq(A_row):
    def fc(x, grad):
        _counts["dphis"] += 1
        if grad.size > 0:
            grad[:] = A_row
        return float(A_row @ x - 1.0)  # sum_j dphis[g,j] - 1 <= 0
    return fc


dofs = np.asarray(Jstress.x, dtype=float)
lb, ub = Jstress.bounds
lb = np.asarray(lb, dtype=float)
ub = np.asarray(ub, dtype=float)
A = _sum_dphis_A(Jstress.dof_names)
dphis_c = A @ dofs - 1.0

n = len(dofs)
_x0 = dofs.copy()
if _EVAL_LOG.exists():
    _EVAL_LOG.unlink()

# Snapshot at x0 before AUGLAG. Jbsd/Jbca are FEM-free; J/dJ reuse the
# init solve already paid for by summary() / save_run_vtu.
J0 = float(Jstress.J())
g0 = np.asarray(Jstress.dJ(), dtype=float)
c_bsd = float(Jbsd.J())
c_bca = float(Jbca.J())
viol = np.array(
    [BSD_PENALTY_WEIGHT * c_bsd, c_bca, *np.maximum(dphis_c, 0.0)],
    dtype=float,
)
con2 = float(np.sum(viol ** 2))
rho_guess = 10.0 if con2 <= 0 else max(1e-6, min(10.0, 2.0 * abs(J0) / con2))
print("MAXITER =", MAXITER)
print("# free dofs =", n)
print("# linear inequality constraints =", A.shape[0])
print("BSD_PENALTY_WEIGHT =", BSD_PENALTY_WEIGHT)
print("x0 finite", np.all(np.isfinite(dofs)), "in bounds",
      bool(np.all(dofs >= lb - 1e-15) and np.all(dofs <= ub + 1e-15)))
print("lb/ub finite", np.all(np.isfinite(lb)), np.all(np.isfinite(ub)))
print("J0", J0, "||g0||", float(np.linalg.norm(g0)),
      "g0 finite", bool(np.all(np.isfinite(g0))))
print("Jbsd", c_bsd, "Jbsd*w", BSD_PENALTY_WEIGHT * c_bsd)
print("Jbca", c_bca)
print("dphis residuals (sum-1)", dphis_c)
print("con2", con2, "auglag rho guess", rho_guess)
print("nlopt", getattr(nlopt, "__version__", "?"))

local = nlopt.opt(nlopt.LD_LBFGS, n)
local.set_xtol_rel(1e-5)
local.set_ftol_rel(1e-5)

opt = nlopt.opt(nlopt.AUGLAG, n)
opt.set_local_optimizer(local)
opt.set_min_objective(nlopt_fun)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)
opt.set_maxeval(MAXITER)
opt.set_xtol_rel(1e-5)
opt.set_ftol_rel(1e-5)
opt.add_inequality_constraint(
    nlopt_ineq_from_optimizable(Jbsd, BSD_PENALTY_WEIGHT/plasma_surface.minor_radius(), tag="Jbsd"), 1e-8
)
opt.add_inequality_constraint(
    nlopt_ineq_from_optimizable(Jbca, 1.0, tag="Jbca"), 1e-8
)
for i, row in enumerate(A):
    opt.add_inequality_constraint(nlopt_dphis_ineq(row), 1e-8)
opt.set_exceptions_enabled(False)

print("starting opt.optimize", flush=True)
_t0 = time.time()
time_filament_1 = _t0
xopt = opt.optimize(dofs)
time_filament_2 = time.time()
Jstress.x = xopt
result_code = opt.last_optimize_result()
minf = opt.last_optimum_value()
result_name = _NLOPT_RESULT.get(result_code, f"UNKNOWN({result_code})")
print("time", time_filament_2 - time_filament_1)
print("result_code", result_code, result_name)
print("minf", minf)
print("||xopt-x0||", float(np.linalg.norm(np.asarray(xopt) - _x0)))
print("eval counts", dict(_counts))
print("xopt", xopt)
if result_code < 0:
    print(
        "NLopt returned a failure code; exceptions were disabled so the "
        "script continued and wrote fin_* at the last evaluated x "
        "(x0 if no step was accepted)."
    )
save([Jstress], "fin_Jstress_auglag.json")
Jstress.save_run_vtu("fin_run_auglag")
with open("fin_results_auglag.pkl", "wb") as file:
    pickle.dump({
        "result_code": result_code,
        "result_name": result_name,
        "minf": minf,
        "x": np.asarray(xopt),
        "x0": _x0,
        "J0": J0,
        "g0_norm": float(np.linalg.norm(g0)),
        "Jbsd0": c_bsd,
        "Jbca0": c_bca,
        "con2": con2,
        "rho_guess": rho_guess,
        "counts": dict(_counts),
        "time": time_filament_2 - time_filament_1,
    }, file)

with open("fin_summary_auglag.json", "w") as fp:
    summary = Jstress.summary()
    json.dump(summary, fp)
