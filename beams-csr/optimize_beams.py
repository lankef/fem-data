# This protototype py optimizes support only fixing
# coil dofs. Its goal is to investigate how challenging
# support optimization is. If support optimization is
# not too bad, then we can handle support like an optimization
# subproblem, because handling support and coils together
# can cause coils to converge prematurely.
#
# Constrained variant: CoilSupportBeamsCSRSorted + scipy trust-constr
# with box bounds from build_problem and linear span inequalities:
# coil-side / CC / CF / start_cr sum beam increments after the first
# (per coil/group, ub=1); dphis_end_cr sums coil increments after the
# first (per beam column, ub=one CSR sector). Plus CSRVolume < 9,
# CSRCurveDistance == 0, and CSRSurfaceDistance == 0.

from coil_fem.simsopt import (
    CoilSupportBeamsCSRSorted,
    BeamSurfaceDistance,
    BeamCurveAngle,
    BeamCurveDistance,
    CoilFEMObjective,
    CSRVolume,
    CSRCurveDistance,
    CSRSurfaceDistance,
)
from simsopt.configs import get_data
from simsopt.mhd import Vmec
from simsopt.geo import CurveSurfaceDistance
from simsopt import save, load
import json
import math
import numpy as np
import jax
import time
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from simsopt.field import Coil
from scipy.optimize import minimize, LinearConstraint

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from opt_utils import build_problem

# Loading the W7-X standard configuration
# plasma surface. wout file comes from Landreman"s
# VMEC equilibrium archive:
# https://github.com/landreman/vmec_equilibria/blob/master/W7-X/Standard/
eq = Vmec("../fixed-continuation/wout.nc", keep_all_files=True)
# Adjusting resolution and taking a half-field-period
n_phi = 25
n_theta = 50
MAXITER = 2000
MAXFUN = 20000
plasma_surface = type(eq.boundary)(
    nfp=eq.boundary.nfp, stellsym=eq.boundary.stellsym,
    mpol=eq.boundary.mpol, ntor=eq.boundary.ntor,
    quadpoints_phi=np.linspace(0, 1/2/eq.boundary.nfp, n_phi, endpoint=False),
    quadpoints_theta=np.linspace(0, 1, n_theta, endpoint=False),
)
plasma_surface.set_dofs(eq.boundary.get_dofs())

r_beam = 0.08
# Orientations and beam radius fixed; csr_curve_dofs left free for volume.
fixed_dof_names = [
    "thetas_orientation_cc",
    "thetas_orientation_cf",
    "thetas_orientation_cr",
    "r_beam",
]

# Loading the W7-X coils.
coil_per_half_fp = 5
# Cold start init state
# The number of quadpoints here controls the mesh
# density in coil-fem. The meshing routine will
# try choose the cell number in the cross section
# to make the aspect ratio close to one. Please see
# the next sections for details.
# curves, currents, axis, nfp, bs = get_data(
#     "w7x",
#     coil_order=8,
#     points_per_period=24 # <<<<< SET RESOLUTION HERE
# )
# base_curves = curves[:coil_per_half_fp]
# base_currents = currents[:coil_per_half_fp]
Jstress = load('../beams-qss/Jstress_csr.json')[0]
base_curves = Jstress._coil_support.base_curves
base_currents = Jstress._coil_support.base_currents

# Coil-surface distance
Jcsdist_init = CurveSurfaceDistance(base_curves, plasma_surface, 0)
dmin_cp = Jcsdist_init.shortest_distance()

# ----- FEM / support options -----

_OPTIONS_PATH = Path(__file__).resolve().parent.parent / "beam-options.json"
opts = json.load(open(_OPTIONS_PATH))
mesh_options = opts["mesh_options"]
material_options = opts["material_options"]
gravity_options = opts["gravity_options"]
problem_options = opts["problem_options"]
physics_options = opts["physics_options"]
beam_options = opts["beam_options"]
beam_options["n_beam_cr"] = 1
fixed_clamp_options = opts["fixed_clamp_options"]
mesh_scale = 0.5

# Circular CSR: R=4 m, section 0.3 x 0.5 m, Fourier order 2.
csr_order = 1 # 2
csr_r = 3.5
csr_options = {
    "order": csr_order,
    "w1": 0.3,   # width
    "w2": 0.5,   # height
    "n_phi": 64,
    "E": beam_options["E"],
    "nu": beam_options["nu"],
}
# stellsym CurveRZFourier: 2*order+1 DOFs; circle = rc0=R
csr_curve_dofs = np.zeros(2 * csr_order + 1)
csr_curve_dofs[0] = csr_r

# ----- Defining optimizable -----

# One support object covers the whole base coilset
base_coils = [Coil(c, I) for c, I in zip(base_curves, base_currents)]
coil_support = CoilSupportBeamsCSRSorted(
    base_coils=base_coils,
    nfp=eq.boundary.nfp,
    stellsym=eq.boundary.stellsym,
    beam_options=beam_options,
    csr_options=csr_options,
    problem_options=problem_options,
    csr_curve_dofs=csr_curve_dofs,
    r_beam=r_beam,
    fixed_clamp_options={"enabled": False}, # fixed_clamp_options,
    fixed_dof_names=fixed_dof_names,
)

# The Simsopt wrapper for a differentiable FEM problem.
# It behaves like a simsopt objective.
# max_von_mises_lse is strictly inferior to l2_von_mises (it can't make max lower)
# Jstress = CoilFEMObjective(
#     coil_support,
#     metrics          = ("sq_max_von_mises_lse"), # ("sq_max_von_mises_lse"), # ("l2_von_mises",),
#     metric_weights   = (1.,),
#     mesh_options     = mesh_options,
#     material_options = material_options,
#     gravity_options  = gravity_options,
#     problem_options  = problem_options,
#     physics_options  = physics_options,
#     coupling         = "monolithic",
# )
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
    coil_support, minimum_angle=np.pi/8, mode="all"
)

# ----- Beam-curve distance

target_bcd = r_beam + np.sqrt(mesh_options["w1"]**2 + mesh_options["w2"]**2)
Jbcd = BeamCurveDistance(
    coil_support,
    dead_length=target_bcd*2,
    minimum_distance=target_bcd*0.9,
)

# ----- CSR volume -----

Jvol = CSRVolume(coil_support)

# ----- CSR–coil centreline distance -----

dmin_csrcc = (
    0.5 * math.hypot(csr_options["w1"], csr_options["w2"])
    + 0.5 * math.hypot(mesh_options["w1"], mesh_options["w2"])
)
Jcsrcc = CSRCurveDistance(coil_support, dmin_csrcc)

# ----- CSR–plasma surface distance -----

dmin_csrs = dmin_csrcc + dmin_cp
Jcsrs = CSRSurfaceDistance(coil_support, plasma_surface, dmin_csrs)

# ----- Optimization -----

# Fix every coil degree of freedom (geometry + current) so only the free
# CoilSupportBeamsCSRSorted dofs (the beam network + CSR curve) are optimized.
for c in base_curves:
    c.fix_all()
for cur in base_currents:
    cur.fix_all()


def _sum_dphis_constraint(dof_names):
    """Linear inequalities on Sorted angle increments (attachment span).

    Sorted encoding is ``phi = cumsum(dphis)`` along the increment axis, so
    the sum of increments after the first is the span of the cluster.

    Coil-side / CC / CF / clamp keys increment along the beam axis: for each
    ``(key, i_coil)``, ``sum_{j>=1} dphis*(i,j) <= ub``.  ``dphis_end_cr``
    increments along the coil axis: for each beam column ``j``,
    ``sum_{i>=1} dphis_end_cr(i,j) <= ub_end_cr``.

    ``ub`` is ``1`` for coil-side / CC / CF / clamp keys.  For
    ``dphis_end_cr`` (CSR attachment angles) it is one field period
    (``1/nfp``), or half of that under stellarator symmetry
    (``1/(2 nfp)``), matching the CSR sector CR ends attach to.
    """
    nfp = int(coil_support.nfp)
    stellsym = bool(coil_support.stellsym)
    ub_end_cr = 1.0 / nfp / (2.0 if stellsym else 1.0)

    # Keys we constrain, and their default span upper bound.
    keys = {
        "dphis": 1.0,
        "dphis_start_cc": 1.0,
        "dphis_end_cc": 1.0,
        "dphis_start_cf": 1.0,
        "dphis_start_cr": 1.0,
        "dphis_end_cr": ub_end_cr,
    }

    # Group free DOFs.  simsopt names look like:
    # "CoilSupportBeamsCSRSorted1:dphis_end_cr(0,2)"
    # → key="dphis_end_cr", i_coil=0, beam_j=2.
    # Coil-side keys group by (key, i_coil) and sum beam_j >= 1.
    # dphis_end_cr groups by (key, beam_j) and sums i_coil >= 1.
    groups = defaultdict(list)
    for dof_index, name in enumerate(dof_names):
        local = name.split(":", 1)[-1]
        if "(" not in local:
            continue
        key, rest = local.split("(", 1)
        if key not in keys:
            continue
        idxs = rest.rstrip(")").split(",")
        i_coil = int(idxs[0])
        beam_j = int(idxs[1]) if len(idxs) > 1 else 0
        if key == "dphis_end_cr":
            groups[(key, beam_j)].append((dof_index, i_coil))
        else:
            groups[(key, i_coil)].append((dof_index, beam_j))

    n = len(dof_names)
    rows = []
    ub_list = []
    for (key, _grp), entries in groups.items():
        skip_first = 1  # exclude the first increment along the increment axis
        idxs = [
            dof_index for dof_index, along in entries if along >= skip_first
        ]
        if not idxs:
            continue  # only one increment → span is identically 0
        rows.append(idxs)
        ub_list.append(keys[key])

    if not rows:
        return LinearConstraint(np.zeros((0, n)), -np.inf, np.zeros(0))

    A = np.zeros((len(rows), n))
    for row, idxs in enumerate(rows):
        A[row, idxs] = 1.0
    return LinearConstraint(A, -np.inf, np.asarray(ub_list, dtype=float))


# # Profiling
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):

x0, fun, bounds, constraints = build_problem(
    Jstress,
    [
        (Jbsd,   -np.inf, 0),
        (Jbca,   -np.inf, 0),
        # (Jbcd, 0, 0),
        (Jvol,   -np.inf, 9.0),
        (Jcsrcc, -np.inf, 0),
        (Jcsrs,  -np.inf, 0),
    ],
    linear_constraint_fns=[_sum_dphis_constraint],
)
print("MAXITER =", MAXITER)
print("# free dofs =", len(x0))
print("# linear inequality constraints =", constraints[0].A.shape[0])
print("initial CSRVolume =", Jvol.J())
time_filament_1 = time.time()
res = minimize(
    fun, x0, jac=True, method="trust-constr",
    bounds=bounds,
    constraints=constraints,
    options={
        "maxiter": MAXITER,
        "gtol": 1e-3,
        "xtol": 1e-5,
        "barrier_tol": 1e-5,
        "verbose": 2,
    },
)
time_filament_2 = time.time()
print("time", time_filament_2 - time_filament_1)
print("res ", res)
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
