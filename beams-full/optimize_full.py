# This protototype py optimizes support only fixing 
# coil dofs. Its goal is to investigate how challenging 
# support optimization is. If support optimization is 
# not too bad, then we can handle support like an optimization
# subproblem, because handling support and coils together
# can cause coils to converge prematurely. 
#
# Constrained variant: CoilSupportBeamsSorted + scipy trust-constr
# with box bounds from Jstress.bounds and linear inequalities
# sum_j dphis*[i][j] <= 1 per coil/group for dphis, dphis_start_cc,
# and dphis_end_cc.

from coil_fem.simsopt import (
    CoilSupportBeamsSorted, 
    BeamSurfaceDistance, 
    BeamCurveAngle,
    BeamCurveDistance,
    CoilFEMObjective,
)
from simsopt.configs import get_data
from simsopt.mhd import Vmec
from simsopt import save
import json
import numpy as np
import jax
import time
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from simsopt.field import Coil, coils_via_symmetries, BiotSavart
from scipy.optimize import minimize, Bounds, LinearConstraint
from simsopt.objectives import SquaredFlux
from simsopt.field.force import LpCurveForce
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    SurfaceRZFourier,
    CurveSurfaceDistance,
    LinkingNumber,
    create_equally_spaced_curves,
    LpCurveCurvature
)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from opt_utils import (
    load_eq,
    ppp_for_target_quadpoints,
    increase_base_curve_order,
    build_problem,
)

# Loading the W7-X standard configuration 
# plasma surface. wout file comes from Landreman"s
# VMEC equilibrium archive: 
# https://github.com/landreman/vmec_equilibria/blob/master/W7-X/Standard/
# eq = Vmec("../fixed-continuation/wout.nc", keep_all_files=True)
_WOUT_PATH = str(Path(__file__).resolve().parent / "../fixed-continuation/wout.nc")
eq, Bnormal_plasma, plasma_surface, vc = load_eq(_WOUT_PATH)
# Adjusting resolution and taking a half-field-period
n_phi = 25
n_theta = 50
MAXITER = 1000
MAXFUN = 10000
Lp = 2
# plasma_surface = type(eq.boundary)(
#     nfp=eq.boundary.nfp, stellsym=eq.boundary.stellsym,
#     mpol=eq.boundary.mpol, ntor=eq.boundary.ntor,
#     quadpoints_phi=np.linspace(0, 1/2/eq.boundary.nfp, n_phi, endpoint=False),
#     quadpoints_theta=np.linspace(0, 1, n_theta, endpoint=False),
# )
plasma_surface.set_dofs(eq.boundary.get_dofs())
r_beam = 0.08
# w1_beam = 0.2
# w2_beam = 0.12
# t_beam = 0.03
fixed_dof_names = [
    "thetas_orientation_cc",
    # "w1_beam",
    # "w2_beam",
    # "t_beam",
    "r_beam"
]

# Loading the W7-X coils.
coil_per_half_fp = 5
# The number of quadpoints here controls the mesh
# density in coil-fem. The meshing routine will
# try choose the cell number in the cross section 
# to make the aspect ratio close to one. Please see 
# the next sections for details.  
curves, currents, axis, nfp, bs = get_data(
    "w7x", 
    coil_order=8, 
    points_per_period=12 # <<<<< SET RESOLUTION HERE
)
base_curves = curves[:coil_per_half_fp]
base_currents = currents[:coil_per_half_fp]
coils = coils_via_symmetries(base_curves, base_currents, plasma_surface.nfp, True)
curves_for_ccd = [c.curve for c in coils]

# ----- Optimization targets -----

# Coil-coil distance
Jccdist_init = CurveCurveDistance(curves_for_ccd, 0)
CC_TARGET = Jccdist_init.shortest_distance()

# Coil-surface distance
Jcsdist_init = CurveSurfaceDistance(base_curves, plasma_surface, 0)
CP_TARGET = Jcsdist_init.shortest_distance()

# Curvature 
CURVATURE_TARGET = np.max([c.kappa() for c in base_curves])

# ----- FEM / support options -----

_OPTIONS_PATH = Path(__file__).resolve().parent.parent / "beam-options.json"
opts = json.load(open(_OPTIONS_PATH))
mesh_options = opts["mesh_options"]
material_options = opts["material_options"]
gravity_options = opts["gravity_options"]
problem_options = opts["problem_options"]
physics_options = opts["physics_options"]
beam_options = opts["beam_options"]
# beam_options["cross_section_type"] = "hollow_rectangle"
fixed_clamp_options = opts["fixed_clamp_options"]
mesh_scale = 0.5

# ----- Defining optimizable ----- 

# One support object covers the whole base coilset
base_coils = [Coil(c, I) for c, I in zip(base_curves, base_currents)]
coil_support = CoilSupportBeamsSorted(
    base_coils=base_coils,
    nfp=eq.boundary.nfp,
    stellsym=eq.boundary.stellsym,
    beam_options=beam_options,
    # w1_beam=w1_beam,
    # w2_beam=w2_beam,
    # t_beam=t_beam,
    r_beam=r_beam,
    fixed_clamp_options={"enabled": False}, # fixed_clamp_options,
    fixed_dof_names=fixed_dof_names,
)

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

# ----- Beam-curve distance -----

target_bcd = r_beam + np.sqrt(mesh_options["w1"]**2 + mesh_options["w2"]**2)
# target_bcd = np.sqrt(w1_beam**2 + w2_beam**2) + np.sqrt(mesh_options["w1"]**2 + mesh_options["w2"]**2)
Jbcd = BeamCurveDistance(
    coil_support, 
    dead_length=target_bcd*2,
    minimum_distance=target_bcd*0.9,
)

# ----- Biot-Savart -----

Jf_norm = SquaredFlux(
    plasma_surface,
    bs,
    target=Bnormal_plasma,
    definition='normalized',
)
FLUX_NORM_TARGET = Jf_norm.J()

# ----- CC and CS distance -----

Jcc = CurveCurveDistance(curves_for_ccd, CC_TARGET)
Jcs = CurveSurfaceDistance(base_curves, plasma_surface, CP_TARGET)
curv_objs = [LpCurveCurvature(c, Lp, threshold=CURVATURE_TARGET) for c in base_curves]

# ----- Optimization -----

# Fix every coil degree of freedom (geometry + current) so only the free
# CoilSupportBeamsSorted dofs (the beam network) are optimized.
# for c in base_curves:
#     c.fix_all()
for cur in base_currents:
    cur.fix_all()

def _sum_dphis_constraint(dof_names):
    """Linear inequalities sum_j dphis*[i][j] <= 1 for each coil/group.

    Applies to free DOFs named ``dphis``, ``dphis_start_cc``, and
    ``dphis_end_cc`` (simsopt names like ``...:dphis_start_cc(i,j)``).
    """
    keys = ("dphis", "dphis_start_cc", "dphis_end_cc")
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

# # Profiling
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):

x0, fun, bounds, constraints = build_problem(
    Jstress,
    [
        (Jbsd,    0, 0),
        (Jbca,    0, 0),
        (Jbcd,    0, 0),
        (Jf_norm, 0, FLUX_NORM_TARGET),
        (Jcc,     0, 0),
        (Jcs,     0, 0),
    ] + [(c, 0, 0) for c in curv_objs],
    linear_constraint_fns=[_sum_dphis_constraint],
)
print("MAXITER =", MAXITER)
print("# free dofs =", len(x0))
print("# linear inequality constraints =", constraints[0].A.shape[0])
time_filament_1 = time.time()
res = minimize(
    fun, x0, jac=True, method="trust-constr",
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

