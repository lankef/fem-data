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

from coil_fem.simsopt import CoilSupportBeamsSorted
from coil_fem.simsopt import CoilFEMObjective
import gmsh
from simsopt.configs import get_data
from simsopt.mhd import Vmec
from simsopt import save
import json
import numpy as np
import jax
import time
from collections import defaultdict
from pathlib import Path
from simsopt.field import Coil
from scipy.optimize import minimize, Bounds, LinearConstraint

# Loading the W7-X standard configuration 
# plasma surface. wout file comes from Landreman's
# VMEC equilibrium archive: 
# https://github.com/landreman/vmec_equilibria/blob/master/W7-X/Standard/
eq = Vmec('../fixed-continuation/wout.nc', keep_all_files=True)
# Adjusting resolution and taking a half-field-period
n_phi = 25
n_theta = 50
MAXITER = 1000
MAXFUN = 10000
plasma_surface = type(eq.boundary)(
    nfp=eq.boundary.nfp, stellsym=eq.boundary.stellsym,
    mpol=eq.boundary.mpol, ntor=eq.boundary.ntor,
    quadpoints_phi=np.linspace(0, 1/2/eq.boundary.nfp, n_phi, endpoint=False),
    quadpoints_theta=np.linspace(0, 1, n_theta, endpoint=False),
)
plasma_surface.set_dofs(eq.boundary.get_dofs())

# Loading the W7-X coils.
coil_per_half_fp = 5
# The number of quadpoints here controls the mesh
# density in coil-fem. The meshing routine will
# try choose the cell number in the cross section 
# to make the aspect ratio close to one. Please see 
# the next sections for details.  
curves, currents, axis, nfp, bs = get_data(
    'w7x', coil_order=8, points_per_period=12
)
base_curves = curves[:coil_per_half_fp]
base_currents = currents[:coil_per_half_fp]

# ----- FEM / support options -----

_OPTIONS_PATH = Path(__file__).resolve().parent.parent / 'beam-options.json'
opts = json.load(open(_OPTIONS_PATH))
mesh_options = opts['mesh_options']
material_options = opts['material_options']
gravity_options = opts['gravity_options']
problem_options = opts['problem_options']
physics_options = opts['physics_options']
beam_options = opts['beam_options']
fixed_clamp_options = opts['fixed_clamp_options']
mesh_scale = 0.5

# ----- Defining optimizable ----- 

# One support object covers the whole base coilset
base_coils = [Coil(c, I) for c, I in zip(base_curves, base_currents)]
coil_support = CoilSupportBeamsSorted(
    base_coils=base_coils,
    nfp=eq.boundary.nfp,
    stellsym=eq.boundary.stellsym,
    beam_options=beam_options,
    r_beam=opts['r_beam'],
    fixed_clamp_options=fixed_clamp_options,
    fixed_dof_names=tuple(opts['fixed_dof_names']),
)

# The Simsopt wrapper for a differentiable FEM problem.
# It behaves like a simsopt objective.
Jstress = CoilFEMObjective(
    coil_support,
    metrics          = ('l2_von_mises',),
    metric_weights   = (1.,),
    mesh_options     = mesh_options,
    material_options = material_options,
    gravity_options  = gravity_options,
    problem_options  = problem_options,
    physics_options  = physics_options,
    coupling         = opts['coupling'],
)
save([Jstress], 'Jstress_init.json')
print('# mesh node for all coils:', Jstress.n_nodes)
print('# mesh cell for all coils:', Jstress.n_cells)
# Viusualizing the mesh nodes and support clamps before optimization
Jstress.save_support_vtu('supports_init')

# ----- Optimization -----

# Fix every coil degree of freedom (geometry + current) so only the free
# CoilSupportBeamsSorted dofs (the beam network) are optimized.
for c in base_curves:
    c.fix_all()
for cur in base_currents:
    cur.fix_all()
    
def fun(dofs):
    Jstress.x = dofs
    J = Jstress.J()
    grad = Jstress.dJ()
    return J, grad


def _sum_dphis_constraint(dof_names):
    """Linear inequalities sum_j dphis*[i][j] <= 1 for each coil/group.

    Applies to free DOFs named ``dphis``, ``dphis_start_cc``, and
    ``dphis_end_cc`` (simsopt names like ``...:dphis_start_cc(i,j)``).
    """
    keys = ('dphis', 'dphis_start_cc', 'dphis_end_cc')
    groups = defaultdict(list)
    for j, name in enumerate(dof_names):
        # simsopt: "CoilSupportBeamsSorted1:dphis_start_cc(0,3)"
        local = name.split(':', 1)[-1]
        key = local.split('(', 1)[0]
        if key not in keys:
            continue
        i_coil = int(local.split('(', 1)[1].split(',', 1)[0])
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

dofs = Jstress.x
lb, ub = Jstress.bounds
bounds = Bounds(lb, ub)
constraints = [_sum_dphis_constraint(Jstress.dof_names)]
print('MAXITER =', MAXITER)
print('# free dofs =', len(dofs))
print('# linear inequality constraints =', constraints[0].A.shape[0])
time_filament_1 = time.time()
res = minimize(
    fun, dofs, jac=True, method='trust-constr',
    bounds=bounds,
    constraints=constraints,
    options={
        'maxiter': MAXITER,
        'gtol': 1e-10,
        'xtol': 1e-10,
        'barrier_tol': 1e-10,
        'verbose': 2,
    },
)
save([Jstress], 'Jstress_fin.json')
time_filament_2 = time.time()
Jstress.save_run_vtu('fin_run')
print('time', time_filament_2 - time_filament_1)
print('res ', res)
