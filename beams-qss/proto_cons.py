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
import numpy as np
import jax
import time
from collections import defaultdict
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
MAXITER = 50
MAXFUN = 500
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
    'w7x', coil_order=8, points_per_period=8
)
base_curves = curves[:coil_per_half_fp]
base_currents = currents[:coil_per_half_fp]

# ----- Loading material properties -----

# Rectangular 100 mm × 50 mm cross-section (half-widths in metres).
# This single dict will be broadcast to all base coils automatically.
# The mesh density is controlled by the quadpoint number in base_curve.
mesh_options = {
    'shape'       : 'rect',  # Rectangular cross sections
    'w1'          : 0.2,     # 0.20 m full-width
    'w2'          : 0.2,     # 0.20 m full-width
    'frame'       : 'rmf',   # Cross sections oriented to a rotation-minimizing frame
    'aspect_ratio': 1.0,     # Aim for cubic elements
    'mesh_type'   : 'TET10', # Quadratic tetrahedral mesh and elements.
}

# Problem and solver options.
problem_options = {
    'solver'         : 'cudss', # A GPU direct sparse solver. Strongly 
    'adjoint_solver' : 'cudss', # recommended.
}

# Material options. Here, we assumes the coils are composed
# uniformly of 316LN stainless steel. Material properties cited 
# from DOI: 10.1016/j.nucengdes.2013.08.016
material_options = {
    'E'      : 205000000000, # Young's modulus (Pa)
    'nu'     : 0.3,          # Poisson's ratio (dimless)
    'density': 8000,         # Density (kg/m³)
    'itc'    : 0.0,          # Integral thermal contraction (ΔL/L, dimless)
                             # Was 0.0029, turned off because it's not supported 
                             # yet in beam network
}

# Gravity options.
# This is a stellarator-symmetric case.
# For now gravity won't be correct.
gravity_options = {
    'g_vec': (0, 0, 0), # g vector in (x, y, z)   (m/s²)
}
# Meshing scale in msh and dolfinx validation case
mesh_scale = 0.5

# ----- Support options ----- 

beam_options = {
    'n_beam_cc': 4,     # 3 coil-coil beams
    'n_beam_cf': 0,     # No coil-foundation beams for simplicity
    'E': material_options['E'],  # 316LN's Young's modulus (Pa)
    'nu': material_options['nu'],          # 316LN's Poisson's ratio (dimless)
    'cross_section_type': 'solid_circle',
    'attachment_type': 'direct',
    # k_clamp will be calculated automatically from the beams' stiffness 
    # based on their lengths and Young's modulus
}
# Fixed clamps have to be enabled when there are no CF beamsto prevent rigid body modes
fixed_clamp_options = {
    'enabled': True, 
    'r_clamp': 1.73 * mesh_options['w1']/2, # Diagonal of a cube with side length of w_1
    'n_clamp': 2,
    # k_clamp will be calculated automatically from the coil's stiffness 
    # based on their lengths and Young's modulus
    'E_coil': material_options['E'],  
}
physics_options = {
    'type': 'elastic',
}

# ----- Defining optimizable ----- 

# One support object covers the whole base coilset
base_coils = [Coil(c, I) for c, I in zip(base_curves, base_currents)]
coil_support = CoilSupportBeamsSorted(
    base_coils=base_coils,
    nfp=eq.boundary.nfp,
    stellsym=eq.boundary.stellsym,
    # ----- Beam info -----
    beam_options=beam_options,
    # dphis_start_cc # Default values.
    # dphis_end_cc # Default values.
    # dphis_start_cf # Default values.
    # x_foundation # Default values, no cf beams
    # thetas_orientation_cc # Default values. Circular cross sections.
    # thetas_orientation_cf # Default values.
    r_beam=0.06, # When using circular beams, the cross sec rad is a dof
                # and must be supplied as a kwarg. We fix it in this optimization.
    # ----- Fixed clamp info -----
    fixed_clamp_options=fixed_clamp_options,
    # ----- Simsopt info -----
    # Fixing beam network dofs. Some of these
    # will result in ill-posed problems if not fixed.
    # If a certain type of beam has n=0, then the 
    # associated dofs will be automatically fixed.
    fixed_dof_names=(
        # 'dphis_end_cc',
        # 'dphis_start_cc',
        # 'dphis_start_cf',
        'thetas_orientation_cc',
        # 'thetas_orientation_cf',
        # 'x_foundation',
        'r_beam',
    ),
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
    coupling         = 'monolithic',
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
print('time', time_filament_2 - time_filament_1)
print('res ', res)
