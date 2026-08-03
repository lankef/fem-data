# Taylor test for the support-only optimization in ./proto.py.
#
# proto.py was observed to "converge" after only ~30 function evaluations
# with no visible iteration progress. That's the signature of either a
# gradient that doesn't match the objective (L-BFGS-B's line search bails
# out almost immediately) or of a problem with very few free dofs that's
# already near a stationary point. This script isolates Jstress.J()/dJ()
# from the optimizer and checks the gradient directly via a standard
# centered-difference Taylor test, following the pattern used throughout
# simsopt (see examples/2_Intermediate/stage_two_optimization.py).
#
# Everything up to the optimization block is copied verbatim from proto.py
# so that Jstress and its dofs are set up identically.

from coil_fem.simsopt import CoilSupportBeams
from coil_fem.simsopt import CoilFEMObjective
import gmsh
from simsopt.configs import get_data
from simsopt.mhd import Vmec
import numpy as np
from simsopt.field import Coil

# Loading the W7-X standard configuration
# plasma surface. wout file comes from Landreman's
# VMEC equilibrium archive:
# https://github.com/landreman/vmec_equilibria/blob/master/W7-X/Standard/
eq = Vmec('../fixed-continuation/wout.nc', keep_all_files=True)
# Adjusting resolution and taking a half-field-period
n_phi = 25
n_theta = 50
plasma_surface = type(eq.boundary)(
    nfp=eq.boundary.nfp, stellsym=eq.boundary.stellsym,
    mpol=eq.boundary.mpol, ntor=eq.boundary.ntor,
    quadpoints_phi=np.linspace(0, 1/2/eq.boundary.nfp, n_phi, endpoint=False),
    quadpoints_theta=np.linspace(0, 1, n_theta, endpoint=False),
)
plasma_surface.set_dofs(eq.boundary.get_dofs())

# Loading the W7-X coils.
coil_per_half_fp = 5
curves, currents, axis, nfp, bs = get_data(
    'w7x', coil_order=8, points_per_period=8
)
base_curves = curves[:coil_per_half_fp]
base_currents = currents[:coil_per_half_fp]

# ----- Loading material properties -----

mesh_options = {
    'shape'       : 'rect',
    'w1'          : 0.2,
    'w2'          : 0.2,
    'frame'       : 'rmf',
    'aspect_ratio': 1.0,
    'mesh_type'   : 'TET10',
}

problem_options = {
    'solver'         : 'cudss',
    'adjoint_solver' : 'cudss',
}

material_options = {
    'E'      : 205000000000,
    'nu'     : 0.3,
    'density': 8000,
    'itc'    : 0.0,
}

gravity_options = {
    'g_vec': (0, 0, 0),
}
mesh_scale = 0.5

# ----- Support options -----

beam_options = {
    'n_beam_cc': 4,
    'n_beam_cf': 0,
    'E': material_options['E'],
    'nu': material_options['nu'],
    'cross_section_type': 'solid_circle',
    'attachment_type': 'direct',
}
fixed_clamp_options = {
    'enabled': True,
    'r_clamp': 1.73 * mesh_options['w1']/2,
    'n_clamp': 2,
    'E_coil': material_options['E'],
}
physics_options = {
    'type': 'elastic',
}

# ----- Defining optimizable -----

base_coils = [Coil(c, I) for c, I in zip(base_curves, base_currents)]
coil_support = CoilSupportBeams(
    base_coils=base_coils,
    nfp=eq.boundary.nfp,
    stellsym=eq.boundary.stellsym,
    beam_options=beam_options,
    r_beam=0.06,
    fixed_clamp_options=fixed_clamp_options,
    fixed_dof_names=(
        'thetas_orientation_cc',
        'r_beam',
    ),
)

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
print('# mesh node for all coils:', Jstress.n_nodes)
print('# mesh cell for all coils:', Jstress.n_cells)

# Fix every coil degree of freedom (geometry + current) so only the free
# CoilSupportBeams dofs (the beam network) are optimized, exactly as in
# proto.py.
for c in base_curves:
    c.fix_all()
for cur in base_currents:
    cur.fix_all()


def fun(dofs):
    Jstress.x = dofs
    J = Jstress.J()
    grad = Jstress.dJ()
    return J, grad


print("""
################################################################################
### Perform a Taylor test #####################################################
################################################################################
""")
f = fun
dofs = Jstress.x
print('# free dofs:', dofs.shape, flush=True)
print('dof names  :', Jstress.dof_names, flush=True)

np.random.seed(1)
h = np.random.uniform(size=dofs.shape)

J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
print('J0  =', J0, flush=True)
print('dJh =', dJh, flush=True)
print(flush=True)
print(f"{'eps':>10} {'centered diff':>18} {'err':>18} {'err/eps^2':>18}", flush=True)

prev_err = None
for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    fd = (J1 - J2) / (2*eps)
    err = fd - dJh
    ratio = err / eps**2
    print(f"{eps:10.1e} {fd:18.10e} {err:18.10e} {ratio:18.10e}", flush=True)
    if prev_err is not None:
        print(f"    (err ratio vs previous eps: {err/prev_err:.3f}, expect ~1e2 if O(eps^2))", flush=True)
    prev_err = err
