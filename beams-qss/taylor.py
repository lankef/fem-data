# Centered-difference Taylor test of CoilFEMObjective.J/dJ on free
# CoilSupportBeamsSorted support DOFs (same setup as proto_cons.py).

# Import coil_fem before gmsh: loading gmsh first can pull the module
# anaconda libstdc++ ahead of the conda env's, breaking basix.
import coil_fem  # noqa: F401
from coil_fem.simsopt import CoilSupportBeamsSorted, CoilFEMObjective
import gmsh  # noqa: F401
from simsopt.configs import get_data
from simsopt.mhd import Vmec
from simsopt.field import Coil
import numpy as np

eps_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
seed = 1

# ----- Problem (support-only, coils fixed) -----

eq = Vmec('../fixed-continuation/wout.nc', keep_all_files=True)
coil_per_half_fp = 5
curves, currents, axis, nfp, bs = get_data(
    'w7x', coil_order=8, points_per_period=8
)
base_curves = curves[:coil_per_half_fp]
base_currents = currents[:coil_per_half_fp]

mesh_options = {
    'shape': 'rect',
    'w1': 0.2,
    'w2': 0.2,
    'frame': 'rmf',
    'aspect_ratio': 1.0,
    'mesh_type': 'TET10',
}
problem_options = {
    'solver': 'cudss',
    'adjoint_solver': 'cudss',
}
material_options = {
    'E': 205000000000,
    'nu': 0.3,
    'density': 8000,
    'itc': 0.0,
}
gravity_options = {'g_vec': (0, 0, 0)}
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
    'r_clamp': 1.73 * mesh_options['w1'] / 2,
    'n_clamp': 2,
    'E_coil': material_options['E'],
}
physics_options = {'type': 'elastic'}

base_coils = [Coil(c, I) for c, I in zip(base_curves, base_currents)]
coil_support = CoilSupportBeamsSorted(
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
    metrics=('l2_von_mises',),
    metric_weights=(1.,),
    mesh_options=mesh_options,
    material_options=material_options,
    gravity_options=gravity_options,
    problem_options=problem_options,
    physics_options=physics_options,
    coupling='monolithic',
)

for c in base_curves:
    c.fix_all()
for cur in base_currents:
    cur.fix_all()

print('# mesh node for all coils:', Jstress.n_nodes)
print('# mesh cell for all coils:', Jstress.n_cells)
print('# free dofs:', len(Jstress.x))
print('dof names:', Jstress.dof_names)

# ----- Taylor test -----

dofs = np.asarray(Jstress.x, dtype=float)
h = np.random.default_rng(seed).uniform(size=dofs.shape)


def fun(x):
    Jstress.x = x
    return Jstress.J(), Jstress.dJ()


J0, dJ0 = fun(dofs)
dJ0 = np.asarray(dJ0, dtype=float).reshape(-1)
dJh = float(np.dot(dJ0, h))
print(f'J0  = {J0}')
print(f'dJh = {dJh}')
print(f'|dJ| = {np.linalg.norm(dJ0)}')
print()
print(f"{'eps':>10} {'centered diff':>18} {'err':>18} {'err/eps^2':>18} {'|err|/|dJh|':>14}")

prev_err = None
for eps in eps_list:
    J1, _ = fun(dofs + eps * h)
    J2, _ = fun(dofs - eps * h)
    fd = (J1 - J2) / (2 * eps)
    err = fd - dJh
    rel = abs(err) / max(abs(dJh), 1.0)
    print(f'{eps:10.1e} {fd:18.10e} {err:18.10e} {err / eps**2:18.10e} {rel:14.6e}')
    if prev_err is not None:
        ratio = err / prev_err if prev_err != 0 else float('nan')
        print(
            f'    (err ratio vs previous eps: {ratio:.3f}, '
            f'expect ~1e-2 if O(eps^2); ~1 if systematic bias)'
        )
    prev_err = err

print('Done.')
