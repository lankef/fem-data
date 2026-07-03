import json
import numpy as np
import jax.numpy as jnp
import time
from pathlib import Path
from scipy.optimize import minimize
from simsopt.objectives import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.field import (
    BiotSavart, Current,
    coils_via_symmetries,
)
from simsopt.field.force import LpCurveForce
from simsopt.field.coil import RegularizedCoil
from simsopt.field.selffield import regularization_rect
from simsopt import load, save
from simsopt.mhd import Vmec
from simsopt.mhd.virtual_casing import VirtualCasing
from simsopt.geo import (
    plot, CurveXYZFourier,
    CurveLength, CurveCurveDistance,
    SurfaceRZFourier,
    CurveSurfaceDistance,
    LinkingNumber,
    create_equally_spaced_curves,
    curves_to_vtk,
    LpCurveCurvature
)
from simsopt.configs import get_data
from simsopt.geo import plot
from coilforce.simsopt_bridge    import CoilFEMObjective
from coilforce.support           import CoilSupportDiscrete, CoilSupportTopBottom

# ----- FEM options -----

# Rectangular 100 mm × 50 mm cross-section (half-widths in metres).
# A single dict is broadcast to all base coils automatically.
mesh_options = dict(
    shape        = 'rect',
    w1           = 0.2,   # 0.20 m half-width
    w2           = 0.2,   # 0.20 m half-width
    frame        = 'rmf',
    aspect_ratio = 1.0,   # aim for cubic elements
    mesh_type    = 'TET10',
)

# Winkler BC and linear-solver settings.
problem_options = dict(
    winkler_k      = 1e10,    # Winkler spring stiffness [N/m³]
    solver         = 'cudss', # A CPU sparse solver. Future version 
    adjoint_solver = 'cudss', # will support cuSparse GPU sparse solvers.
)

# Material settings
# ─────────────────────────────────────────────────────────────────────────────
# Material, thermal-contraction and gravity parameters are centralised in
# ``fem-data/properties.json`` (repo root) and loaded here so every script uses
# the same values + literature references.  W7-X coil casings / support
# structure are AISI 316LN austenitic stainless steel; values (E, nu, density,
# 293→4 K integral thermal contraction) are from Foussat et al. 2013 (see
# properties.json).
# To disable thermal or gravity, edit properties.json (set gravity.enabled=false
# or drop the itc key); no code changes needed.
# ─────────────────────────────────────────────────────────────────────────────
_PROPERTIES_PATH = Path(__file__).resolve().parent.parent / 'properties.json'
with open(_PROPERTIES_PATH) as _f:
    _PROPERTIES = json.load(_f)

_mat = _PROPERTIES['material']
_grav = _PROPERTIES.get('gravity', {})

# Elastic + thermal material options forwarded to CoilFEMObjective.
material_options = dict(
    E       = float(_mat['E_Pa']),
    nu      = float(_mat['nu']),
    density = float(_mat['density_kg_m3']),
    itc     = float(_mat['itc']),   # integral thermal contraction ΔL/L; eps_th = -itc·I
)

# Gravity body-force options (None disables the gravity load).
if _grav.get('enabled', False):
    gravity_options = dict(
        density = float(_mat['density_kg_m3']),
        g_vec   = tuple(float(c) for c in _grav['g_vec_m_s2']),
    )
else:
    gravity_options = None

# Backward-compatible aliases: existing notebooks import these names.  Both now
# point at the single JSON-sourced dict (thermal itc included).
material_options_const_temp = material_options
material_options_variable_temp = material_options

# Load curves from lists of arrays containing x, y, and z.
def simsopt_curves_from_xyz(
    contour_X,
    contour_Y,
    contour_Z, 
    order=None, ppp=20):
    num_coils = len(contour_X)
    try:     
        from simsopt.geo import CurveXYZFourier
    except:
        raise ImportError('Simsopt is required to use the coil-cutting features.')
    # Calculating order
    if not order:
        order=float('inf')
        for i in range(num_coils):
            xArr = contour_X[i]
            yArr = contour_Y[i]
            zArr = contour_Z[i]
            for x in [xArr, yArr, zArr]:
                if len(x)//2<order:
                    order = len(x)//2
    
    coils = [CurveXYZFourier(order*ppp, order) for i in range(num_coils)]
    # Compute the Fourier coefficients for each coil
    for ic in range(num_coils):
        xArr = contour_X[ic]
        yArr = contour_Y[ic]
        zArr = contour_Z[ic]

        # Compute the Fourier coefficients
        dofs=[]
        for x in [xArr, yArr, zArr]:
            dof_i = ifft_simsopt(x, order)
            dofs.append(dof_i)

        coils[ic].local_x = np.concatenate(dofs)
    return coils

# ----- Loading coils -----

coil_per_half_fp = 5
Lp = 2
curves, currents, axis, nfp, bs = get_data('w7x', coil_order=20, points_per_period=2)
base_curves = curves[:coil_per_half_fp]
base_currents = currents[:coil_per_half_fp]

# ----- Loading equilibrium -----

n_phi = 25        # half fp like in virtual casing convention
n_theta = 50
vc_src_nphi = 40  # half fp like in virtual casing convention
vc_src_ntheta = 80

def load_eq(file_name):
    eq = Vmec(file_name, keep_all_files=True)
    vc = VirtualCasing.from_vmec(
        file_name,
        src_nphi=vc_src_nphi,
        src_ntheta=vc_src_ntheta,
        trgt_nphi=n_phi,
        trgt_ntheta=n_theta,
    )
    # This is a vacuum case!
    Bnormal_plasma = jnp.zeros_like(vc.B_external_normal)
    plasma_surface_vc = type(eq.boundary)(
        nfp=eq.boundary.nfp,
        stellsym=eq.boundary.stellsym,
        mpol=eq.boundary.mpol, ntor=eq.boundary.ntor,
        quadpoints_phi=np.linspace(0, 1/2/eq.boundary.nfp, n_phi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, n_theta, endpoint=False),
    )
    plasma_surface_vc.set_dofs(eq.boundary.get_dofs())
    return eq, Bnormal_plasma, plasma_surface_vc, vc

eq, Bnormal_plasma, plasma_surface_vc, vc = load_eq('wout.nc')

# ----- Loading constant parameters -----
# Setting targets
Jf_norm = SquaredFlux(
    plasma_surface_vc,
    bs,
    target=Bnormal_plasma,
    definition='normalized',
)
FLUX_NORM_TARGET = Jf_norm.J()

# Coil-coil distance
coils = coils_via_symmetries(base_curves, base_currents, plasma_surface_vc.nfp, True)
curves_for_ccd = [c.curve for c in coils]
Jccdist_init = CurveCurveDistance(curves_for_ccd, 0)
CC_TARGET = Jccdist_init.shortest_distance()

# Coil-surface distance
Jcsdist_init = CurveSurfaceDistance(base_curves, plasma_surface_vc, 0)
CP_TARGET = Jcsdist_init.shortest_distance()

# Curvature 
CURVATURE_TARGET = np.max([c.kappa() for c in base_curves])

# ---- Optimization parameters ----
# To be increased for actual scan
MAXFUN = 1e7
CURVATURE_WEIGHT = 1000
FLUX_WEIGHT = 500000
FORCE_WEIGHT = 5/1e5
LINK_WEIGHT = 10
Lp = 2  # p of Lp curve curvature
CC_WEIGHT = 100
CS_WEIGHT = 100
LENGTH_WEIGHT = 200
FLUX_NORM_TARGET = 5e-4
MAXITER = 5
STRESS_WEIGHT = 1e-18

# ---- Fourier continuation parameters ----
# Start from circular coils and progressively increase the Fourier order of the
# base curves, re-optimizing at each step (modelled after
# quasi_single_stage/filaments/shared.py:filament_study).
INIT_ORDER = 4          # Fourier order of the initial circular coils
ORDER_INCREMENT = 2     # order added to the base curves each continuation step
CONT_STEPS = 3          # number of continuation steps
CIRCLE_RADIUS_FACTOR = 3  # circular coil radius R1 = factor * plasma minor radius


def increase_base_curve_order(base_curves, coils_per_half_field_period, increment):
    order_in = base_curves[0].order
    contour_X = []
    contour_Y = []
    contour_Z = []
    for curve_i in base_curves:
        gamma_i = curve_i.gamma()
        contour_X.append(gamma_i[:, 0])
        contour_Y.append(gamma_i[:, 1])
        contour_Z.append(gamma_i[:, 2])
    base_curves_out = simsopt_curves_from_xyz(
        contour_X,
        contour_Y,
        contour_Z,
        order=order_in + increment,
        ppp=20,
    )
    return base_curves_out
def run_filament_free(
        base_curves, base_currents, 
        plasma_surface, Bnormal_plasma, 
        MAXITER, force_mode,
        support_type, support_kwargs,
        flux_norm_target=None,
    ):

    import logging
    logging.getLogger('jax_fem').setLevel(logging.WARNING)
    # --------------------------------------
    coils = coils_via_symmetries(base_curves, base_currents, plasma_surface.nfp, True)

    curves_for_ccd = [c.curve for c in coils]

    bs = BiotSavart(coils)
    print('Btarget should have shape', plasma_surface.normal().shape[:2])
    print('Btarget has shape', Bnormal_plasma.shape)
    bs.set_points(plasma_surface.gamma().reshape((-1, 3)))
    Jf_norm = SquaredFlux(
        plasma_surface,
        bs,
        target=Bnormal_plasma,
        definition='normalized',
    )
    Jf_actual = SquaredFlux(
        plasma_surface,
        bs,
        target=Bnormal_plasma,
        definition='quadratic flux',
    )
    print('Initial normalized flux', Jf_norm.J())
    if flux_norm_target is None:
        FLUX_NORM_TARGET = Jf_norm.J()
        print('Setting normalized flux target to this value.')
    else:
        FLUX_NORM_TARGET = flux_norm_target
        print('Using fixed normalized flux target', FLUX_NORM_TARGET)
    Jf_norm_cons = QuadraticPenalty(Jf_norm, FLUX_NORM_TARGET, 'max')
    Jls = [
        QuadraticPenalty(
            CurveLength(c),
            CurveLength(c).J(),
            "max", 
        ) for c in base_curves
    ]
    print('Initial lengths', [CurveLength(c).J() for c in base_curves])
    print('Setting length targets to these values.')

    if force_mode:
        regularized_coils = [RegularizedCoil(
            c.curve, c.current, regularization_rect(0.54, 0.54)
        ) for c in coils]
        Jforce = LpCurveForce(
            target_coils=regularized_coils[:coil_per_half_fp],
            source_coils_coarse=regularized_coils,
        )
        print('Optimizing L2 force')
    else:
        Jstress = CoilFEMObjective(
            base_curves      = base_curves,
            base_currents    = base_currents,
            base_supports    = [support_type(**support_kwargs)
                                for _ in base_curves],
            metrics          = ('l2_von_mises',),
            metric_weights   = (1.,),
            nfp              = plasma_surface.nfp,
            stellsym         = plasma_surface.stellsym,
            mesh_options     = mesh_options,
            material_options = material_options,
            gravity_options  = gravity_options,
            problem_options  = problem_options,
        )
        print('Optimizing max Von Mises')
    Jccdist_actual = CurveCurveDistance(curves_for_ccd, CC_TARGET)
    Jccdist = Jccdist_actual * (1 / CC_TARGET)
    print('Begin ----------')
    plasma_surface_cs = SurfaceRZFourier(
        nfp=plasma_surface.nfp,
        stellsym=plasma_surface.stellsym,
        mpol=plasma_surface.mpol,
        ntor=plasma_surface.ntor,
        quadpoints_phi=np.arange(64)/64,
        quadpoints_theta=np.arange(64)/64,
    )
    plasma_surface_cs.set_dofs(plasma_surface.get_dofs())
    Jcsdist_actual = CurveSurfaceDistance(base_curves, plasma_surface_cs, CP_TARGET)
    Jcsdist = Jcsdist_actual * (1 / plasma_surface.minor_radius())
    linkNum = LinkingNumber(curves_for_ccd)
    Jcs = [
        LpCurveCurvature(
            c, Lp, threshold=CURVATURE_TARGET
        )*(plasma_surface.minor_radius()**Lp) for c in base_curves
    ]
    JF = (
        FLUX_WEIGHT ** 2 * Jf_norm_cons
        + CC_WEIGHT * Jccdist
        + CS_WEIGHT * Jcsdist
        + CURVATURE_WEIGHT * sum(Jcs)
        + LENGTH_WEIGHT * sum(Jls)
        + LINK_WEIGHT * linkNum
    )
    if force_mode:
        JF = JF + FORCE_WEIGHT * Jforce
    else:
        JF = JF + STRESS_WEIGHT * Jstress
        
    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        return J, grad

    dofs = JF.x

    print('MAXITER =', MAXITER)
    time_filament_1 = time.time()
    res = minimize(
        fun, dofs, jac=True, method='L-BFGS-B',
        options={
            'maxiter': MAXITER,
            'maxcor': 300,
            'maxfun': MAXFUN,
            'maxls': 60,
        },
        tol=1e-10,
    )
    time_filament_2 = time.time()
    filament_time = time_filament_2 - time_filament_1
    print('Filament time', filament_time)
    print('Normalized flux', Jf_norm.J())
    if force_mode:
        print('L2 force', Jforce.J())
    else:
        print('Max (lse) von mises', Jstress.J())
    print('Lengths', [j.J() for j in Jls])
    print('Max curvatures', [jnp.max(jnp.abs(c.kappa())) for c in base_curves])
    print('Final value of terms')
    print('    FLUX_WEIGHT**2 * Jf_norm_cons ', FLUX_WEIGHT**2 * Jf_norm_cons.J())
    print('    + CC_WEIGHT * Jccdist         ', CC_WEIGHT * Jccdist.J())
    print('    + CS_WEIGHT * Jcsdist         ', CS_WEIGHT * Jcsdist.J())
    if force_mode:
        print('    + FORCE_WEIGHT * Jforce       ', FORCE_WEIGHT * Jforce.J())
    else:
        print('    + STRESS_WEIGHT * Jstress     ', STRESS_WEIGHT * Jstress.J())
    print('    + CURVATURE_WEIGHT * sum(Jcs) ', CURVATURE_WEIGHT * sum(Jcs).J())
    print('    + LINK_WEIGHT * linkNum       ', LINK_WEIGHT * linkNum.J())
    print('    + LENGTH_WEIGHT * sum(Jls)    ', (LENGTH_WEIGHT * sum(Jls)).J())
    print('-----------------------------------------')
    print('res', res)
    if force_mode:
        return (
            coils, curves_for_ccd, res, filament_time,
            Jf_norm, Jf_actual, 
            Jccdist, Jccdist_actual,
            Jcsdist, Jcsdist_actual,
            Jforce,
            Jls, linkNum,
        )
    else:
        return (
            coils, curves_for_ccd, res, filament_time,
            Jf_norm, Jf_actual, 
            Jccdist, Jccdist_actual,
            Jcsdist, Jcsdist_actual,
            Jstress,
            Jls, linkNum,
        )


def run_continuation(
        base_currents,
        plasma_surface, Bnormal_plasma,
        MAXITER, force_mode,
        support_type, support_kwargs,
        fix_geometry=False,
    ):
    """Fourier-order continuation starting from circular coils.

    The base coils are initialized as circles of radius
    ``CIRCLE_RADIUS_FACTOR * plasma_surface.minor_radius()`` at Fourier order
    ``INIT_ORDER``.  The problem is re-optimized ``CONT_STEPS`` times, raising the
    curve order by ``ORDER_INCREMENT`` between steps.  Modelled after
    ``quasi_single_stage/filaments/shared.py:filament_study``.
    """
    R0 = plasma_surface.major_radius()
    R1 = CIRCLE_RADIUS_FACTOR * plasma_surface.minor_radius()
    print('Circular coil init: R0 =', R0, 'R1 =', R1)
    base_curves = create_equally_spaced_curves(
        coil_per_half_fp, plasma_surface.nfp, stellsym=True,
        R0=R0, R1=R1, order=INIT_ORDER,
    )

    result = None
    for step in range(CONT_STEPS):
        print('======================================================')
        print('Continuation step', step, '/ order', base_curves[0].order)
        if fix_geometry:
            for c in base_curves:
                c.fix_all()
        result = run_filament_free(
            base_curves=base_curves,
            base_currents=base_currents,
            plasma_surface=plasma_surface,
            Bnormal_plasma=Bnormal_plasma,
            MAXITER=MAXITER,
            force_mode=force_mode,
            support_type=support_type,
            support_kwargs=support_kwargs,
            flux_norm_target=FLUX_NORM_TARGET,
        )
        if step < CONT_STEPS - 1:
            base_curves = increase_base_curve_order(
                base_curves, coil_per_half_fp, ORDER_INCREMENT
            )
    return result