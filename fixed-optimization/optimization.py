import json
import sys
import numpy as np
import jax.numpy as jnp
import time
from pathlib import Path
from scipy.optimize import minimize
from simsopt.objectives import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.field import (
    BiotSavart, Coil,
    coils_via_symmetries,
)
from simsopt.field.force import LpCurveForce
from simsopt.field.coil import RegularizedCoil
from simsopt.field.selffield import regularization_rect
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    SurfaceRZFourier,
    CurveSurfaceDistance,
    LinkingNumber,
    LpCurveCurvature
)
from simsopt.configs import get_data
from coil_fem.simsopt            import CoilFEMObjective

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from opt_utils import load_eq

# ----- FEM options -----

_OPTIONS_PATH = _ROOT / 'beam-options.json'
opts = json.load(open(_OPTIONS_PATH))

mesh_options = opts['mesh_options']
# Fixed-family override: thermal contraction + real gravity (beams preset has 0).
material_options = {**opts['material_options'], 'itc': 0.0029}
gravity_options = {'g_vec': [0.0, 0.0, -9.80665]}
problem_options = opts['problem_options']
physics_options = opts['physics_options']
fixed_clamp_options = opts['fixed_clamp_options']

# Backward-compatible aliases: existing notebooks import these names.
material_options_const_temp = material_options
material_options_variable_temp = material_options

# ----- Loading coils -----

coil_per_half_fp = 5
Lp = 2
curves, currents, axis, nfp, bs = get_data('w7x', coil_order=20, points_per_period=2)
base_curves = curves[:coil_per_half_fp]
base_currents = currents[:coil_per_half_fp]

_WOUT_PATH = Path(__file__).resolve().parent / 'wout.nc'
eq, Bnormal_plasma, plasma_surface_vc, vc = load_eq(str(_WOUT_PATH))

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


def run_filament_free(
        base_curves, base_currents, 
        plasma_surface, Bnormal_plasma, 
        MAXITER, force_mode,
        support_type, support_kwargs
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
    print('Setting normalized flux target to this value.')
    FLUX_NORM_TARGET = Jf_norm.J()
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
        base_coils = [Coil(c, I) for c, I in zip(base_curves, base_currents)]
        coil_support = support_type(
            base_coils,
            nfp=plasma_surface.nfp,
            stellsym=plasma_surface.stellsym,
            **support_kwargs,
        )
        Jstress = CoilFEMObjective(
            coil_support,
            metrics          = ('l2_von_mises',),
            metric_weights   = (1.,),
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