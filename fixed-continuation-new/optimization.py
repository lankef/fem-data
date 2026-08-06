import json
import sys
import numpy as np
from pathlib import Path
from simsopt.objectives import SquaredFlux
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
    create_equally_spaced_curves,
    LpCurveCurvature
)
from simsopt.configs import get_data
from coil_fem.simsopt            import CoilFEMObjective

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from opt_utils import (
    load_eq,
    ppp_for_target_quadpoints,
    increase_base_curve_order,
)

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

# ----- Loading coils -----

coil_per_half_fp = 5
Lp = 2
curves, currents, axis, nfp, bs = get_data('w7x', coil_order=20, points_per_period=2)
base_curves = curves[:coil_per_half_fp]
base_currents = currents[:coil_per_half_fp]

# Prefer local wout if present; else fall back to fixed-continuation's.
_WOUT_PATH = Path(__file__).resolve().parent / 'wout.nc'
if not _WOUT_PATH.is_file():
    _WOUT_PATH = _ROOT / 'fixed-continuation' / 'wout.nc'
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
Lp = 2  # p of Lp curve curvature

# ---- Fourier continuation parameters ----
# Start from circular coils and progressively increase the Fourier order of the
# base curves, re-optimizing at each step (modelled after
# quasi_single_stage/filaments/shared.py:filament_study).
INIT_ORDER = 4          # Fourier order of the initial circular coils
ORDER_INCREMENT = 2     # order added to the base curves each continuation step
CONT_STEPS = 3          # number of continuation steps
CIRCLE_RADIUS_FACTOR = 3  # circular coil radius R1 = factor * plasma minor radius

def run_filament_free(
        base_curves, base_currents, 
        plasma_surface, Bnormal_plasma, 
        MAXITER, force_mode,
        support_type, support_kwargs,
        mesh_options=mesh_options,
        material_options=material_options,
        gravity_options=gravity_options,
        problem_options=problem_options,
        physics_options=None,
        coupling=None,
    ):
    """Run one filament-free (stress/force-aware) coil optimization.

    ``mesh_options``/``material_options``/``gravity_options``/``problem_options``
    default to this module's own FEM settings (unchanged behaviour for
    existing callers).  ``physics_options``/``coupling`` are only forwarded to
    ``CoilFEMObjective`` when explicitly provided, so callers that don't pass
    them (this module's own scripts) see the exact same construction as
    before.  Other modules (e.g. ``beams-optimization/optimization.py``) can
    override any of these per call to reuse this function with a different
    FEM/support setup (e.g. ``CoilSupportBeams``).
    """

    import logging
    logging.getLogger('jax_fem').setLevel(logging.WARNING)
    # --------------------------------------
    coils = coils_via_symmetries(base_curves, base_currents, plasma_surface.nfp, True)

    curves_for_ccd = [c.curve for c in coils]

    # ----- Field error ------

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

    # ----- Curve lengths -----

    len_init = [CurveLength(c).J() for c in base_curves]
    print('Initial lengths', len_init, 'setting length targets to these values.')

    # ----- Force/stress ----- 

    if force_mode:
        regularized_coils = [RegularizedCoil(
            c.curve, c.current, regularization_rect(0.54, 0.54)
        ) for c in coils]
        Jopt = LpCurveForce(
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
        _fem_kwargs = dict(
            metrics          = ('l2_von_mises',),
            metric_weights   = (1.,),
            mesh_options     = mesh_options,
            material_options = material_options,
            gravity_options  = gravity_options,
            problem_options  = problem_options,
        )
        if physics_options is not None:
            _fem_kwargs['physics_options'] = physics_options
        if coupling is not None:
            _fem_kwargs['coupling'] = coupling
        Jopt = CoilFEMObjective(coil_support, **_fem_kwargs)
        print('Optimizing max Von Mises')
    
    # ----- Curve-curve distance ----- 

    Jccdist_actual = CurveCurveDistance(curves_for_ccd, CC_TARGET)

    # ----- Curve-plasma distance ----- 

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

    # ----- Linking number ----- 

    linkNum = LinkingNumber(curves_for_ccd)

    # ----- Curvature ----- 

    Jcs = [
        LpCurveCurvature(
            c, Lp, threshold=CURVATURE_TARGET
        )*(plasma_surface.minor_radius()**Lp) for c in base_curves
    ]
    
    


def run_continuation(
        base_currents,
        plasma_surface, Bnormal_plasma,
        MAXITER, force_mode,
        support_type, support_kwargs,
        fix_geometry=False,
        mesh_options=mesh_options,
        material_options=material_options,
        gravity_options=gravity_options,
        problem_options=problem_options,
        physics_options=None,
        coupling=None,
    ):
    """Fourier-order continuation starting from circular coils.

    The base coils are initialized as circles of radius
    ``CIRCLE_RADIUS_FACTOR * plasma_surface.minor_radius()`` at Fourier order
    ``INIT_ORDER``.  The problem is re-optimized ``CONT_STEPS`` times, raising the
    curve order by ``ORDER_INCREMENT`` between steps.  Modelled after
    ``quasi_single_stage/filaments/shared.py:filament_study``.

    ``mesh_options``/.../``coupling`` are forwarded to ``run_filament_free``
    verbatim each step (see its docstring); defaults reproduce this module's
    own behaviour unchanged.
    """
    R0 = plasma_surface.major_radius()
    R1 = CIRCLE_RADIUS_FACTOR * plasma_surface.minor_radius()
    print('Circular coil init: R0 =', R0, 'R1 =', R1)
    base_curves = create_equally_spaced_curves(
        coil_per_half_fp, plasma_surface.nfp, stellsym=True,
        R0=R0, R1=R1, order=INIT_ORDER,
        numquadpoints=INIT_ORDER * ppp_for_target_quadpoints(INIT_ORDER),
    )

    result = None
    nit_list = []
    nfev_list = []
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
            mesh_options=mesh_options,
            material_options=material_options,
            gravity_options=gravity_options,
            problem_options=problem_options,
            physics_options=physics_options,
            coupling=coupling,
            # flux_norm_target=FLUX_NORM_TARGET,
        )
        nit_list.append(result[2].nit)
        nfev_list.append(result[2].nfev)
        if step < CONT_STEPS - 1:
            base_curves = increase_base_curve_order(
                base_curves, ORDER_INCREMENT
            )
    return result, nit_list, nfev_list