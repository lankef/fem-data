import numpy as np
import jax.numpy as jnp
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
from coilforce.support           import CoilSupport

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
    winkler_k      = 1e10,      # Winkler spring stiffness [N/m³]
    solver         = 'umfpack', # A CPU sparse solver. Future version 
    adjoint_solver = 'umfpack', # will support cuSparse GPU sparse solvers.
)

# Material settings
material_options_const_temp = dict(
    E       = 200e9,   # Young's modulus [Pa]
    nu      = 0.30,    # Poisson ratio
    density = 7800.0,  # mass density [kg m⁻³]
)

material_options_variable_temp = dict(
    alpha             = 16.5e-6,  # CTE for 316 LN stainless steel [1/K]
    init_temperature  = 293.15,   # stress-free reference temperature [K]
    final_temperature = 4.0,      # superconducting service temperature [K]
) | material_options_const_temp

# Set the size of the support clamp (in other words, the region
# with non-zero Winkler spring coefficients).
# Clamp radius = 2 × coil half-width; sigmoid sharpness tuned to clamp_radius.
clamp_radius = 2 * max(mesh_options['w1'], mesh_options['w2'])
# The "steepness" of the sigmoid function at the edge of the clamp.
sigmoid_beta  = 20.0 / clamp_radius

class TopBottomSupport(CoilSupport):
    """Static soft-sphere support at the top and bottom of the coil centreline.

    Has no optimisable DOFs (``dofs={}``); ``clamp_radius`` and ``sigmoid_beta``
    are fixed constants.
    """

    def __init__(self, clamp_radius, sigmoid_beta):
        super().__init__(
            dofs={},
            constants={'clamp_radius': float(clamp_radius),
                       'sigmoid_beta': float(sigmoid_beta)},
        )

    @staticmethod
    def support_fn(surface_points, curve_jax, dofs, *, clamp_radius, sigmoid_beta):
        gamma  = curve_jax.gamma()                         # (n_quad, 3)
        top    = gamma[jnp.argmax(gamma[:, 2])]            # (3,) highest point
        bottom = gamma[jnp.argmin(gamma[:, 2])]            # (3,) lowest point

        # Safe norm: jnp.linalg.norm gradient is NaN at zero distance;
        # adding eps inside sqrt keeps the backward pass finite.
        d_top    = jnp.sqrt(jnp.sum((surface_points - top)**2,    axis=-1) + 1e-30)
        d_bottom = jnp.sqrt(jnp.sum((surface_points - bottom)**2, axis=-1) + 1e-30)

        # sigmoid(beta*(R-d)): ~1 inside sphere of radius clamp_radius, ~0 outside
        w_top    = jax.nn.sigmoid(sigmoid_beta * (clamp_radius - d_top))
        w_bottom = jax.nn.sigmoid(sigmoid_beta * (clamp_radius - d_bottom))
        return jnp.maximum(w_top, w_bottom)   # union of the two spheres


# ----- Loading coils -----

coil_per_half_fp = 5
Lp = 2
curves, currents, axis, nfp, bs = get_data('w7x', coil_order=20, points_per_period=2)
base_curves = curves[:coil_per_half_fp]
base_currents = currents[:coil_per_half_fp]

# ----- Loading equilibrium -----

def load_eq(file_name):
    eq = Vmec(file_name, keep_all_files=True)
    vc = VirtualCasing.from_vmec(
        file_name,
        src_nphi=vc_src_nphi,
        src_ntheta=vc_src_ntheta,
        trgt_nphi=n_phi,
        trgt_ntheta=n_theta,
    )
    Bnormal_plasma = vc.B_external_normal
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

# ----- Virtual casing resolution ----

n_phi = 25        # half fp like in virtual casing convention
n_theta = 50
vc_src_nphi = 40  # half fp like in virtual casing convention
vc_src_ntheta = 80

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
FLUX_WEIGHT = 10000
FORCE_WEIGHT = 5/1e5
LINK_WEIGHT = 10
Lp = 2  # p of Lp curve curvature
CC_WEIGHT = 100
CS_WEIGHT = 100
LENGTH_WEIGHT = 200
FLUX_NORM_TARGET = 5e-4
MAXITER = 5
STRESS_WEIGHT = 1e-10

def run_filament_free(
        base_curves, base_currents, 
        plasma_surface, Bnormal_plasma, 
        MAXITER, force_mode
    ):

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
        Jstress = CoilFEMObjective(
            base_curves      = base_curves,
            base_currents    = base_currents,
            base_supports    = [TopBottomSupport(clamp_radius, sigmoid_beta)
                                for _ in base_curves],
            metrics          = ('soft_max_von_mises',),
            metric_weights   = (1.,),
            nfp              = plasma_surface.nfp,
            stellsym         = plasma_surface.stellsym,
            mesh_options     = mesh_options,
            material_options = material_options_const_temp,
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
    print('L2 force', Jforce.J())
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
return (
    coils,
    curves_for_ccd,
    res,
    filament_time,
    Jf_norm,
    Jf_actual,
    Jccdist,
    Jccdist_actual,
    Jcsdist,
    Jcsdist_actual,
    Jforce,
    Jls,
    linkNum,
)