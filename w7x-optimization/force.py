from optimization import (
    load_eq, 
    run_filament_free, 
    increase_base_curve_order
)

# Loading config from simsopt
from simsopt.configs import get_data
from simsopt.geo import plot

eq, Bnormal_plasma, plasma_surface_vc, vc = load_eq('wout.nc')
curves, currents, axis, nfp, bs = get_data('w7x', coil_order=20, points_per_period=4)

# list of CurveXYZFourierJAX, one per *base* coil (before symmetry expansion)
base_curves = curves[:5]
# 1-D JAX array of currents [A], shape (n_base,)
base_currents = currents[:5]
stellsym = True

import logging
logging.getLogger('jax_fem').setLevel(logging.WARNING)

(
    coils, curves_for_ccd, res, filament_time,
    Jf_norm, Jf_actual, 
    Jccdist, Jccdist_actual,
    Jcsdist, Jcsdist_actual,
    Jstress,
    Jls, linkNum,
) = run_filament_free(
    base_curves=base_curves, 
    base_currents=base_currents, 
    plasma_surface=plasma_surface_vc, 
    Bnormal_plasma=Bnormal_plasma, 
    MAXITER=1000, 
    force_mode=True,
    support_type=None, 
    support_kwargs={}
)


save(
    {
        'coils': coils,
        'Jf_norm': Jf_norm,
        'Jf_actual': Jf_actual,
        'Jccdist': Jccdist,
        'Jccdist_actual': Jccdist_actual,
        'Jcsdist': Jcsdist,
        'Jcsdist_actual': Jcsdist_actual,
        'Jstress': Jstress,
        'Jls': Jls,
    },
    filename='force.json'
)