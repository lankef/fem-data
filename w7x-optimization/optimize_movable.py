from optimization import (
    load_eq, 
    run_filament_free, 
    increase_base_curve_order
)

# Loading config from simsopt
from simsopt.configs import get_data
from simsopt.geo import plot
from coilforce.support import CoilSupportDiscrete, CoilSupportTopBottom
from simsopt import save
import time
import numpy as np

eq, Bnormal_plasma, plasma_surface_vc, vc = load_eq('wout.nc')
curves, currents, axis, nfp, bs = get_data('w7x', coil_order=10, points_per_period=4)

# list of CurveXYZFourierJAX, one per *base* coil (before symmetry expansion)
base_curves = curves[:5]
# 1-D JAX array of currents [A], shape (n_base,)
base_currents = currents[:5]
stellsym = True

import logging
logging.getLogger('jax_fem').setLevel(logging.WARNING)

time1 = time.time()
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
    MAXITER=500, 
    force_mode=False,
    support_type=CoilSupportDiscrete, 
    support_kwargs={
        'clamp_num': 2,
        'clamp_radius': 0.3,
    }
)
time2 = time.time()
np.save('misc_movable', {
    'nit': res.nit,
    'time': time2-time1, 
})

save(coils, filename='coils_movable.json')
save(
    {
        'coils': coils,
        'Jf_norm': Jf_norm,
        'Jf_actual': Jf_actual,
        'Jccdist': Jccdist,
        'Jccdist_actual': Jccdist_actual,
        'Jcsdist': Jcsdist,
        'Jcsdist_actual': Jcsdist_actual,
        'Jls': Jls,
    },
    filename='data_movable.json'
)
save([Jstress], 'Jstress_movable.json')