from optimization import (
    load_eq,
    run_filament_free,
    beam_support_kwargs,
)

# Loading config from simsopt
from simsopt.configs import get_data
from simsopt.geo import plot
from coil_fem.simsopt import CoilSupportBeams
from simsopt import save
import time
import numpy as np

eq, Bnormal_plasma, plasma_surface_vc, vc = load_eq('../fixed-continuation/wout.nc')
# Same base coils used to build beams-consistency/Jstress.json
# (coil_order=10, points_per_period=16 -> 160 quadpoints/coil).
curves, currents, axis, nfp, bs = get_data('w7x', coil_order=10, points_per_period=16)

# list of CurveXYZFourierJAX, one per *base* coil (before symmetry expansion)
base_curves = curves[:5]
# 1-D JAX array of currents [A], shape (n_base,)
base_currents = currents[:5]
stellsym = True

import logging
logging.getLogger('jax_fem').setLevel(logging.WARNING)

# Fix every coil degree of freedom (geometry + current) so only the free
# CoilSupportBeams dofs (the beam network) are optimized.
for c in base_curves:
    c.fix_all()
for cur in base_currents:
    cur.fix_all()

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
    support_type=CoilSupportBeams,
    support_kwargs=beam_support_kwargs,
)
time2 = time.time()
np.save('misc_support', {
    'nit': res.nit,
    'nfev': res.nfev,
    'time': time2-time1, 
})

save(coils, filename='coils_support.json')
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
    filename='data_support.json'
)
save([Jstress], 'Jstress_support.json')
Jstress.save_run_vtu('support_run')
