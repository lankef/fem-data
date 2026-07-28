from optimization import (
    load_eq,
    run_continuation,
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
# Same base-coil currents used to build beams-consistency/Jstress.json
# (coil_order=10, points_per_period=16 -> 160 quadpoints/coil).
curves, currents, axis, nfp, bs = get_data('w7x', coil_order=10, points_per_period=16)

# 1-D JAX array of currents [A], shape (n_base,)
base_currents = currents[:5]
stellsym = True

import logging
logging.getLogger('jax_fem').setLevel(logging.WARNING)

# Coil geometry starts from circles (via run_continuation) and is optimized
# together with the currents and the CoilSupportBeams beam-network dofs.
time1 = time.time()
(
    coils, curves_for_ccd, res, filament_time,
    Jf_norm, Jf_actual, 
    Jccdist, Jccdist_actual,
    Jcsdist, Jcsdist_actual,
    Jstress,
    Jls, linkNum,
), nit_list, nfev_list = run_continuation(
    base_currents=base_currents, 
    plasma_surface=plasma_surface_vc, 
    Bnormal_plasma=Bnormal_plasma, 
    MAXITER=500, 
    force_mode=False,
    support_type=CoilSupportBeams,
    support_kwargs=beam_support_kwargs,
)
time2 = time.time()
np.save('misc_full', {
    'nit': res.nit,
    'nfev': res.nfev,
    'nit_list': nit_list, 
    'nfev_list': nfev_list,
    'time': time2-time1, 
})

save(coils, filename='coils_full.json')
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
    filename='data_full.json'
)
save([Jstress], 'Jstress_full.json')
