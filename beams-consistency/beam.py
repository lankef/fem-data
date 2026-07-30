from simsopt import load
import time
import numpy as np

Jstress = load('Jstress.json')[0]
Jstress.save_run_vtu('./beam_run')
time1 = time.time()
Jstress.save_run_vtu('./beam_run')
time2 = time.time()
np.save('time', time2-time1)