import time
import numpy as np
from pathlib import Path
from simsopt import load
from coil_fem.io import to_full_body
for path in sorted(Path('.').glob('Jstress_*.json')):
    time1 = time.time()
    name = path.stem.removeprefix('Jstress_')  # e.g. Jstress_full.json -> full
    Jstress = load(str(path))[0]
    out_dir = Path(f'./{name}_dolfinx')
    out_dir.mkdir(parents=True, exist_ok=True)
    to_full_body(Jstress, path=f'./{name}_dolfinx/full_body_fields.vtu', mesh_scale=0.6)
    time2 = time.time()
    np.save(f'time_export_{name}', time2-time1)