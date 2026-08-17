import time
import numpy as np
from pathlib import Path
from simsopt import load
from coil_fem.io import to_full_body

for path in sorted(Path('.').glob('*_Jstress.json')):
    time1 = time.time()
    name = path.stem.removesuffix('_Jstress')
    print(f"=== {name} ===", flush=True)
    Jstress = load(str(path))[0]
    out_dir = Path(f'./{name}_dolfinx')
    out_dir.mkdir(parents=True, exist_ok=True)
    to_full_body(
        Jstress,
        path=out_dir / 'full_body_fields.vtu',
        mesh_scale=0.6,
    )
    time2 = time.time()
    print(f"=== {name} done in {time2 - time1:.1f}s ===", flush=True)
    np.save(f'time_export_{name}', time2 - time1)
