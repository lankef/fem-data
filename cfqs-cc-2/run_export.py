import sys
import time
import numpy as np
from pathlib import Path
from simsopt import load
from coil_fem.io import to_full_body

CASES = ("init_auglag", "fin_auglag")

if len(sys.argv) != 2 or sys.argv[1] not in CASES:
    sys.exit(f"usage: run_export.py <case>, where case is one of {CASES}")

name = sys.argv[1]
path = Path(f"{name}_Jstress.json")

time1 = time.time()
print(f"=== {name} ===", flush=True)
Jstress = load(str(path))[0]
out_dir = Path(f"./{name}_dolfinx")
out_dir.mkdir(parents=True, exist_ok=True)
to_full_body(
    Jstress,
    path=out_dir / "full_body_fields.vtu",
    mesh_scale=0.2,
)
time2 = time.time()
print(f"=== {name} done in {time2 - time1:.1f}s ===", flush=True)
np.save(f"time_export_{name}", time2 - time1)
