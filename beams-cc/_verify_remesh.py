import sys
import time
import numpy as np
import meshio
from simsopt import load
import coil_fem
from coil_fem.io import to_full_body

print("coil_fem:", coil_fem.__file__)
case = sys.argv[1] if len(sys.argv) > 1 else "fin"
json = f"{case}_Jstress.json"
out = f"./_verify_{case}.vtu"

t0 = time.time()
J = load(json)[0]
to_full_body(J, path=out, mesh_scale=0.6)
print(f"[{case}] to_full_body done in {time.time()-t0:.1f}s")

m = meshio.read(out)
counts = {c.type: len(c.data) for c in m.cells}
oc = m.point_data.get("owner_coil")
osym = m.point_data.get("owner_sym")
print(f"[{case}] points={m.points.shape[0]} cells={counts}")
if oc is not None:
    print(f"[{case}] owner_coil>=0: {int((oc>=0).sum())}/{oc.size} "
          f"(coils {sorted(set(oc[oc>=0].tolist()))})")
if osym is not None:
    print(f"[{case}] owner_sym>=0: {int((osym>=0).sum())}/{osym.size}")
