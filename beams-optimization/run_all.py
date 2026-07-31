from pathlib import Path
from simsopt import load
for path in sorted(Path('.').glob('Jstress_*.json')):
    name = path.stem.removeprefix('Jstress_')  # e.g. Jstress_full.json -> full
    Jstress = load(str(path))[0]
    out_dir = Path(f'./{name}_run')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'{path.name} -> {out_dir}/')
    Jstress.save_run_vtu(str(out_dir))
