# Coil stress optimization data
This is the coil stress optimization dataset. 

## List of files/folders
- `coilforce`: The prototype for coil-fem that this dataset is produced with.
- `fixed-convergence`: The benchmark runs with fixed clamps. To re-run, execute `submit.sh` and `submit_petsc.sh`.
- `fixed-continuation`: The optimization runs with fixed clamps. To re-run, run `sbatch jobscript.sh` and `sbatch jobscript_force.sh`.
- `fixed-optimization`: The optimization runs with fixed clamps, using w7x coils as the initial conditions. Not included in the manuscript. To re-run, run `sbatch jobscript.sh` and `sbatch jobscript_force.sh`.
- `beams-remat`: Demonstrating the effectiveness of rematerialization in the cuDSS backend.
- `beams-profiling`: Seeing how big of a problem an L40s can fit.
- `figures`: The figures.
- `beam-options.json`: Hand-authored CoilFEM / CoilSupport option preset (API-key dict). Defaults are the beams / qss family (`itc=0`, zero gravity, beam network + clamps). Fixed-family scripts load the same file and override `itc` / `g_vec` locally.

## Material / gravity references

W7-X coil casings / support structure are modelled as AISI 316LN austenitic stainless steel.

- A. Foussat et al., Mechanical design and construction qualification program on ITER correction coils structures, Nuclear Engineering and Design 269 (2013) 116-124. Table 5 (316LN FEA inputs at cryogenic temperature: E = 205 GPa, G = 78.8 GPa, nu = 0.30, rho = 8000 kg/m³, integral thermal contraction 293/4 K = 0.29 %). DOI: [10.1016/j.nucengdes.2013.08.016](https://doi.org/10.1016/j.nucengdes.2013.08.016).
- Standard gravity g₀ = 9.80665 m/s² (defined constant; 3rd CGPM, 1901; ISO 80000-3).
