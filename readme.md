# Coil stress optimization data
This is the coil stress optimization dataset. 

## List of files/folders
- `coilforce`: The prototype for coil-fem that this dataset is produced with.
- `w7x-convergence`: The benchmark runs in Section 3.1. To re-run, execute `submit.sh` and `submit_petsc.sh`.
- `w7x-continuation`: The optimization runs in Section 3.2. To re-run, run `sbatch jobscript.sh` and `sbatch jobscript_force.sh`.
- `w7x-optimization`: The optimization runs in Section 3.2, using w7x coils as the initial conditions. Not included in the manuscript. To re-run, run `sbatch jobscript.sh` and `sbatch jobscript_force.sh`.
- `figures`: The figures.
- `properties.json`: The material properties and physical constants.

