#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=5000
#SBATCH --account=torch_pr_292_courant
#SBATCH --time=00:50:00

module load anaconda3/2025.06
# conda clean --packages --tarballs -y
# conda remove -y --all -n desc
# conda create -y -n desc python=3.12 ipykernel
source $(conda info --base)/etc/profile.d/conda.sh
conda activate desc
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH
conda install -c conda-forge fenics-dolfinx gmsh 