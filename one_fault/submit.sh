#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=geospark
#SBATCH --ntasks-per-node=32
#SBATCH --mem=512G
#SBATCH --error='job-error.out'
#SBATCH --output='job-out.out'
#SBATCH --export=ALL
#SBATCH --chdir=/ceph/home/kav86926/projects/GM2FEM/one_fault
#SBATCH --job-name=gmsh_mesh_gen

# -------- Environment setup --------
module purge



# -------- Run your Python script --------
python3 GM2FEM_mesh_generator.py
