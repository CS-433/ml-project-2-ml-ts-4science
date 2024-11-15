#!/bin/bash
#SBATCH --job-name=preprocess_panda   # Job name
#SBATCH --output=/capstor/scratch/cscs/vsubrama/slurm/logs/%x/preprocess/%j.log         # Standard output and error log
#SBATCH --ntasks=1                      # Number of task
#SBATCH --cpus-per-task=32              # Number of CPU cores per task
#SBATCH --nodes=1
#SBATCH --time=10:00:00          
#SBATCH --environment=/capstor/scratch/cscs/vsubrama/edf/python_openslide_bristen.toml
#SBATCH --account=a02
# #SBATCH --begin=2024-08-08T01:00:00

python /capstor/scratch/cscs/vsubrama/code/preprocessing/preprocess.py
