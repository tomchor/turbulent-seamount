#!/bin/bash
#SBATCH --account=m4740_g
#SBATCH --job-name={simname_ascii}
#SBATCH --output=logs/{simname_ascii}.log
#SBATCH --error=logs/{simname_ascii}.log
#SBATCH --time=48:00:00
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --mem=48GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=tchor@umd.edu

export SLURM_CPU_BIND="cores"

cd $SCRATCH/tokara-strait/simulations/
srun julia --project seamount.jl
