#!/bin/bash
#SBATCH --account=m4740_g
#SBATCH --job-name={simname_ascii}
#SBATCH --output=logs/{simname_ascii}.log
#SBATCH --error=logs/{simname_ascii}.log
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48GB
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=tchor@umd.edu
{options_string}

export SLURM_CPU_BIND="cores"

echo $CUDA_VISIBLE_DEVICES

cd $SCRATCH/tokara-strait/simulations/
time julia --project --pkgimages=no {julia_script} {run_options} --simname={simname} 2>&1 | tee logs/{simname_ascii}.out

#scontrol show job $PBS_JOBID >> logs/{simname_ascii}.log
#scontrol show job $PBS_JOBID >> logs/{simname_ascii}.out
