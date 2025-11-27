"""
Auxiliary script for running seamount simulations on HPC systems.
Contains shared functionality for parameter sweeps and batch job submission.
"""

from os import system
import numpy as np
import sys
sys.path.append("..")
from postprocessing.src.aux00_utils import aggregate_parameters

#+++ Get submission options
def get_submission_options(scheduler, job_size, gpu_type=None):
    """
    Get PBS/SLURM submission options based on job size.

    Parameters:
    -----------
    scheduler : str
        Either "pbs" or "slurm"
    job_size : str
        One of "very_small", "small", or "big"
    gpu_type : str, optional
        GPU type to use (overrides default for job_size). Only used for PBS scheduler.

    Returns:
    --------
    str
        Formatted submission options string
    """

    if job_size == "very_small":
        if scheduler == "pbs":
            default_gpu_type = "v100"
            gpu_type_override = gpu_type if gpu_type is not None else default_gpu_type
            options = [f"select=1:ncpus=1:ngpus=1:gpu_type={gpu_type_override}:mem=50GB"]
        elif scheduler == "slurm":
            options = ["--ntasks=1",
                       "--constraint=gpu",
                       "--cpus-per-task=32",
                       "--gpus-per-task=1",
                       "--time=2:00:00"]

    elif job_size == "small":
        if scheduler == "pbs":
            default_gpu_type = "cc80"
            gpu_type_override = gpu_type if gpu_type is not None else default_gpu_type
            options = [f"select=1:ncpus=1:ngpus=1:gpu_type={gpu_type_override}:mem=200GB"]
        elif scheduler == "slurm":
            options = ["--ntasks=1",
                       "--constraint=gpu",
                       "--cpus-per-task=32",
                       "--gpus-per-task=1",
                       "--time=4:00:00"]

    elif job_size == "big":
        if scheduler == "pbs":
            default_gpu_type = "h100"
            gpu_type_override = gpu_type if gpu_type is not None else default_gpu_type
            options = [f"select=1:ncpus=1:ngpus=1:gpu_type={gpu_type_override}:mem=200GB",
                       "job_priority=regular"]
        elif scheduler == "slurm":
            options = ["--ntasks=1",
                       "--constraint=gpu&hbm80g",
                       "--cpus-per-task=32",
                       "--gpus-per-task=1",
                       "--time=12:00:00"]
    else:
        raise ValueError(f"Unknown job_size: {job_size}")

    if scheduler == "pbs":
        return "\n".join([f"#PBS -l {option}" for option in options])
    elif scheduler == "slurm":
        return "\n".join([f"#SBATCH {option}" for option in options])
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")
#---

#+++ Get submission command
def get_submission_command(scheduler, job_size, aux_filename, only_one_job=False):
    """
    Get the command to submit jobs based on scheduler and job size.

    Parameters:
    -----------
    scheduler : str
        Either "pbs" or "slurm"
    job_size : str
        One of "very_small", "small", or "big"
    aux_filename : str
        Name of the auxiliary submission script file
    only_one_job : bool
        If True, submit only one job even for big jobs

    Returns:
    --------
    str
        Command to submit the job
    """

    if job_size in ["very_small", "small"]:
        if scheduler == "pbs":
            return f"qsub {aux_filename}"
        elif scheduler == "slurm":
            return f"sbatch {aux_filename}"

    elif job_size == "big":
        if only_one_job:
            # Use small submission command even for big jobs
            if scheduler == "pbs":
                return f"qsub {aux_filename}"
            elif scheduler == "slurm":
                return f"sbatch {aux_filename}"
        else:
            # Chain jobs for big simulations
            if scheduler == "pbs":
                return f"JID1=`qsub {aux_filename}`; JID2=`qsub -W depend=afterok:$JID1 {aux_filename}`; qrls $JID1"
            elif scheduler == "slurm":
                return f"sbatch {aux_filename}"

    raise ValueError(f"Unknown job_size: {job_size}")
#---

#+++ Determine job size
def determine_job_size(modifiers):
    """
    Determine job size based on resolution (dz parameter).

    Parameters:
    -----------
    modifiers : dict
        Dictionary containing simulation parameters

    Returns:
    --------
    str
        Job size: "very_small", "small", or "big"
    """
    dz = modifiers.get("dz", np.inf)

    if dz >= 4:
        return "very_small"
    elif dz >= 2:
        return "small"
    else:
        return "big"
#---

#+++ Run simulation batch
def run_simulation_batch(runs, simname_base, julia_script, scheduler="pbs",
                         remove_checkpoints=False, only_one_job=False,
                         dry_run=False, verbose=1, aux_filename="aux_submission_script.sh",
                         gpu_type=None):
    """
    Run a batch of simulations with the given parameters.

    Parameters:
    -----------
    runs : iterable
        Iterable of parameter dictionaries for each simulation
    simname_base : str
        Base name for simulations
    julia_script : str
        Name of the Julia script to run
    scheduler : str
        Scheduler type ("pbs" or "slurm")
    remove_checkpoints : bool
        Whether to remove existing checkpoints
    only_one_job : bool
        Whether to limit big jobs to single submission
    dry_run : bool
        If True, don't actually submit jobs
    verbose : int
        Verbosity level (0, 1, or 2)
    aux_filename : str
        Name for temporary submission script file
    gpu_type : str, optional
        GPU type to use (overrides default for job_size). Only used for PBS scheduler.
    """

    # Load template
    template = open(f"template.{scheduler}", "r").read()

    for modifiers in runs:
        run_options = aggregate_parameters(modifiers, use_equals=True)
        simname = f"{simname_base}_" + aggregate_parameters(modifiers, sep="_", prefix="")
        print(simname)

        # Remove previous checkpoints if requested
        if remove_checkpoints:
            cmd0 = f"rm data/chk.{simname}*.jld2"
            if verbose > 0:
                print(cmd0)
            if not dry_run:
                system(cmd0)

        # Determine job size and get submission options
        job_size = determine_job_size(modifiers)
        options_string = get_submission_options(scheduler, job_size, gpu_type=gpu_type)
        cmd1 = get_submission_command(scheduler, job_size, aux_filename, only_one_job)

        # Create submission script
        submission_script = template.format(
            simname=simname,
            simname_ascii=simname,
            julia_script=julia_script,
            run_options=run_options,
            options_string=options_string
        )

        if verbose > 1:
            print(submission_script)
        if verbose > 0:
            print(cmd1)

        # Submit job
        if not dry_run:
            with open(aux_filename, "w") as f:
                f.write(submission_script)
            system(cmd1)

        print()
#---