from os import system
from cycler import cycler
import sys
sys.path.append("..")
import numpy as np
from postprocessing.src.aux00_utils import aggregate_parameters

#+++ Define run options
simname_base = "seamount"

Rossby_numbers = cycler(Ro_h = [0.1])
Froude_numbers = cycler(Fr_h = [1])
L              = cycler(L = [0, 0.05, 0.1, 0.2, 0.4, 0.8,
                             0.8, 0.8])
FWHM           = cycler(FWHM = [500, 500, 500, 500, 500, 500,
                                200, 100])

resolutions    = cycler(dz = [4, 2, 1])

paramspace = Rossby_numbers * Froude_numbers * (L + FWHM)
configs    = resolutions

runs = paramspace * configs
#---

#+++ Options
remove_checkpoints = False
only_one_job = False
dry_run = False
omit_topology = True

verbose = 1
aux_filename = "aux_submission_script.sh"
julia_script = "seamount.jl"

scheduler = "pbs"
#---

#+++ Open submission script template and define options
template = open(f"template.{scheduler}", "r").read()

def very_small_submission_options(scheduler):
    if scheduler == "pbs":
        options = ["select=1:ncpus=1:ngpus=1",
                   "gpu_type=a100"]
        options_string = "\n".join([ "#PBS -l " + option for option in options ])

    elif scheduler == "slurm":
        options = ["--ntasks=1",
                   "--constraint=gpu",
                   "--cpus-per-task=32",
                   "--gpus-per-task=1",
                   "--time=1:00:00",
                   ]
        options_string = "\n".join([ "#SBATCH " + option for option in options ])
    return options_string

def small_submission_options(scheduler):
    if scheduler == "pbs":
        options = ["select=1:ncpus=1:ngpus=1",
                   "gpu_type=a100"]
        options_string = "\n".join([ "#PBS -l " + option for option in options ])

    elif scheduler == "slurm":
        options = ["--ntasks=1",
                   "--constraint=gpu",
                   "--cpus-per-task=32",
                   "--gpus-per-task=1",
                   "--time=2:00:00",
                   ]
        options_string = "\n".join([ "#SBATCH " + option for option in options ])
    return options_string

def big_submission_options(scheduler):
    if scheduler == "pbs":
        options = ["select=1:ncpus=1:ngpus=1",
                   "gpu_type=h100"]
        options_string = "\n".join([ "#PBS -l " + option for option in options ])

    elif scheduler == "slurm":
        options = ["--ntasks=1",
                   "--constraint=gpu&hbm80g",
                   "--cpus-per-task=32",
                   "--gpus-per-task=1",
                   "--time=4:00:00",
                   ]
        options_string = "\n".join([ "#SBATCH " + option for option in options ])
    return options_string

def very_small_submission_command(scheduler):
    if scheduler == "pbs":
        cmd1 = f"qsub {aux_filename}"
    elif scheduler == "slurm":
        cmd1 = f"sbatch {aux_filename}"
    return cmd1

def small_submission_command(scheduler):
    return very_small_submission_command(scheduler)

def big_submission_command(scheduler):
    if scheduler == "pbs":
        cmd1 = f"JID1=`qsub {aux_filename}`; JID2=`qsub -W depend=afterok:$JID1 {aux_filename}`; qrls $JID1"
    elif scheduler == "slurm":
        cmd1 = small_submission_command(scheduler)
    return cmd1
#---

for modifiers in runs:
    run_options = aggregate_parameters(modifiers, use_equals=True)
    simname = f"{simname_base}_" + aggregate_parameters(modifiers, sep="_", prefix="")
    print(simname)

    #+++ Remove previous checkpoints
    if remove_checkpoints:
        cmd0 = f"rm data/chk.{simname}*.jld2"
        if verbose>0: print(cmd0)
        system(cmd0)
    #---

    #+++ Fill pbs script and define command
    Δz = modifiers["dz"] if "dz" in modifiers.keys() else np.inf
    if Δz >= 4:
        options_string = very_small_submission_options(scheduler)
        cmd1           = very_small_submission_command(scheduler)
    elif Δz >= 2:
        options_string = small_submission_options(scheduler)
        cmd1           = small_submission_command(scheduler)
    else:
        options_string = big_submission_options(scheduler)

        if only_one_job:
            cmd1 = small_submission_command(scheduler)
        else:
            cmd1 = big_submission_command(scheduler)

    submission_script = template.format(simname = simname,
                                        simname_ascii = simname,
                                        julia_script = julia_script,
                                        run_options = run_options,
                                        options_string = options_string,)
    if verbose>1: print(submission_script)
    if verbose>0: print(cmd1)
    #---

    #+++ Run command
    if not dry_run:
        with open(aux_filename, "w") as f:
            f.write(submission_script)
        system(cmd1)
    #---

    print()
