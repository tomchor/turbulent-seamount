from os import system
from cycler import cycler
import sys
sys.path.append("..")
from aux00_utils import aggregate_parameters

#+++ Define run options
simname_base = "seamount"

Rossby_numbers = cycler(Ro_h = [0.2, 1.25])
Froude_numbers = cycler(Fr_h = [0.2, 1.25])
L              = cycler(L = [0, 300])

resolutions    = cycler(dz = [4, 2,])
closures       = cycler(closure = ["AMD", "AMC", "CSM", "DSM", "NON"])
closures       = cycler(closure = ["AMD", "CSM"])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions * closures

runs = paramspace * configs
#---

#+++ Options
remove_checkpoints = False
only_one_job = False
dry_run = False
omit_topology = True

verbose = 1
aux_filename = "aux_pbs.sh"
julia_script = "seamount.jl"
#---

#+++ PBS script template
pbs_script = \
"""#!/bin/bash -l
#PBS -A UMCP0028
#PBS -N {simname_ascii}
#PBS -o logs/{simname_ascii}.log
#PBS -e logs/{simname_ascii}.log
#PBS -l walltime=23:59:00
#PBS -q casper
#PBS -l {options_string1}
#PBS -l {options_string2}
#PBS -M tchor@umd.edu
#PBS -m ae
#PBS -r n

# Clear the environment from any previously loaded modules
module li
module --force purge
module load ncarenv/23.10
module load cuda
module li

#/glade/u/apps/ch/opt/usr/bin/dumpenv # Dumps environment (for debugging with CISL support)

export JULIA_DEPOT_PATH="/glade/work/tomasc/.julia"
echo $CUDA_VISIBLE_DEVICES

time /glade/u/home/tomasc/bin/julia-1.10.9/bin/julia --project --pkgimages=no {julia_script} {run_options} --simname={simname} 2>&1 | tee logs/{simname_ascii}.out

qstat -f $PBS_JOBID >> logs/{simname_ascii}.log
qstat -f $PBS_JOBID >> logs/{simname_ascii}.out
"""
#---

for modifiers in runs:
    run_options = aggregate_parameters(modifiers)
    simname = f"{simname_base}_" + aggregate_parameters(modifiers, sep="_", prefix="")
    simname_ascii = simname.replace("=", "")
    print(simname_ascii)

    #+++ Remove previous checkpoints
    if remove_checkpoints:
        cmd0 = f"rm data/chk.{simname}*.jld2"
        if verbose>0: print(cmd0)
        system(cmd0)
    #---

    #+++ Fill pbs script and define command
    options1 = dict(select=1, ncpus=1, ngpus=1)
    options2 = dict()

    Δz = modifiers["dz"] if "dz" in modifiers.keys() else np.inf
    if Δz >= 8:
        options2 = options2 | dict(gpu_type = "v100")
        cmd1 = f"qsub {aux_filename}"
    elif Δz >= 2:
        options2 = options2 | dict(gpu_type = "a100")
        cmd1 = f"qsub {aux_filename}"
    else:
        options1 = options1 | dict(cpu_type = "milan")
        options2 = options2 | dict(gpu_type = "a100")

        if modifiers["α"] > 0.1:
            cmd1 = f"qsub {aux_filename}"
        else:
            if only_one_job:
                cmd1 = f"qsub {aux_filename}"
            else:
                cmd1 = f"JID1=`qsub {aux_filename}`; JID2=`qsub -W depend=afterok:$JID1 {aux_filename}`; qrls $JID1"

    options_string1 = ":".join([ f"{key}={val}" for key, val in options1.items() ])
    options_string2 = ":".join([ f"{key}={val}" for key, val in options2.items() ])

    pbs_script_filled = pbs_script.format(simname_ascii = simname_ascii,
                                          simname = simname,
                                          julia_script = julia_script,
                                          run_options = run_options,
                                          options_string1 = options_string1,
                                          options_string2 = options_string2)
    if verbose>1: print(pbs_script_filled)
    if verbose>0: print(cmd1)
    #---

    #+++ Run command
    if not dry_run:
        with open(aux_filename, "w") as f:
            f.write(pbs_script_filled)
        system(cmd1)
    #---

    print()
