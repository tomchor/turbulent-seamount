from cycler import cycler
from simulation_runner import run_simulation_batch

#+++ Define run options
simname_base = "seamount"

Rossby_numbers     = cycler(Ro_h = [0.05, 0.1, 0.2, 0.5])
Froude_numbers     = cycler(Fr_h = [0.02, 0.08, 0.3, 1])
L                  = cycler(L = [0, 0.8])

resolutions    = cycler(dz = [4, 2, 1])
T_advective_spinups = cycler(T_advective_spinup = [12])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions * T_advective_spinups

runs = paramspace * configs

#+++ Options
remove_checkpoints = False
only_one_job = False
dry_run = False

verbose = 1
aux_filename = "aux_submission_script.sh"
julia_script = "seamount.jl"
scheduler = "pbs"
#---

#+++ Run simulations
run_simulation_batch(
    runs=runs,
    simname_base=simname_base,
    julia_script=julia_script,
    scheduler=scheduler,
    remove_checkpoints=remove_checkpoints,
    only_one_job=only_one_job,
    dry_run=dry_run,
    verbose=verbose,
    aux_filename=aux_filename
)
