from cycler import cycler
from simulation_runner import run_simulation_batch

#+++ Define run options
# Define physical parameters
Rossby_numbers      = cycler(Ro_h = [0.05])
Froude_numbers      = cycler(Fr_h = [0.05])
L                   = cycler(L = [0, 0.8])

# Define numerical parameters
resolutions         = cycler(dz = [2])
T_adv_spinups = cycler(T_adv_spinup = [12])

paramspace = Rossby_numbers * Froude_numbers * L
configs    = resolutions * T_adv_spinups

runs = paramspace * configs
#---

#+++ Run simulations
run_simulation_batch(
    runs = runs,
    simname_base = "seamount",
    julia_script = "seamount.jl",
    scheduler = "pbs",
    remove_checkpoints = False,
    only_one_job = False,
    dry_run = False,
    verbose = 1,
    aux_filename = "aux_submission_script.sh"
)
#---