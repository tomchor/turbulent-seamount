from cycler import cycler
from simulation_runner import run_simulation_batch

#+++ Define run options
# Define physical parameters
Rossby_numbers = cycler(Ro_b = [0.1])
Froude_numbers = cycler(Fr_b = [1])
L              = cycler(L = [0, 0.05, 0.1, 0.2, 0.4, 0.8])
FWHM           = cycler(FWHM = [500, 500, 500, 500, 500, 500])

# Define numerical parameters
resolutions    = cycler(dz = [4, 2, 1])

paramspace = Rossby_numbers * Froude_numbers * (L + FWHM)
configs    = resolutions

runs = paramspace * configs
#---

#+++ Run simulations
run_simulation_batch(
    runs = runs,
    simname_base = "balanus",
    julia_script = "seamount.jl",
    scheduler = "pbs",
    remove_checkpoints = False,
    only_one_job = True,
    dry_run = False,
    verbose = 1,
    aux_filename = "aux_submission_script.sh"
)
#---