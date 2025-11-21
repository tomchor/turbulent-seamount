from cycler import cycler
from simulation_runner import run_simulation_batch

#+++ Define run options
# Define physical parameters
Rossby_numbers = cycler(Ro_b = [0.1])
Froude_numbers = cycler(Fr_b = [0.8])
L              = cycler(L = [0, 0.05, 0.1, 0.2, 0.4, 0.8])
FWHM           = cycler(FWHM = [1000])
Lx             = cycler(Lx = [9000])
Ly             = cycler(Ly = [4000])

# Define numerical parameters
resolutions    = cycler(dz = [2])

paramspace = Rossby_numbers * Froude_numbers * L * FWHM * Lx * Ly
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
    aux_filename = "aux_submission_script.sh",
    gpu_type = "a100_80gb"  # Set to desired GPU type (e.g., "a100_80gb", "cc80", "h100") to override defaults
)
#---
