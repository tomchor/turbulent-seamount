import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from src.aux00_utils import (open_simulation, adjust_times, aggregate_parameters, gather_attributes_as_variables,
                             condense_velocities, condense_velocity_gradient_tensor, condense_reynolds_stress_tensor,
                             configure_dask_for_performance)
from src.aux01_physfuncs import temporal_average
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)

# Configure dask for optimal performance
configure_dask_for_performance(memory_fraction=0.3)

print("Starting xyza dataset creation script")

#+++ Define directory and simulation name
if not basename(__file__).startswith("00_postproc_"):
    simdata_path = "../simulations/data/"
    simname_base = "balanus"

    Rossby_numbers = cycler(Ro_b = [0.1])
    Froude_numbers = cycler(Fr_b = [0.8])
    L              = cycler(L = [0])

    resolutions    = cycler(dz = [2])

    paramspace = Rossby_numbers * Froude_numbers * L
    configs    = resolutions

    runs = paramspace * configs
#---

#+++ Options
indices = [1, 2, 3]
#---


for j, config in enumerate(runs):
    simname = f"{simname_base}_" + aggregate_parameters(config, sep="_", prefix="")

    #+++ Open dataset
    print(f"\nOpening {simname} xyzi")
    xyzi = open_simulation(simdata_path+f"xyzi.{simname}.nc",
                           use_advective_periods = True,
                           squeeze = True,
                           load = False,
                           get_grid = False,
                           open_dataset_kwargs = dict(chunks="auto"),
                           )
    #---

    #+++ Get dataset ready
    #+++ Get rid of slight misalignment in times
    xyzi = adjust_times(xyzi, round_times=True)
    #---

    #+++ Preliminary definitions and checks
    print("Doing prelim checks")
    Δt = xyzi.time.diff("time").median()
    Δt_tol = Δt/100
    if np.all(xyzi.time.diff("time") > Δt_tol):
        print(Fore.GREEN + f"Δt is consistent for {simname}", Style.RESET_ALL)
    else:
        print(f"Δt is inconsistent for {simname}")
        print(np.count_nonzero(xyzi.time.diff("time") < Δt_tol), "extra time steps")

        tslice1 = slice(0.0, None, None)
        xyzi = xyzi.sel(time=tslice1)

        xyzi = xyzi.reindex(dict(time=np.arange(0, xyzi.time[-1]+1e-5, Δt)), method="nearest", tolerance=Δt/Δt_tol)
    #---

    #+++ Trimming domain
    t_slice_inclusive = slice(xyzi.T_adv_spinup, np.inf) # For snapshots, we want to include t=T_adv_spinup
    t_slice_exclusive = slice(xyzi.T_adv_spinup + 0.01, np.inf) # For time-averaged outputs, we want to exclude t=T_adv_spinup
    x_slice = slice(None, xyzi.x_faa[-2] - 2*xyzi.Δx_faa.values.max()) # Cut off last two points
    y_slice = slice(None)

    xyzi = xyzi.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    #---
    #---

    #+++ Condense and time-average tensors
    print("Processing xyzi tensors...")
    xyzi = condense_velocities(xyzi, indices=indices)
    xyzi = condense_velocity_gradient_tensor(xyzi, indices=indices)
    xyzi = condense_reynolds_stress_tensor(xyzi, indices=indices)
    print("Computing temporal average...")

    # Drop some variables that are not needed
    xyzi = xyzi.drop_vars(["ω_x", "κ", "Ri", "peripheral_nodes_ccf", "peripheral_nodes_cfc", "peripheral_nodes_fcc"])

    xyza = temporal_average(xyzi)
    print("✓ Completed xyzi processing")
    #---

    #+++ Compute and save dataset with dask for optimal performance
    print("Computing dataset with dask...")
    xyza = gather_attributes_as_variables(xyza)

    # Compute all dask arrays in parallel with progress tracking
    outname = f"data/xyza.{simname}.nc"

    with ProgressBar(minimum=5, dt=5):
        xyza = xyza.compute()
        print("Dask computation completed!")
        print(f"Saving results to {outname}...")
        xyza.to_netcdf(outname)
        print("Done!\n")
    xyza.close()
    #---
