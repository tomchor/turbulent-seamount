import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from src.aux00_utils import (open_simulation, adjust_times, aggregate_parameters, gather_attributes_as_variables,
                             condense_velocities, condense_velocity_gradient_tensor, condense_reynolds_stress_tensor)
from src.aux01_physfuncs import temporal_average
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
xr.set_options(display_width=140, display_max_rows=30)

print("Starting xyza and xyia dataset creation script")

#+++ Define directory and simulation name
if basename(__file__) != "00_run_postproc.py":
    path = "../simulations/data/"
    simname_base = "seamount"

    Rossby_numbers = cycler(Ro_h = [0.1])
    Froude_numbers = cycler(Fr_h = [1])
    L              = cycler(L = [0])

    resolutions    = cycler(dz = [8])
    FWHM           = cycler(FWHM = [500])

    paramspace = Rossby_numbers * Froude_numbers * (L + FWHM)
    configs    = resolutions

    runs = paramspace * configs
#---

#+++ Options
indices = [1, 2, 3]
write_xyza = True
#---

def process_dataset(ds, dataset_type, indices):
    """Process a dataset with tensor condensation and temporal averaging"""
    print(f"Processing {dataset_type} tensors...")
    ds = condense_velocities(ds, indices=indices)
    ds = condense_velocity_gradient_tensor(ds, indices=indices)
    ds = condense_reynolds_stress_tensor(ds, indices=indices)
    print(f"Computing {dataset_type} temporal average...")
    ds_avg = temporal_average(ds)
    return ds_avg

for j, config in enumerate(runs):
    simname = f"{simname_base}_" + aggregate_parameters(config, sep="_", prefix="")

    #+++ Open datasets
    print(f"\nOpening {simname} xyzi")
    xyzi = open_simulation(path+f"xyzi.{simname}.nc",
                           use_advective_periods = True,
                           squeeze = True,
                           load = False,
                           get_grid = False,
                           open_dataset_kwargs = dict(chunks="auto"),
                           )
    print(f"Opening {simname} xyii")
    xyii = open_simulation(path+f"xyii.{simname}.nc",
                           use_advective_periods = True,
                           squeeze = True,
                           load = False,
                           get_grid = False,
                           open_dataset_kwargs = dict(chunks="auto"),
                           )
    #---

    #+++ Get datasets ready
    #+++ Get rid of slight misalignment in times
    xyzi = adjust_times(xyzi, round_times=True)
    xyii = adjust_times(xyii, round_times=True)
    #---

    #+++ Preliminary definitions and checks
    print("Doing prelim checks")
    Δt = xyzi.time.diff("time").median()
    Δt_tol = Δt/100
    if np.all(xyii.time.diff("time") > Δt_tol):
        print(Fore.GREEN + f"Δt is consistent for {simname}", Style.RESET_ALL)
    else:
        print(f"Δt is inconsistent for {simname}")
        print(np.count_nonzero(xyzi.time.diff("time") < Δt_tol), "extra time steps")

        tslice1 = slice(0.0, None, None)
        xyzi = xyzi.sel(time=tslice1)

        xyii = xyii.reindex(dict(time=np.arange(0, xyzi.time[-1]+1e-5, Δt)), method="nearest", tolerance=Δt/Δt_tol)
        xyzi = xyzi.reindex(dict(time=np.arange(0, xyzi.time[-1]+1e-5, Δt)), method="nearest", tolerance=Δt/Δt_tol)
    #---

    #+++ Trimming domain
    t_slice_inclusive = slice(xyzi.T_advective_spinup, np.inf) # For snapshots, we want to include t=T_advective_spinup
    t_slice_exclusive = slice(xyzi.T_advective_spinup + 0.01, np.inf) # For time-averaged outputs, we want to exclude t=T_advective_spinup
    x_slice = slice(None, xyzi.x_faa[-2] - 2*xyzi.Δx_faa.values.max()) # Cut off last two points
    y_slice = slice(None)
    z_slice = slice(None, xyzi.z_aaf[-1] - xyzi.h_sponge) # Cut off top sponge

    xyzi = xyzi.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice, z_aac=z_slice, z_aaf=z_slice)
    xyii = xyii.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    #---
    #---

    #+++ Condense and time-average tensors in parallel
    print("Processing datasets in parallel...")
    results = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Submit both processing tasks
        future_xyzi = executor.submit(process_dataset, xyzi, 'xyzi', indices)
        future_xyii = executor.submit(process_dataset, xyii, 'xyii', indices)
        
        # Collect results as they complete
        for future in as_completed([future_xyzi, future_xyii]):
            if future == future_xyzi:
                xyza = future.result()
                print("✓ Completed xyzi processing")
            else:
                xyia = future.result()
                print("✓ Completed xyii processing")
    #---

    #+++ Save xyia
    outname = f"data/xyia.{simname}.nc"
    xyia = gather_attributes_as_variables(xyia)
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving results to {outname}...")
        xyia.to_netcdf(outname)
        print("Done!\n")
    xyia.close()
    #---

    #+++ Save xyza
    if write_xyza:
        outname = f"data/xyza.{simname}.nc"
        xyza = gather_attributes_as_variables(xyza)
        with ProgressBar(minimum=5, dt=5):
            print(f"Saving results to {outname}...")
            xyza.to_netcdf(outname)
            print("Done!\n")
        xyza.close()
    #---