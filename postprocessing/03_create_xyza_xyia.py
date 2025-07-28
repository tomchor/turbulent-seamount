import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from src.aux00_utils import (open_simulation, adjust_times, aggregate_parameters, gather_attributes_as_variables,
                             condense_velocities, condense_velocity_gradient_tensor, condense_reynolds_stress_tensor,
                             condense_reynolds_stress_tensor_diagonal)
from src.aux01_physfuncs import temporal_average
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)

print("Starting xyza and xyia dataset creation script")

#+++ Define directory and simulation name
if basename(__file__) != "00_run_postproc.py":
    path = "../simulations/data/"
    simname_base = "seamount"

    Rossby_numbers = cycler(Ro_h = [0.2])
    Froude_numbers = cycler(Fr_h = [1.25])
    L              = cycler(L = [0])

    resolutions    = cycler(dz = [8])

    paramspace = Rossby_numbers * Froude_numbers * L
    configs    = resolutions

    runs = paramspace * configs
#---

#+++ Options
indices = [1, 2, 3]
write_xyza = True
#---

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
    t_slice_inclusive = slice(xyii.T_advective_spinup, np.inf) # For snapshots, we want to include t=T_advective_spinup
    t_slice_exclusive = slice(xyii.T_advective_spinup + 0.01, np.inf) # For time-averaged outputs, we want to exclude t=T_advective_spinup
    x_slice = slice(None, np.inf)
    y_slice = slice(None, xyzi.y_afa[-2] - 2*xyzi.Δy_afa.values.max()) # Cut off last two points
    z_slice = slice(None, xyzi.z_aaf[-1] - xyii.h_sponge) # Cut off top sponge

    xyzi = xyzi.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice, z_aac=z_slice, z_aaf=z_slice)
    xyii = xyii.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    #---

    #+++ Condense and time-average tensors
    xyzi = condense_velocities(xyzi, indices=indices)
    xyzi = condense_velocity_gradient_tensor(xyzi, indices=indices)
    xyzi = condense_reynolds_stress_tensor(xyzi, indices=indices)
    xyza = temporal_average(xyzi)


    xyii = condense_velocities(xyii, indices=indices)
    xyii = condense_velocity_gradient_tensor(xyii, indices=indices)
    xyii = condense_reynolds_stress_tensor(xyii, indices=indices)
    xyia = temporal_average(xyii)
    #---

    #+++ Save xyia
    outname = f"data_post/xyia.{simname}.nc"
    xyia = gather_attributes_as_variables(xyia)
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving results to {outname}...")
        xyia.to_netcdf(outname)
        print("Done!\n")
    xyia.close()
    #---

    #+++ Save xyza
    if write_xyza:
        outname = f"data_post/xyza.{simname}.nc"
        xyza = gather_attributes_as_variables(xyza)
        with ProgressBar(minimum=5, dt=5):
            print(f"Saving results to {outname}...")
            xyza.to_netcdf(outname)
            print("Done!\n")
        xyza.close()
    #---