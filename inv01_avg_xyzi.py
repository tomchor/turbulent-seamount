import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from aux00_utils import open_simulation, condense, adjust_times, aggregate_parameters, condense_velocities, condense_velocity_gradient_tensor, condense_reynolds_stress_tensor
from aux01_physfuncs import (temporal_average, temporal_average_xyza,
                             get_turbulent_Reynolds_stress_tensor, get_SPR,
                             get_buoyancy_production_rates, get_turbulent_kinetic_energy)
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)
π = np.pi

print("Starting energy transfer script")

#+++ Define directory and simulation name
if basename(__file__) != "00_run_postproc.py":
    path = "simulations/data/"
    simname_base = "seamount"

    Rossby_numbers = cycler(Ro_h = [0.2])
    Froude_numbers = cycler(Fr_h = [1.25])
    L              = cycler(L = [0])

    resolutions    = cycler(dz = [4,])
    closures       = cycler(closure = ["AMD", "AMC", "CSM", "DSM", "NON"])
    closures       = cycler(closure = ["DSM"])

    paramspace = Rossby_numbers * Froude_numbers * L
    configs    = resolutions * closures

    runs = paramspace * configs
#---

#+++ Options
indices = [1, 2, 3]
#---

for j, config in enumerate(runs):
    simname = f"{simname_base}_" + aggregate_parameters(config, sep="_", prefix="")

    #+++ Open datasets
    print(f"\nOpening {simname} xyzi")
    xyzi = open_simulation(path+f"xyzi.{simname}.nc",
                           use_advective_periods = True,
                           topology = simname[:3],
                           squeeze = True,
                           load = False,
                           get_grid = False,
                           open_dataset_kwargs = dict(chunks="auto"),
                           ) 
    print(f"Opening {simname} xyii")
    xyii = open_simulation(path+f"xyii.{simname}.nc",
                           use_advective_periods = True,
                           topology = simname[:3],
                           squeeze = True,
                           load = False,
                           get_grid = False,
                           open_dataset_kwargs = dict(chunks="auto"),
                           ) 
    print(f"Opening {simname} xyza")
    xyza = open_simulation(path+f"xyza.{simname}.nc",
                           use_advective_periods = True,
                           topology = simname[:3],
                           squeeze = False,
                           load = False,
                           get_grid = False,
                           open_dataset_kwargs = dict(chunks="auto"),
                           ) 
    print(f"Opening {simname} xyia")
    xyia = open_simulation(path+f"xyia.{simname}.nc",
                           use_advective_periods = True,
                           topology = simname[:3],
                           squeeze = False,
                           load = False,
                           get_grid = False,
                           open_dataset_kwargs = dict(chunks="auto"),
                           )
    #---

    #+++ Get rid of slight misalignment in times
    xyzi = adjust_times(xyzi, round_times=True)
    xyii = adjust_times(xyii, round_times=True)
    xyza = adjust_times(xyza, round_times=True)
    xyia = adjust_times(xyia, round_times=True)

    xyza = xyza.assign_coords(x_caa=xyia.x_caa.values, y_aca=xyia.y_aca.values) # This is needed just as long as xyza is float32 and xyia is float64
    #---

    #+++ Trimming domain
    #t_slice_inclusive = slice(xyia.T_advective_spinup, np.inf) # For snapshots, we want to include t=T_advective_spinup
    t_slice_inclusive = slice(10, np.inf) # For snapshots, we want to include t=T_advective_spinup
    t_slice_exclusive = slice(xyia.T_advective_spinup + 0.01, np.inf) # For time-averaged outputs, we want to exclude t=T_advective_spinup
    x_slice = slice(None, np.inf)
    y_slice = slice(None, np.inf)

    xyzi = xyzi.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    xyii = xyii.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    xyza = xyza.sel(time=t_slice_exclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    xyia = xyia.sel(time=t_slice_exclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    #---

    #+++ Condense tensors
    xyza = condense_velocities(xyza, indices=indices)
    #xyza = condense_velocity_gradient_tensor(xyza, indices=indices)
    xyza = condense_reynolds_stress_tensor(xyza, indices=indices)

    xyia = condense_velocities(xyia, indices=indices)
    xyia = condense_velocity_gradient_tensor(xyia, indices=indices)
    xyia = condense_reynolds_stress_tensor(xyia, indices=indices)
    #xyia = condense(xyia, ["dbdx", "dbdy", "dbdz"], "∂ⱼb", dimname="j", indices=indices)

    xyii = condense_velocities(xyii, indices=indices)
    xyii = condense_velocity_gradient_tensor(xyii, indices=indices)
    xyii = condense_reynolds_stress_tensor(xyii, indices=indices)
    #---

    #+++ Time average
    # Here ū and ⟨u⟩ₜ are interchangeable
    xyia = temporal_average(xyia).squeeze()
    print(f"Averaging out {len(xyii.time)} time steps: {xyii.time.values}")
    xyit = temporal_average(xyii).squeeze()

    if False:
        from matplotlib import pyplot as plt
        for var in ["R̄o", "ε̄ₖ", "q̄", "κ̄"]:
            fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
            im = xyia[var].pnplot(ax=axes[0], robust=True)
            vmin, vmax = im.get_clim()
            xyit[var].pnplot(ax=axes[1], vmin=vmin, vmax=vmax, cmap=im.get_cmap())
    xyza = temporal_average_xyza(xyza)
    #---

    #+++ Get turbulent Reynolds stress tensor
    xyza = get_turbulent_Reynolds_stress_tensor(xyza)
    xyia = get_turbulent_Reynolds_stress_tensor(xyia)
    xyit = get_turbulent_Reynolds_stress_tensor(xyit)

    if False:
        from matplotlib import pyplot as plt
        for i in [2]:
            fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            for j in [0, 1, 2]:
                im = xyia["⟨u′ⱼu′ᵢ⟩ₜ"].isel(i=i, j=j).pnplot(ax=axes[0, j], robust=True)
                vmin, vmax = im.get_clim()
                xyit["⟨u′ⱼu′ᵢ⟩ₜ"].isel(i=i, j=j).pnplot(ax=axes[1, j], vmin=vmin, vmax=vmax, cmap=im.get_cmap())
    #---

    #+++ Get shear production rates
    xyia = get_SPR(xyia)
    xyit = get_SPR(xyit)

    if False:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        for j in [0, 1, 2]:
            im = xyia["SPR"].isel(j=j).pnplot(ax=axes[0, j], robust=True)
            vmin, vmax = im.get_clim()
            xyit["SPR"].isel(j=j).pnplot(ax=axes[1, j], vmin=vmin, vmax=vmax, cmap=im.get_cmap())
    #---

    #+++ Get buoyancy production rates
    xyia = get_buoyancy_production_rates(xyia)
    xyit = get_buoyancy_production_rates(xyit)

    if False:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
        im = xyia["⟨w′b′⟩ₜ"].pnplot(ax=axes[0], robust=True)
        vmin, vmax = im.get_clim()
        xyit["⟨w′b′⟩ₜ"].pnplot(ax=axes[1], vmin=vmin, vmax=vmax, cmap=im.get_cmap())
    #---

    #+++ Get TKE
    xyia = get_turbulent_kinetic_energy(xyia)
    xyit = get_turbulent_kinetic_energy(xyit)

    if False:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
        im = xyia["⟨Ek′⟩ₜ"].pnplot(ax=axes[0], robust=True)
        vmin, vmax = im.get_clim()
        xyit["⟨Ek′⟩ₜ"].pnplot(ax=axes[1], vmin=vmin, vmax=vmax, cmap=im.get_cmap())
    #---
