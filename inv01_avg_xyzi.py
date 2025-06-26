import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from aux00_utils import open_simulation, condense, adjust_times, aggregate_parameters
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)
π = 2*np.pi

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
    def condense_velocities(ds):
        return condense(ds, ["u", "v", "w"], "uᵢ", dimname="i", indices=indices)

    def condense_velocity_gradient_tensor(ds):
        ds = condense(ds, ["∂u∂x", "∂v∂x", "∂w∂x"], "∂₁uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂u∂y", "∂v∂y", "∂w∂y"], "∂₂uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂u∂z", "∂v∂z", "∂w∂z"], "∂₃uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂₁uᵢ", "∂₂uᵢ", "∂₃uᵢ"], "∂ⱼuᵢ", dimname="j", indices=indices)
        return ds

    def condense_reynolds_stress_tensor(ds):
        ds["vu"] = ds.uv
        ds["wv"] = ds.vw
        ds["wu"] = ds.uw
        ds = condense(ds, ["uu",   "uv",   "uw"],   "u₁uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["vu",   "vv",   "vw"],   "u₂uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["wu",   "wv",   "ww"],   "u₃uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["u₁uᵢ", "u₂uᵢ", "u₃uᵢ"], "uⱼuᵢ", dimname="j", indices=indices)
        return ds

    xyza = condense_velocities(xyza)
    #xyza = condense_velocity_gradient_tensor(xyza)
    xyza = condense_reynolds_stress_tensor(xyza)

    xyia = condense_velocities(xyia)
    xyia = condense_velocity_gradient_tensor(xyia)
    xyia = condense_reynolds_stress_tensor(xyia)
    #xyia = condense(xyia, ["dbdx", "dbdy", "dbdz"], "∂ⱼb", dimname="j", indices=indices)

    xyii = condense_velocities(xyii)
    xyii = condense_velocity_gradient_tensor(xyii)
    xyii = condense_reynolds_stress_tensor(xyii)
    #---

    #+++ Time average
    # Here ū and ⟨u⟩ₜ are interchangeable
    def temporal_average(ds):
        ds = ds.mean("time", keep_attrs=True).rename({"uᵢ"   : "ūᵢ",
                                                      "b"    : "b̄",
                                                      "uⱼuᵢ" : "⟨uⱼuᵢ⟩ₜ",
                                                      "wb"   : "⟨wb⟩ₜ",
                                                      "∂ⱼuᵢ"    : "∂ⱼūᵢ",
                                                      "εₖ"      : "ε̄ₖ",
                                                      "εₚ"      : "ε̄ₚ",
                                                      "ν"       : "ν̄",
                                                      "κ"       : "κ̄",
                                                      "PV"      : "q̄",
                                                      "Ri"      : "R̄i",
                                                      "Ro"      : "R̄o",
                                                      "ω_y"     : "ω̄_y",
                                                      })
        return ds

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
    xyza = xyza.mean("time", keep_attrs=True).rename({"uᵢ"   : "ūᵢ",
                                                      "b"    : "b̄",
                                                      "uⱼuᵢ" : "⟨uⱼuᵢ⟩ₜ",
                                                      "wb"   : "⟨wb⟩ₜ",
                                                      "εₖ"   : "ε̄ₖ",
                                                      "εₚ"   : "ε̄ₚ",
                                                      "κ"    : "κ̄",
                                                      })
    #---

    #+++ Get turbulent Reynolds stress tensor
    def get_turbulent_Reynolds_stress_tensor(ds):
        ds["ūⱼūᵢ"]      = ds["ūᵢ"] * ds["ūᵢ"].rename(i="j")
        ds["⟨u′ⱼu′ᵢ⟩ₜ"] = ds["⟨uⱼuᵢ⟩ₜ"] - ds["ūⱼūᵢ"]
        return ds

    xyza = get_turbulent_Reynolds_stress_tensor(xyza)
    xyia = get_turbulent_Reynolds_stress_tensor(xyia)
    xyit = get_turbulent_Reynolds_stress_tensor(xyit)

    if True:
        from matplotlib import pyplot as plt
        for i in [2]:
            fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            for j in [0, 1, 2]:
                im = xyia["⟨u′ⱼu′ᵢ⟩ₜ"].isel(i=i, j=j).pnplot(ax=axes[0, j], robust=True)
                vmin, vmax = im.get_clim()
                xyit["⟨u′ⱼu′ᵢ⟩ₜ"].isel(i=i, j=j).pnplot(ax=axes[1, j], vmin=vmin, vmax=vmax, cmap=im.get_cmap())
    #---

    #+++ Get shear production rates
    def get_SPR(ds):
        ds["SPR"] = - (ds["⟨u′ⱼu′ᵢ⟩ₜ"] * ds["∂ⱼūᵢ"]).sum("i")
        return ds

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
    def get_buoyanct_production_rates(ds):
        ds["w̄b̄"]      = ds["ūᵢ"].sel(i=3) * ds["b̄"]
        ds["⟨w′b′⟩ₜ"] = ds["⟨wb⟩ₜ"] - ds["w̄b̄"]
        return ds

    xyia = get_buoyanct_production_rates(xyia)
    xyit = get_buoyanct_production_rates(xyit)

    if False:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
        im = xyia["⟨w′b′⟩ₜ"].pnplot(ax=axes[0], robust=True)
        vmin, vmax = im.get_clim()
        xyit["⟨w′b′⟩ₜ"].pnplot(ax=axes[1], vmin=vmin, vmax=vmax, cmap=im.get_cmap())
    #---

    #+++ Get TKE
    def get_turbulent_kinetic_energy(ds):
        ds["⟨Ek′⟩ₜ"] = (ds["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=1, j=1) + ds["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=2, j=2) + ds["⟨u′ⱼu′ᵢ⟩ₜ"].sel(i=3, j=3)) / 2
        return ds

    xyia = get_turbulent_kinetic_energy(xyia)
    xyit = get_turbulent_kinetic_energy(xyit)

    if False:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
        im = xyia["⟨Ek′⟩ₜ"].pnplot(ax=axes[0], robust=True)
        vmin, vmax = im.get_clim()
        xyit["⟨Ek′⟩ₜ"].pnplot(ax=axes[1], vmin=vmin, vmax=vmax, cmap=im.get_cmap())
    #---
