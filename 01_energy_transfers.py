import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from aux00_utils import (open_simulation, condense, adjust_times, aggregate_parameters,
                         condense_velocities, condense_velocity_gradient_tensor, condense_reynolds_stress_tensor)
from aux01_physfuncs import (temporal_average,
                             get_turbulent_Reynolds_stress_tensor, get_shear_production_rates,
                             get_buoyancy_production_rates, get_turbulent_kinetic_energy)
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
    L              = cycler(L = [0,])

    resolutions    = cycler(dz = [2,])
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
    y_slice = slice(None, np.inf)

    xyzi = xyzi.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    xyii = xyii.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice)
    #---

    #+++ Condense and time-average tensors
    xyzi = condense_velocities(xyzi, indices=indices)
    xyzi = condense_velocity_gradient_tensor(xyzi, indices=indices)
    xyzi = condense_reynolds_stress_tensor(xyzi, indices=indices)
    xyzi = temporal_average(xyzi)

    xyii = condense_velocities(xyii, indices=indices)
    xyii = condense_velocity_gradient_tensor(xyii, indices=indices)
    xyii = condense_reynolds_stress_tensor(xyii, indices=indices)
    xyii = temporal_average(xyii)
    #---
    #---

    #+++ Get turbulent variables
    xyzi = get_turbulent_Reynolds_stress_tensor(xyzi)
    xyzi = get_shear_production_rates(xyzi)
    xyzi = get_buoyancy_production_rates(xyzi)
    xyzi = get_turbulent_kinetic_energy(xyzi)

    xyii = get_turbulent_Reynolds_stress_tensor(xyii)
    xyii = get_shear_production_rates(xyii)
    xyii = get_buoyancy_production_rates(xyii)
    xyii = get_turbulent_kinetic_energy(xyii)
    #---

    #+++ Volume-average/integrate results so far
    xyzi["ΔxΔyΔz"] = xyzi["Δx_caa"] * xyzi["Δy_aca"] * xyzi["Δz_aac"]
    xyzi["ΔxΔy"] = xyzi["Δx_caa"] * xyzi["Δy_aca"]
    xyzi["ΔyΔz"] = xyzi["Δy_aca"] * xyzi["Δz_aac"]

    def integrate(da, dV = None, dims=("x", "y", "z")):
        if dV is None:
            if dims == ("x", "y", "z"):
                dV = xyia["ΔxΔyΔz"]
            elif dims == ("x", "y"):
                dV =  xyia["ΔxΔy"]
            elif dims == ("y", "z"):
                dV =  xyia["ΔyΔz"]
        return (da * dV).pnsum(dims)


    def drop_faces(ds, drop_coords=True):
        """
        Drop all variables that have face coordinates (x_faa, y_afa, z_aaf)

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to filter
        drop_coords : bool, optional
            Whether to also drop coordinates with face dimensions. Default True.

        Returns
        -------
        xarray.Dataset
            Dataset with face variables removed
        """
        face_dims = ["x_faa", "y_afa", "z_aaf"]

        # Get variables to drop
        vars_to_drop = []
        for var in ds.variables:
            if any(dim in ds[var].dims for dim in face_dims):
                if var in ds.coords and not drop_coords:
                    continue
                vars_to_drop.append(var)

        return ds.drop_vars(vars_to_drop)

    xyzi = xyzi.where(xyzi.distance_condition_10meters.compute(), drop=True, other=np.nan)
    for var in ["ε̄ₚ", "ε̄ₖ", "⟨Ek′⟩ₜ", "⟨w′b′⟩ₜ", "SPR"]:
        int_buf = f"∭⁵{var}dV"
        xyzi[int_buf] = integrate(xyzi[var], dV=xyzi.ΔxΔyΔz)

    xyza["average_turbulence_mask"] = xyza["ε̄ₖ"] > 1e-11
    xyza["1"] = mask_immersed(xr.ones_like(xyza["ε̄ₖ"]))
    for var in ["ε̄ₖ", "1"]:
        int_turb = f"∬ᵋ{var}dxdy"
        xyza[int_turb] = integrate(xyza[var], dV=xyza.ΔxΔyΔz.where(xyza.average_turbulence_mask), dims=("x", "y"))
    pause
    #---

    #+++ Create bulk dataset
    bulk = xr.Dataset()
    bulk.attrs = xyia.attrs

    bulk["∭⁵ε̄ₖdV"]     = xyia["∭⁵ε̄ₖdV"]
    bulk["∭⁵ε̄ₚdV"]     = xyia["∭⁵ε̄ₚdV"]
    bulk["∭¹⁰ε̄ₖdV"]    = xyia["∭¹⁰ε̄ₖdV"]
    bulk["∭¹⁰ε̄ₚdV"]    = xyia["∭¹⁰ε̄ₚdV"]
    bulk["∭²⁰ε̄ₖdV"]    = xyia["∭²⁰ε̄ₖdV"]
    bulk["∭²⁰ε̄ₚdV"]    = xyia["∭²⁰ε̄ₚdV"]

    bulk["∭⁵⟨Ek′⟩ₜdV"]  = xyza["∭⁵⟨Ek′⟩ₜdV"]
    bulk["∭⁵⟨w′b′⟩ₜdV"] = xyza["∭⁵⟨w′b′⟩ₜdV"]

    bulk["V∞∬⟨Ek′⟩ₜdxdz"] = xyza.attrs["V∞"] * integrate(xyza["⟨Ek′⟩ₜ"].pnsel(y=np.inf, method="nearest"), dV=xyza.Δx_caa*xyza.Δz_aac, dims=["x", "z"])

    bulk["∬⁰SPRdxdy"]  = integrate(xyia["SPR"], dims = ("x", "y"))
    bulk["∬⁰Πdxdy"]    = bulk["∬⁰SPRdxdy"].sum("j")

    altitude = xyzi.altitude.pnsel(z=xyia.z_aac, method="nearest")
    bulk["∬⁵SPRdxdy"]  = integrate(xyia["SPR"].where(altitude > 5, other=0), dims = ("x", "y"))
    bulk["∬⁵Πdxdy"]    = bulk["∬⁵SPRdxdy"].sum("j")

    bulk["∬ᵋε̄ₖdxdy"] = xyza["∬ᵋε̄ₖdxdy"]
    bulk["⟨ε̄ₖ⟩ᵋ"]    = xyza["∬ᵋε̄ₖdxdy"] / xyza["∬ᵋ1dxdy"]
    bulk["Loᵋ"]      = 2*π * np.sqrt(bulk["⟨ε̄ₖ⟩ᵋ"] / bulk.N2_inf**(3/2))

    bulk["Δz̃"] = bulk.Δz_min / bulk["Loᵋ"]

    bulk["Slope_Bu"] = bulk.Slope_Bu
    bulk["Ro_h"] = bulk.Ro_h
    bulk["Fr_h"] = bulk.Fr_h
    bulk["α"]    = bulk.α

    bulk["Bu_h"] = bulk.Bu_h
    bulk["Γ"]    = bulk.Γ
    bulk["c_dz"] = bulk.c_dz

    bulk["f₀"]  = bulk.attrs["f₀"]
    bulk["N²∞"] = bulk.attrs["N²∞"]
    bulk["V∞"]  = bulk.attrs["V∞"]
    bulk["L"]   = bulk.L
    bulk["Δx_min"] = xyia["Δx_caa"].where(xyia["Δx_caa"] > 0).min().values
    bulk["Δy_min"] = xyia["Δy_aca"].where(xyia["Δy_aca"] > 0).min().values
    bulk["Δz_min"] = xyia["Δz_aac"].where(xyia["Δz_aac"] > 0).min().values

    bulk["RoFr"] = bulk.Ro_h * bulk.Fr_h
    bulk["V∞³÷L"] = bulk.attrs["V∞"]**3 / bulk.L
    bulk["V∞²N∞"] = bulk.attrs["V∞"]**2 * np.sqrt(bulk.N2_inf)
    bulk["N∞³L²"] = np.sqrt(bulk.N2_inf)**3 * bulk.L**2
    #---

    #+++ Save Datasets
    #+++ Drop unnecessary vars
    xyza = xyza.drop_vars(["ūⱼūᵢ", "⟨uⱼuᵢ⟩ₜ", "⟨u′ⱼu′ᵢ⟩ₜ", "x_faa", "y_afa", "z_aaf"])
    xyia = xyia.drop_dims(("x_faa", "y_afa",))
    #---

    #+++ Save xyia
    outname = f"data_post/tafields_{simname}.nc"
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving results to {outname}...")
        xyia.to_netcdf(outname)
        print("Done!\n")
    xyii.close(); xyzi.close(); xyia.close()
    #---

    #+++ Save xyza
    outname = f"data_post/xyza_{simname}.nc"
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving results to {outname}...")
        xyza.to_netcdf(outname)
        print("Done!\n")
    xyza.close(); xyza.close()
    #---

    #+++ Save bulkstats
    outname = f"data_post/bulkstats_{simname}.nc"
    with ProgressBar(minimum=2, dt=5):
        print(f"Saving bulk results to {outname}...")
        bulk.to_netcdf(outname)
    #---
    #---
