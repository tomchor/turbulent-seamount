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
π = 2*np.pi

print("Starting energy transfer script")

#+++ Define directory and simulation name
if basename(__file__) != "00_run_postproc.py":
    path = "simulations/data/"
    simname_base = "seamount"

    Rossby_numbers = cycler(Ro_h = [0.2])
    Froude_numbers = cycler(Fr_h = [0.2])
    L              = cycler(L = [0, 300])

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
    t_slice_inclusive = slice(xyia.T_advective_spinup, np.inf) # For snapshots, we want to include t=T_advective_spinup
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
    #---

    #+++ Time average
    # Here ū and ⟨u⟩ₜ are interchangeable
    xyia = temporal_average(xyia).rename({"∭⁵εₖdV"  : "∭⁵ε̄ₖdV",
                                          "∭⁵εₚdV"  : "∭⁵ε̄ₚdV",
                                          "∭¹⁰εₖdV" : "∭¹⁰ε̄ₖdV",
                                          "∭¹⁰εₚdV" : "∭¹⁰ε̄ₚdV",
                                          "∭²⁰εₖdV" : "∭²⁰ε̄ₖdV",
                                          "∭²⁰εₚdV" : "∭²⁰ε̄ₚdV",
                                          })
    xyia = xyia.drop(["uᵢ", "uⱼuᵢ", "b", "wb"])

    xyza = temporal_average_xyza(xyza)
    #---

    #+++ Get turbulent Reynolds stress tensor
    xyza = get_turbulent_Reynolds_stress_tensor(xyza)
    #---

    #+++ Get shear production rates
    xyia = get_SPR(xyia)
    #---

    #+++ Get buoyancy production rates
    xyza = get_buoyancy_production_rates(xyza)
    #---

    #+++ Get TKE
    xyza = get_turbulent_kinetic_energy(xyza)
    #---

    #+++ Volume-average/integrate results so far
    def mask_immersed(da, bathymetric_mask=xyza.peripheral_nodes_ccc):
        return da.where(np.logical_not(bathymetric_mask))

    xyza["ΔxΔyΔz"] = xyza["Δx_caa"] * xyza["Δy_aca"] * xyza["Δz_aac"]
    xyza["ΔxΔz"]   = xyza["Δx_caa"] * xyza["Δz_aac"]
    xyia["ΔxΔy"]   = xyia["Δx_caa"] * xyia["Δy_aca"]

    def integrate(da, dV = None, dims=("x", "y", "z")):
        if dV is None:
            if dims == ("x", "y", "z"):
                dV = xyia["ΔxΔyΔz"]
            elif dims == ("x", "y"):
                dV =  xyia["ΔxΔy"]
        return (da * dV).pnsum(dims)

    distance_mask = xyza.altitude > 5 # meters
    for var in ["⟨Ek′⟩ₜ", "⟨w′b′⟩ₜ"]:
        int_buf = f"∭⁵{var}dV"
        xyza[int_buf] = integrate(xyza[var], dV=xyza.ΔxΔyΔz.where(distance_mask))

    xyza["average_turbulence_mask"] = xyza["ε̄ₖ"] > 1e-8
    xyza["1"] = mask_immersed(xr.ones_like(xyza["ε̄ₖ"]))
    for var in ["ε̄ₖ", "1"]:
        int_turb = f"∬ᵋ{var}dxdy"
        xyza[int_turb] = integrate(xyza[var], dV=xyza.ΔxΔyΔz.where(xyza.average_turbulence_mask), dims=("x", "y"))
    #---

    #+++ Drop unnecessary vars
    xyza = xyza.drop_vars(["ūⱼūᵢ", "⟨uⱼuᵢ⟩ₜ", "⟨u′ⱼu′ᵢ⟩ₜ", "x_faa", "y_afa", "z_aaf"])
    xyia = xyia.drop_dims(("x_faa", "y_afa",))
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
