import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from aux00_utils import aggregate_parameters, normalize_unicode_names_in_dataset
from aux01_physfuncs import (get_turbulent_Reynolds_stress_tensor, get_shear_production_rates,
                             get_buoyancy_production_rates, get_turbulent_kinetic_energy)
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)
π = np.pi

print("Starting turbulent quantities script")

#+++ Define directory and simulation name
if basename(__file__) != "00_run_postproc.py":
    path = "simulations/data/"
    simname_base = "seamount"

    Rossby_numbers = cycler(Ro_h = [0.2])
    Froude_numbers = cycler(Fr_h = [1.25])
    L              = cycler(L = [0])

    resolutions    = cycler(dz = [8])
    closures       = cycler(closure = ["AMD", "AMC", "CSM", "DSM", "NON"])
    closures       = cycler(closure = ["DSM"])

    paramspace = Rossby_numbers * Froude_numbers * L
    configs    = resolutions * closures

    runs = paramspace * configs
#---

for j, config in enumerate(runs):
    simname = f"{simname_base}_" + aggregate_parameters(config, sep="_", prefix="")

    #+++ Load time-averaged datasets
    print(f"\nLoading time-averaged data for {simname}")
    xyza = xr.open_dataset(f"data_post/xyza_{simname}.nc", chunks="auto")
    xyia = xr.open_dataset(f"data_post/xyia_{simname}.nc", chunks="auto")
    aaaa = xr.open_dataset(f"data_post/aaaa_{simname}.nc", chunks="auto")
    #---

    #+++ Normalize Unicode variable names
    xyza = normalize_unicode_names_in_dataset(xyza)
    xyia = normalize_unicode_names_in_dataset(xyia)
    aaaa = normalize_unicode_names_in_dataset(aaaa)
    #---

    #+++ Get turbulent variables
    xyza = get_turbulent_Reynolds_stress_tensor(xyza)
    xyza = get_shear_production_rates(xyza)
    xyza = get_buoyancy_production_rates(xyza)
    xyza = get_turbulent_kinetic_energy(xyza)

    xyia = get_turbulent_Reynolds_stress_tensor(xyia)
    xyia = get_shear_production_rates(xyia)
    xyia = get_buoyancy_production_rates(xyia)
    xyia = get_turbulent_kinetic_energy(xyia)
    #---

    #+++ Volume-average/integrate results so far
    xyza["ΔxΔyΔz"] = xyza["Δx_caa"] * xyza["Δy_aca"] * xyza["Δz_aac"]
    xyza["ΔxΔy"] = xyza["Δx_caa"] * xyza["Δy_aca"]
    xyza["ΔyΔz"] = xyza["Δy_aca"] * xyza["Δz_aac"]

    #+++ Aux functions
    def integrate(da, dV = None, dims=("x", "y", "z")):
        if dV is None:
            if dims == ("x", "y", "z"):
                dV = xyza["ΔxΔyΔz"]
            elif dims == ("x", "y"):
                dV =  xyza["ΔxΔy"]
            elif dims == ("y", "z"):
                dV =  xyza["ΔyΔz"]
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

    def mask_immersed(da, bathymetric_mask=xyza.peripheral_nodes_ccc):
        return da.where(np.logical_not(bathymetric_mask))
    #---

    xyza = drop_faces(xyza, drop_coords=True).where(xyza.distance_condition_10meters, other=np.nan)
    for var in ["ε̄ₚ", "ε̄ₖ", "⟨Ek′⟩ₜ", "⟨w′b′⟩ₜ", "SPR"]:
        int_buf = f"∭⁵{var}dV"
        xyza[int_buf] = integrate(xyza[var], dV=xyza.ΔxΔyΔz)

    xyza["average_turbulence_mask"] = xyza["ε̄ₖ"] > 1e-11
    xyza["1"] = mask_immersed(xr.ones_like(xyza["ε̄ₖ"]))
    for var in ["ε̄ₖ", "1"]:
        int_turb = f"∭ᵋ{var}dxdy"
        xyza[int_turb] = integrate(xyza[var], dV=xyza.ΔxΔyΔz.where(xyza.average_turbulence_mask))
    #---

    #+++ Create bulk dataset
    bulk = xr.Dataset()
    bulk.attrs = xyza.attrs

    bulk["∭⁵⟨Ek′⟩ₜdV"]  = xyza["∭⁵⟨Ek′⟩ₜdV"]
    bulk["∭⁵⟨w′b′⟩ₜdV"] = xyza["∭⁵⟨w′b′⟩ₜdV"]
    bulk["∭⁵SPRdxdy"]   = xyza["∭⁵SPRdV"]

    bulk["V∞∬⟨Ek′⟩ₜdxdz"] = xyza.attrs["V∞"] * integrate(xyza["⟨Ek′⟩ₜ"].pnsel(y=np.inf, method="nearest"), dV=xyza.Δx_caa*xyza.Δz_aac, dims=["x", "z"])

    bulk["∭ᵋε̄ₖdxdy"] = xyza["∭ᵋε̄ₖdxdy"]
    bulk["⟨ε̄ₖ⟩ᵋ"]    = xyza["∭ᵋε̄ₖdxdy"] / xyza["∭ᵋ1dxdy"]
    bulk["Loᵋ"]      = 2*π * np.sqrt(bulk["⟨ε̄ₖ⟩ᵋ"] / bulk.N2_inf**(3/2))
    bulk["Δz̃"]       = bulk.Δz_min / bulk["Loᵋ"]

    bulk["Ro_h"]     = bulk.Ro_h
    bulk["Fr_h"]     = bulk.Fr_h
    bulk["Slope_Bu"] = bulk.Slope_Bu
    bulk["α"]        = bulk.α

    bulk["Bu_h"] = bulk.Bu_h
    bulk["Γ"]    = bulk.Γ
    bulk["c_dz"] = bulk.c_dz

    bulk["f₀"]  = bulk.attrs["f₀"]
    bulk["N²∞"] = bulk.attrs["N²∞"]
    bulk["V∞"]  = bulk.attrs["V∞"]
    bulk["L"]   = bulk.L

    bulk["Δx_min"] = xyza["Δx_caa"].min()
    bulk["Δy_min"] = xyza["Δy_aca"].min()
    bulk["Δz_min"] = xyza["Δz_aac"].min()

    bulk["RoFr"]  = bulk.Ro_h * bulk.Fr_h
    bulk["V∞³÷L"] = bulk.attrs["V∞"]**3 / bulk.L
    bulk["V∞²N∞"] = bulk.attrs["V∞"]**2 * np.sqrt(bulk.N2_inf)
    bulk["N∞³L²"] = np.sqrt(bulk.N2_inf)**3 * bulk.L**2
    #---

    #+++ Save Datasets
    #+++ Drop unnecessary vars to speed up calculations
    xyza = xyza.drop_vars(["ūⱼūᵢ", "⟨uⱼuᵢ⟩ₜ", "⟨u′ⱼu′ᵢ⟩ₜ"])
    #---

    #+++ Save updated xyza with turbulent quantities
    outname = f"data_post/xyza_{simname}_turbulent.nc"
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving turbulent results to {outname}...")
        xyza.to_netcdf(outname)
        print("Done!\n")
    xyza.close()
    #---

    #+++ Save updated xyia with turbulent quantities
    outname = f"data_post/xyia_{simname}_turbulent.nc"
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving turbulent results to {outname}...")
        xyia.to_netcdf(outname)
        print("Done!\n")
    xyia.close()
    #---

    #+++ Save bulkstats
    outname = f"data_post/bulkstats_{simname}.nc"
    with ProgressBar(minimum=2, dt=5):
        print(f"Saving bulk results to {outname}...")
        bulk.to_netcdf(outname)
        print("Done!\n")
    #---
    #---