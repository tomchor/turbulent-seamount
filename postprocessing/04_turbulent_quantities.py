import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from src.aux00_utils import (aggregate_parameters, normalize_unicode_names_in_dataset, integrate,
                             drop_faces, mask_immersed, gather_attributes_as_variables)
from src.aux01_physfuncs import (get_turbulent_Reynolds_stress_tensor, get_shear_production_rates)
from colorama import Fore, Back, Style
from dask.diagnostics.progress import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)
π = np.pi

print("Starting turbulent quantities script")

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

for j, config in enumerate(runs):
    simname = f"{simname_base}_" + aggregate_parameters(config, sep="_", prefix="")

    #+++ Load time-averaged datasets
    print(f"\nLoading time-averaged data for {simname}")
    xyza = xr.open_dataset(f"data/xyza.{simname}.nc", chunks="auto")
    xyia = xr.open_dataset(f"data/xyia.{simname}.nc", chunks="auto")
    xyzd = xr.open_dataset(f"data/xyzd.{simname}.nc", chunks="auto")
    #---

    #+++ Normalize Unicode variable names
    xyza = normalize_unicode_names_in_dataset(xyza)
    xyia = normalize_unicode_names_in_dataset(xyia)
    #---

    #+++ Get turbulent variables (only those not already calculated in xyzd)
    xyza = get_turbulent_Reynolds_stress_tensor(xyza)
    xyza = get_shear_production_rates(xyza)

    xyia = get_turbulent_Reynolds_stress_tensor(xyia)
    xyia = get_shear_production_rates(xyia)
    #---

    #+++ Volume-average/integrate results so far
    xyza["ΔxΔyΔz"] = xyza["Δx_caa"] * xyza["Δy_aca"] * xyza["Δz_aac"]
    xyza["ΔxΔy"] = xyza["Δx_caa"] * xyza["Δy_aca"]
    xyza["ΔyΔz"] = xyza["Δy_aca"] * xyza["Δz_aac"]

    xyza = drop_faces(xyza, drop_coords=True).where(xyza.distance_condition_10meters, other=np.nan)
    for var in ["ε̄ₚ", "ε̄ₖ", "SPR"]:
        int_buf = f"∭⁵{var}dV"
        xyza[int_buf] = integrate(xyza[var], dV=xyza.ΔxΔyΔz)

    xyza["average_turbulence_mask"] = xyza["ε̄ₖ"] > 1e-11
    xyza["1"] = mask_immersed(xr.ones_like(xyza["ε̄ₖ"]), xyza.peripheral_nodes_ccc)
    for var in ["ε̄ₖ", "1"]:
        int_turb = f"∭ᵋ{var}dxdy"
        xyza[int_turb] = integrate(xyza[var], dV=xyza.ΔxΔyΔz.where(xyza.average_turbulence_mask))
    #---

    #+++ Create bulk dataset
    bulk = xr.Dataset()
    bulk.attrs = xyza.attrs

    # Use pre-calculated values from xyzd dataset
    bulk["∭⁵⟨Ek′⟩ₜdV"]  = xyzd["∭⁵⟨Ek′⟩ₜdV"]
    bulk["∭⁵⟨w′b′⟩ₜdV"] = xyzd["∭⁵⟨w′b′⟩ₜdV"]
    bulk["∭⁵SPRdxdy"]   = xyza["∭⁵SPRdV"]

    bulk["U∞∬⟨Ek′⟩ₜdxdz"] = xyzd["U∞∬⟨Ek′⟩ₜdxdz"]

    bulk["∭ᵋε̄ₖdxdy"] = xyza["∭ᵋε̄ₖdxdy"]
    bulk["⟨ε̄ₖ⟩ᵋ"]    = xyza["∭ᵋε̄ₖdxdy"] / xyza["∭ᵋ1dxdy"]
    bulk["Loᵋ"]      = 2*π * np.sqrt(bulk["⟨ε̄ₖ⟩ᵋ"] / bulk.N2_inf**(3/2))
    bulk["Δz̃"]       = bulk.Δz_min / bulk["Loᵋ"]

    bulk = gather_attributes_as_variables(bulk, ds_ref=xyza)
    #---

    #+++ Save turbstats
    outname = f"data/turbstats_{simname}.nc"
    with ProgressBar(minimum=2, dt=5):
        print(f"Saving bulk results to {outname}...")
        bulk.to_netcdf(outname)
        print("Done!\n")
    #---
