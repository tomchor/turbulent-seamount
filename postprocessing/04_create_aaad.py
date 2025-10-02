import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from src.aux00_utils import (aggregate_parameters, normalize_unicode_names_in_dataset, integrate,
                             drop_faces, mask_immersed, gather_attributes_as_variables)
from dask.diagnostics.progress import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)
π = np.pi

print("Starting turbulent quantities script")

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

for j, config in enumerate(runs):
    simname = f"{simname_base}_" + aggregate_parameters(config, sep="_", prefix="")

    #+++ Load time-averaged datasets
    print(f"\nLoading time-averaged data for {simname}")
    xyza = xr.open_dataset(f"data/xyza.{simname}.nc", chunks="auto")
    xyzd = xr.open_dataset(f"data/xyzd.{simname}.nc", chunks="auto")

    xyza = xr.merge([xyza, xyzd])
    #---

    #+++ Normalize Unicode variable names
    xyza = normalize_unicode_names_in_dataset(xyza)
    #---

    #+++ Volume-average/integrate departure from base state
    xyza["ΔxΔyΔz"] = xyza["Δx_caa"] * xyza["Δy_aca"] * xyza["Δz_aac"]
    xyza["ΔxΔy"]   = xyza["Δx_caa"] * xyza["Δy_aca"]
    xyza["ΔyΔz"]   = xyza["Δy_aca"] * xyza["Δz_aac"]

    xyza = drop_faces(xyza, drop_coords=True).where(xyza.distance_condition_10meters, other=np.nan)
    aaad = xr.Dataset()
    aaad.attrs = xyza.attrs
    for var in ["ε̄ₚ", "ε̄ₖ", "SPR", "⟨Ek′⟩ₜ", "⟨w′b′⟩ₜ"]:
        int_buf = f"∭⁵{var}dV"
        aaad[int_buf] = integrate(xyza[var], dV=xyza.ΔxΔyΔz)
    #---

    #+++ Calculate turbulent quantities for the average turbulence region
    aaad["average_turbulence_mask"] = xyza["ε̄ₖ"] > 1e-11
    xyza["1"] = mask_immersed(xr.ones_like(xyza["ε̄ₖ"]), xyza.peripheral_nodes_ccc)
    for var in ["ε̄ₖ", "1"]:
        int_turb = f"∭ᵋ{var}dxdy"
        aaad[int_turb] = integrate(xyza[var], dV=xyza.ΔxΔyΔz.where(aaad.average_turbulence_mask))
    #---

    #+++ Calculate masked vertical averages of turbulent quantities
    print("Computing masked vertical averages...")
    # Create vertical average datasets with turbulence mask
    for var in ["ε̄ₚ", "ε̄ₖ", "SPR", "⟨Ek′⟩ₜ", "⟨w′b′⟩ₜ"]:
        # Masked vertical average (only where turbulence is significant)
        masked_var = xyza[var].where(aaad.average_turbulence_mask)
        masked_dz = xyza.ΔxΔyΔz.where(aaad.average_turbulence_mask)

        # Vertical average: sum(var * dz) / sum(dz) along z dimension
        vert_avg_name = f"⟨{var}⟩ᶻ"  # vertical average with turbulence mask
        aaad[vert_avg_name] = (masked_var * masked_dz).sum("z_aac") / masked_dz.sum("z_aac")
    #---

    #+++ Create aaad dataset
    aaad["U∞∬⟨Ek′⟩ₜdxdz"] = xyza["U∞∬⟨Ek′⟩ₜdydz"]
    aaad["⟨ε̄ₖ⟩ᵋ"]     = aaad["∭ᵋε̄ₖdxdy"] / aaad["∭ᵋ1dxdy"]
    aaad["Loᵋ"]       = 2*π * np.sqrt(aaad["⟨ε̄ₖ⟩ᵋ"] / aaad.N2_inf**(3/2))
    aaad["Δz̃"]        = aaad.Δz_min / aaad["Loᵋ"]

    aaad = gather_attributes_as_variables(aaad, ds_ref=xyza)
    #---

    #+++ Save aaad
    outname = f"data/aaad.{simname}.nc"
    with ProgressBar(minimum=2, dt=5):
        print(f"Saving aaad results to {outname}...")
        aaad.to_netcdf(outname)
        print("Done!\n")
    #---
