import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from src.aux00_utils import (aggregate_parameters, normalize_unicode_names_in_dataset, integrate,
                             drop_faces, mask_immersed, gather_attributes_as_variables, condense)
from dask.diagnostics.progress import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)
π = np.pi

print("Starting turbulent quantities script")

#+++ Define directory and simulation name
if basename(__file__) != "00_run_postproc.py":
    path = "../simulations/data/"
    simname_base = "seamount"

    Rossby_numbers = cycler(Ro_b = [0.1])
    Froude_numbers = cycler(Fr_b = [1])
    L              = cycler(L = [0])

    resolutions    = cycler(dz = [4])
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

    xyza = drop_faces(xyza, drop_coords=True)
    aaad = xr.Dataset()
    aaad.attrs = xyza.attrs

    # Volume integrations with both 5m and 10m buffers
    for var in ["ε̄ₚ", "ε̄ₖ", "SPR", "⟨Ek′⟩ₜ", "⟨w′b′⟩ₜ"]:
        # 5-meter buffer integration
        int_buf_5m = f"∭⁵{var}dV"
        masked_dV_5m = xyza.ΔxΔyΔz.where(xyza.distance_condition_5meters)
        aaad[int_buf_5m] = integrate(xyza[var], dV=masked_dV_5m)

        # 10-meter buffer integration
        int_buf_10m = f"∭¹⁰{var}dV"
        masked_dV_10m = xyza.ΔxΔyΔz.where(xyza.distance_condition_10meters)
        aaad[int_buf_10m] = integrate(xyza[var], dV=masked_dV_10m)

        aaad = condense(aaad, [int_buf_5m, int_buf_10m], f"∭{var}dV", dimname="buffer", indices=[5, 10])
    #---

    #+++ yz integrals of selected variables using buffer masks
    for var in ["ε̄ₚ", "ε̄ₖ", "SPR", "⟨Ek′⟩ₜ", "⟨w′b′⟩ₜ"]:
        # 5-meter buffer yz integral
        int_yz_5m = f"∬⁵{var}dydz"
        masked_dydz_5m = xyza.ΔyΔz.where(xyza.distance_condition_5meters)
        aaad[int_yz_5m] = integrate(xyza[var], dV=masked_dydz_5m, dims=("y", "z"))

        # 10-meter buffer yz integral
        int_yz_10m = f"∬¹⁰{var}dydz"
        masked_dydz_10m = xyza.ΔyΔz.where(xyza.distance_condition_10meters)
        aaad[int_yz_10m] = integrate(xyza[var], dV=masked_dydz_10m, dims=("y", "z"))

        aaad = condense(aaad, [int_yz_5m, int_yz_10m], f"∬{var}dydz", dimname="buffer", indices=[5, 10])
    #---

    #+++ Calculate turbulent quantities for the average turbulence region
    aaad["average_turbulence_mask"] = xyza["ε̄ₖ"] > 1e-11
    xyza["1"] = mask_immersed(xr.ones_like(xyza["ε̄ₖ"]), xyza.peripheral_nodes_ccc)
    for var in ["ε̄ₖ", "1"]:
        int_turb = f"∭ᵋ{var}dV"
        masked_dV = xyza.ΔxΔyΔz.where(aaad.average_turbulence_mask)
        aaad[int_turb] = integrate(xyza[var], dV=masked_dV)
    #---

    #+++ Calculate masked vertical averages of turbulent quantities
    print("Computing masked vertical averages...")
    # Create vertical average datasets with both 5m and 10m buffers
    for var in ["ε̄ₚ", "ε̄ₖ", "R̄o", "SPR", "⟨Ek′⟩ₜ", "⟨w′b′⟩ₜ"]:
        # 5-meter buffer vertical average
        masked_dz_5m = xyza.Δz_aac.where(xyza.distance_condition_5meters)
        vert_avg_name_5m = f"⟨{var}⟩ᶻ⁵"  # vertical average with 5m buffer
        aaad[vert_avg_name_5m] = (xyza[var] * masked_dz_5m).sum("z_aac") / masked_dz_5m.sum("z_aac")

        # 10-meter buffer vertical average
        masked_dz_10m = xyza.Δz_aac.where(xyza.distance_condition_10meters)
        vert_avg_name_10m = f"⟨{var}⟩ᶻ¹⁰"  # vertical average with 10m buffer
        aaad[vert_avg_name_10m] = (xyza[var] * masked_dz_10m).sum("z_aac") / masked_dz_10m.sum("z_aac")

        aaad = condense(aaad, [vert_avg_name_5m, vert_avg_name_10m], f"⟨{var}⟩ᶻ", dimname="buffer", indices=[5, 10])
    #---

    #+++ Create aaad dataset
    aaad["U∞∬⟨Ek′⟩ₜdxdz"] = xyza["U∞∬⟨Ek′⟩ₜdydz"]
    aaad["⟨ε̄ₖ⟩ᵋ"]         = aaad["∭ᵋε̄ₖdV"] / aaad["∭ᵋ1dV"]
    aaad["Loᵋ"]           = 2*π * np.sqrt(aaad["⟨ε̄ₖ⟩ᵋ"] / aaad.N2_inf**(3/2))
    aaad["Δz̃"]            = aaad.Δz_min / aaad["Loᵋ"]

    aaad = gather_attributes_as_variables(aaad, ds_ref=xyza)
    #---

    #+++ Save aaad
    outname = f"data/aaad.{simname}.nc"
    with ProgressBar(minimum=2, dt=5):
        print(f"Saving aaad results to {outname}...")
        aaad.to_netcdf(outname)
        print("Done!\n")
    #---
