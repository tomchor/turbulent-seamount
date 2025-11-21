import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from src.aux00_utils import (open_simulation, adjust_times, aggregate_parameters, gather_attributes_as_variables,
                             integrate, normalize_unicode_names_in_dataset)
from src.aux01_physfuncs import (get_turbulent_Reynolds_stress_tensor,
                                 get_buoyancy_production_rates, get_turbulent_kinetic_energy,
                                 get_shear_production_rates)
from dask.diagnostics import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)

print("Starting xyzd dataset creation script")

#+++ Define directory and simulation name
if not basename(__file__).startswith("00_postproc_"):
    simdata_path = "../simulations/data/"
    simname_base = "balanus"

    Rossby_numbers = cycler(Ro_b = [0.1])
    Froude_numbers = cycler(Fr_b = [0.8])
    L              = cycler(L = [0])

    resolutions    = cycler(dz = [8])
    FWHM           = cycler(FWHM = [500])

    paramspace = Rossby_numbers * Froude_numbers * (L + FWHM)
    configs    = resolutions

    runs = paramspace * configs
#---

for j, config in enumerate(runs):
    simname = f"{simname_base}_" + aggregate_parameters(config, sep="_", prefix="")

    #+++ Open dataset
    print(f"\nOpening {simname} xyzi")
    xyzi = open_simulation(simdata_path+f"xyzi.{simname}.nc",
                           use_advective_periods = True,
                           squeeze = True,
                           load = False,
                           get_grid = False,
                           open_dataset_kwargs = dict(chunks="auto"),
                           )

    xyza = xr.open_dataset(f"data/xyza.{simname}.nc", chunks="auto")
    xyza = normalize_unicode_names_in_dataset(xyza)
    indices = xyza.i.values
    #---

    #+++ Get dataset ready
    xyzi = adjust_times(xyzi, round_times=True)
    xyzi = xyzi.reindex_like(xyza)

    xyzd = xyza[["b̄", "⟨wb⟩ₜ", "ūᵢ", "⟨uⱼuᵢ⟩ₜ", "∂ⱼūᵢ", "distance_condition_5meters", "distance_condition_10meters", "damping_rate"]]
    #---

    #+++ Get turbulent variables
    xyzd = get_turbulent_Reynolds_stress_tensor(xyzd)
    xyzd = get_buoyancy_production_rates(xyzd)
    xyzd = get_turbulent_kinetic_energy(xyzd)
    xyzd = get_shear_production_rates(xyzd)
    #---

    #+++ Calculate flux of turbulent kinetic energy out of the domain
    xyzd["ΔyΔz"] = xyzi["Δy_aca"] * xyzi["Δz_aac"]
    xyzd["U∞∬⟨Ek′⟩ₜdydz"] = xyzd.attrs["U∞"] * integrate(xyzd["⟨Ek′⟩ₜ"], dV=xyzd.ΔyΔz, dims=["y", "z"]) # Advective flux of TKE out of the domain

    xyzd["ε̄ₛ"] = xyzi.damping_rate * xyzd["⟨Ek′⟩ₜ"] # Dissipation rate of TKE due to damping (proxy for propagation upwards)
    #---

    #+++ Drop variables that are not needed and save xyzd
    xyzd = xyzd.drop(["⟨wb⟩ₜ"])
    outname = f"data/xyzd.{simname}.nc"
    xyzd = gather_attributes_as_variables(xyzd)
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving results to {outname}...")
        xyzd.to_netcdf(outname)
        print("Done!\n")
    xyzd.close()
    #---