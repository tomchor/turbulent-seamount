import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from src.aux00_utils import (open_simulation, adjust_times, aggregate_parameters, gather_attributes_as_variables,
                             condense_velocities, condense_reynolds_stress_tensor_diagonal, integrate)
from src.aux01_physfuncs import (temporal_average, get_turbulent_Reynolds_stress_tensor_diagonal,
                                 get_buoyancy_production_rates, get_turbulent_kinetic_energy)
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)

print("Starting xyzd dataset creation script")

#+++ Define directory and simulation name
if basename(__file__) != "00_run_postproc.py":
    path = "../simulations/data/"
    simname_base = "seamount"

    Rossby_numbers = cycler(Ro_h = [0.1])
    Froude_numbers = cycler(Fr_h = [1])
    L              = cycler(L = [0])

    resolutions    = cycler(dz = [4])
    FWHM           = cycler(FWHM = [500])

    paramspace = Rossby_numbers * Froude_numbers * (L + FWHM)
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
    #---

    #+++ Get datasets ready
    #+++ Get rid of slight misalignment in times
    xyzi = adjust_times(xyzi, round_times=True)
    #---

    #+++ Preliminary definitions and checks
    print("Doing prelim checks")
    Δt = xyzi.time.diff("time").median()
    Δt_tol = Δt/100
    if np.all(xyzi.time.diff("time") > Δt_tol):
        print(Fore.GREEN + f"Δt is consistent for {simname}", Style.RESET_ALL)
    else:
        print(f"Δt is inconsistent for {simname}")
        print(np.count_nonzero(xyzi.time.diff("time") < Δt_tol), "extra time steps")

        tslice1 = slice(0.0, None, None)
        xyzi = xyzi.sel(time=tslice1)

        xyzi = xyzi.reindex(dict(time=np.arange(0, xyzi.time[-1]+1e-5, Δt)), method="nearest", tolerance=Δt/Δt_tol)
    #---

    #+++ Trim domain
    t_slice_inclusive = slice(xyzi.T_advective_spinup, np.inf) # For snapshots, we want to include t=T_advective_spinup
    t_slice_exclusive = slice(xyzi.T_advective_spinup + 0.01, np.inf) # For time-averaged outputs, we want to exclude t=T_advective_spinup
    x_slice = slice(None, np.inf)
    y_slice = slice(None, xyzi.y_afa[-2] - 2*xyzi.Δy_afa.values.max()) # Cut off last two points
    z_slice = slice(None, xyzi.z_aaf[-1] - xyzi.h_sponge) # Cut off top sponge

    xyzi = xyzi.sel(time=t_slice_inclusive, x_caa=x_slice, x_faa=x_slice, y_aca=y_slice, y_afa=y_slice, z_aac=z_slice, z_aaf=z_slice)
    #---

    #+++ Get mean variablesfor turbulent variables calculation
    xyzd = xyzi[["u", "v", "w", "uu", "vv", "ww", "b", "wb"]]
    xyzd = condense_velocities(xyzd, indices=indices)
    xyzd = condense_reynolds_stress_tensor_diagonal(xyzd, indices=indices)
    #---

    #+++ Condense and time-average tensors
    xyzd = temporal_average(xyzd)
    #---

    #+++ Get turbulent variables
    xyzd = get_turbulent_Reynolds_stress_tensor_diagonal(xyzd)
    xyzd = get_buoyancy_production_rates(xyzd)
    xyzd = get_turbulent_kinetic_energy(xyzd)
    #---

    #+++ Volume-average/integrate results so far
    xyzd["ΔxΔyΔz"] = xyzi["Δx_caa"] * xyzi["Δy_aca"] * xyzi["Δz_aac"]
    xyzd["ΔxΔy"] = xyzi["Δx_caa"] * xyzi["Δy_aca"]
    xyzd["ΔxΔz"] = xyzi["Δx_caa"] * xyzi["Δz_aac"]
    xyzd["ΔyΔz"] = xyzi["Δy_aca"] * xyzi["Δz_aac"]

    for var in ["⟨Ek′⟩ₜ", "⟨w′b′⟩ₜ"]:
        int_buf = f"∭⁵{var}dV"
        masked_var = xyzd[var].where(xyzi.distance_condition_10meters)
        xyzd[int_buf] = integrate(masked_var, dV=xyzd.ΔxΔyΔz)
    #---

    #+++ Calculate flux of turbulent kinetic energy out of the domain
    xyzd["U∞∬⟨Ek′⟩ₜdxdz"] = xyzd.attrs["U∞"] * integrate(xyzd["⟨Ek′⟩ₜ"], dV=xyzd.ΔxΔz, dims=["x", "z"]).pnsel(y=np.inf, method="nearest")
    #---

    #+++ Drop variables that are not needed and save xyzd
    xyzd = xyzd.drop(["ūᵢ", "b̄", "⟨uᵢuᵢ⟩ₜ", "⟨wb⟩ₜ"])

    outname = f"data/xyzd.{simname}.nc"
    xyzd = gather_attributes_as_variables(xyzd)
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving results to {outname}...")
        xyzd.to_netcdf(outname)
        print("Done!\n")
    xyzd.close()
    #---