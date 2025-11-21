import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from src.aux00_utils import open_simulation, aggregate_parameters, gather_attributes_as_variables
from src.aux01_physfuncs import temporal_average
from dask.diagnostics import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)
π = np.pi

print("Starting aaaa dataset creation script")

#+++ Define directory and simulation name
if not basename(__file__).startswith("00_postproc_"):
    simdata_path = "../simulations/data/"
    simname_base = "balanus"

    Rossby_numbers = cycler(Ro_b = [0.1])
    Froude_numbers = cycler(Fr_b = [0.8])
    L              = cycler(L = [0, 0.05, 0.1, 0.2, 0.4, 0.8])
    FWHM           = cycler(FWHM = [500, 500, 500, 500, 500, 500])

    resolutions    = cycler(dz = [4])

    paramspace = Rossby_numbers * Froude_numbers * (L + FWHM)
    configs    = resolutions

    runs = paramspace * configs
#---

for j, config in enumerate(runs):
    simname = f"{simname_base}_" + aggregate_parameters(config, sep="_", prefix="")

    #+++ Open aaai dataset
    print(f"\nOpening {simname} aaai")
    aaai = open_simulation(simdata_path+f"aaai.{simname}.nc",
                           use_advective_periods = True,
                           squeeze = True,
                           load = False,
                           get_grid = False,
                           open_dataset_kwargs = dict(chunks="auto"),
                           )
    #---

    #+++ Trimming dataset
    t_slice_inclusive = slice(aaai.T_adv_spinup, np.inf) # For snapshots, we want to include t=T_adv_spinup
    aaai = aaai.sel(time=t_slice_inclusive)

    aaai["Ĥ"] = aaai.bottom_height.max()
    aaai = aaai.drop(["peripheral_nodes_ccc", "peripheral_nodes_ccf", "peripheral_nodes_cfc", "peripheral_nodes_fcc",
                      "x_caa", "x_faa", "y_aca", "y_afa", "z_aac", "z_aaf",
                      "Δx_caa", "Δx_faa", "Δy_aca", "Δy_afa", "Δz_aac", "Δz_aaf", "bottom_height"])
    #---

    #+++ Time-average aaai
    aaaa = temporal_average(aaai)
    #---

    #+++ Save aaaa
    outname = f"data/aaaa.{simname}.nc"
    aaaa = gather_attributes_as_variables(aaaa)
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving results to {outname}...")
        aaaa.to_netcdf(outname)
        print("Done!\n")
    aaaa.close()
    #---
