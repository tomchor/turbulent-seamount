import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from aux00_utils import open_simulation, adjust_times, aggregate_parameters
from aux02_plotting import RdBu_r, solar, balance
from matplotlib import pyplot as plt
xr.set_options(display_width=140, display_max_rows=30)
π = 2*np.pi

#+++ Define directory and simulation name
if basename(__file__) != "00_run_postproc.py":
    path = "simulations/data/"
    simname_base = "seamount"

    Rossby_numbers = cycler(Ro_h = [0.2,])
    Froude_numbers = cycler(Fr_h = [1.25])
    L              = cycler(L = [0, 300])

    resolutions    = cycler(dz = [2,])
    closures       = cycler(closure = ["DSM"])

    paramspace = Rossby_numbers * Froude_numbers * L
    configs    = resolutions * closures

    runs = paramspace * configs
#---

xyz_list = []
for j, config in enumerate(runs):
    simname = f"{simname_base}_" + aggregate_parameters(config, sep="_", prefix="")

    #+++ Open datasets
    print(f"\nOpening {simname} xyz")
    xyz = open_simulation(path+f"xyz.{simname}.nc",
                                    use_advective_periods = True,
                                    topology = simname[:3],
                                    squeeze = True,
                                    load = False,
                                    get_grid = False,
                                    open_dataset_kwargs = dict(chunks="auto"),
                                    )
    #---


    #+++ Calculate new variables and append
    xyz["∂U∂z"] = np.sqrt(xyz["∂u∂z"]**2 + xyz["∂v∂z"]**2)

    xyz_list.append(xyz)
    #---
print("Collected all datasets")

def mask_bathymetry(da, bathymetric_mask=xyz.peripheral_nodes_ccc):
    return da.where(np.logical_not(bathymetric_mask))

#+++ Plot
fig, axes = plt.subplots(nrows=2, ncols=len(xyz_list), figsize=(12, 6),
                         sharex=True, sharey=True, squeeze=True)

sel = dict(x_caa=300, time=np.inf, method="nearest")

V_inf = xyz_list[0].attrs["V∞"]

v_opts = dict(vmin=-1.5*V_inf, vmax=+1.5*V_inf, cmap=RdBu_r)
shear_opts = dict(vmin=0, vmax=0.03, cmap=solar)
for xyz, col in zip(xyz_list, axes.T):
    print("Plotting column")
    xyz = xyz.sel(**sel) 

    print("  Plotting v")
    mask_bathymetry(xyz.v, xyz.peripheral_nodes_ccc).pnplot(ax=col[0], x="y", **v_opts)

    print("  Plotting ∂ᶻU")
    mask_bathymetry(xyz["∂U∂z"], xyz.peripheral_nodes_ccc).pnplot(ax=col[1], x="y", **shear_opts)
#---

for ax in axes.flatten():
    ax.set_title("")
axes[0, 0].set_title("L = 0 m (unsmoothed)")
axes[0, 1].set_title("L = 300 m (smoothed)")
