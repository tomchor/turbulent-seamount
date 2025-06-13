import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
from cycler import cycler
import pynanigans as pn
from aux00_utils import open_simulation, adjust_times, aggregate_parameters
from aux02_plotting import RdBu_r, solar, balance, inferno
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
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

xyzi_list = []
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
    #---


    #+++ Calculate new variables and append
    xyzi["Uz²"] = xyzi["∂u∂z"]**2 + xyzi["∂v∂z"]**2
    xyzi["∫εₚdx"] = xyzi["εₚ"].pnintegrate("x")

    xyzi_list.append(xyzi)
    #---
print("Collected all datasets")

def mask_immersed(da, bathymetric_mask=xyzi.peripheral_nodes_ccc):
    return da.where(np.logical_not(bathymetric_mask))

#+++ Plot
fig, axes = plt.subplots(nrows=3, ncols=len(xyzi_list), figsize=(12, 9),
                         sharex=True, sharey=True, squeeze=True)

sel = dict(x_caa=300, time=np.inf, method="nearest")

V_inf = xyzi_list[0].attrs["V∞"]

common_opts = dict(x="y", rasterized=True)
w_opts = dict(vmin=-2e-1*V_inf, vmax=+2e-1*V_inf, cmap=RdBu_r)
v_opts = dict(vmin=-1.5*V_inf, vmax=+1.5*V_inf, cmap=RdBu_r)
uw_opts = dict(vmin=-3e-3, vmax=+3e-3, cmap=RdBu_r)
shear_opts = dict(vmin=0, vmax=0.03, cmap=solar)
ε_opts = dict(norm=LogNorm(clip=True), vmin=1e-8, vmax=1e-6, cmap=inferno)
for xyzi, col in zip(xyzi_list, axes.T):
    print("Plotting column")
    xyzi = xyzi.sel(**sel) 

    print("  Plotting w")
    mask_immersed(xyzi.w, xyzi.peripheral_nodes_ccc).pnplot(ax=col[0], **(common_opts | w_opts))

    print("  Plotting uw")
    mask_immersed(xyzi["uw"], xyzi.peripheral_nodes_ccc).pnplot(ax=col[1], **(common_opts | uw_opts))
    #print("  Plotting ∂u∂z")
    #mask_immersed(xyzi["∂u∂z"], xyzi.peripheral_nodes_ccc).pnplot(ax=col[1], **(common_opts | shear_opts))

    print("  Plotting εₚ")
    xyzi["∫εₚdx"].pnplot(ax=col[2], **(common_opts | ε_opts))
#---

for ax in axes.flatten():
    ax.set_title("")
axes[0, 0].set_title("L = 0 m (unsmoothed)")
axes[0, 1].set_title("L = 300 m (smoothed)")
