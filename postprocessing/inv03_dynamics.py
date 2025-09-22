import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import pynanigans as pn
from matplotlib.colors import LogNorm
from src.aux00_utils import open_simulation

# plt.rcParams["figure.constrained_layout.use"] = True

#+++ Load datasets
print("Reading xyzi datasets...")
path = "../simulations/data/"

resolution = "dz4"
grid00, ds_L00 = open_simulation(path + f"xyzi.seamount_Ro_h0.1_Fr_h1_L0_FWHM500_{resolution}.nc",
                                 use_advective_periods=True,
                                 squeeze=True,
                                 load=False,
                                 get_grid=True,
                                 open_dataset_kwargs=dict(chunks="auto"))

grid08, ds_L08 = open_simulation(path + f"xyzi.seamount_Ro_h0.1_Fr_h1_L0.8_FWHM500_{resolution}.nc",
                                 use_advective_periods=True,
                                 squeeze=True,
                                 load=False,
                                 get_grid=True,
                                 open_dataset_kwargs=dict(chunks="auto"))
#---

#+++ Create new variables and restrict volume
z_slice = slice(0, ds_L00.Lz - ds_L00.h_sponge) # Exclude sponge layer
x_slice = slice(-ds_L00.FWHM, np.inf)

def prepare_ds(ds, grid):
    # Restrict domain first to reduce data size
    ds = ds.sel(z_aac=z_slice, z_aaf=z_slice, x_caa=x_slice)
    
    # Select only the time we need to avoid processing all times
    ds = ds.sel(time=20, method="nearest")
    
    # Create derived variables
    ds["dUdz"] = np.sqrt(ds["∂u∂z"]**2 + ds["∂v∂z"]**2)

    print("Loading mask to disk")
    ds["mask"] = ds.distance_condition_5meters.where(ds.distance_condition_5meters, other=np.nan).load()
    ds["Ro_zavg"] = grid.average(ds.mask * ds.Ro, "z")
    ds["εₖ_zavg"] = grid.average(ds.mask * ds["εₖ"], "z")
    ds["εₚ_zavg"] = grid.average(ds.mask * ds["εₚ"], "z")
    return ds

ds_L00 = prepare_ds(ds_L00, grid00)
ds_L08 = prepare_ds(ds_L08, grid08)
#---

#+++ Create 4x2 subplot grid
print("Creating subplot grid")
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 14), sharex=True, layout=None)
plt.subplots_adjust(wspace=0, hspace=-10)
#---

#+++ Plot PV for both cases
print("Plotting PV")
xyi_sel_opts = dict(z=ds_L00.H / 3, method="nearest")
datasets = [(ds_L00, "0"), (ds_L08, "0.8")]

# Common y-axis ticks for all panels
yticks = [-500, 0, 500]

PV_inf = ds_L00.N2_inf * ds_L00.f_0
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[0, i]
    # Data is already time-selected, so no need to select time again
    pv_data = ds.PV.pnsel(**xyi_sel_opts)
    im = pv_data.pnplot(ax=ax, x="x", y="y",
                        cmap="RdBu_r", robust=True,
                        add_colorbar=False,
                        rasterized=True,
                        vmin = -1.5*PV_inf,
                        vmax = +1.5*PV_inf,)
    ax.set_title(f"L/FWHM = {L_str}")
    ax.set_xlabel("")
    ax.set_yticks(yticks)
    if i == 0:
        ax.set_ylabel("y [m]")

# Add colorbar for PV row
cbar_ax = fig.add_axes([0.92, 0.775, 0.02, 0.2])  # [left, bottom, width, height]
plt.colorbar(im, cax=cbar_ax, label="PV")
#---

#+++ Plot Ro for both cases
print("Plotting Ro_zavg")

for i, (ds, L_str) in enumerate(datasets):
    ax = axes[1, i]
    # Data is already time-selected, so just plot directly
    im = ds.Ro_zavg.pnplot(ax=ax, x="x", y="y",
                           cmap="RdBu_r", robust=True,
                           add_colorbar=False,
                           rasterized=True,
                           vmin = -0.4,
                           vmax = +0.4)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_yticks(yticks)
    if i == 0:
        ax.set_ylabel("y [m]")

# Add colorbar for Ro row
cbar_ax = fig.add_axes([0.92, 0.575, 0.02, 0.2])  # [left, bottom, width, height]
plt.colorbar(im, cax=cbar_ax, label="Ro")
#---

#+++ Plot εₖ z-averaged
print("Plotting εₖ_zavg")
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[2, i]
    # Data is already time-selected, so just plot directly
    im = ds["εₖ_zavg"].pnplot(ax=ax, x="x", y="y",
                              cmap="inferno", robust=True,
                              add_colorbar=False,
                              rasterized=True,
                              norm=LogNorm(vmin=1e-10, vmax=1e-8))
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_yticks(yticks)
    if i == 0:
        ax.set_ylabel("y [m]")

# Add colorbar for εₖ row
cbar_ax = fig.add_axes([0.92, 0.375, 0.02, 0.2])  # [left, bottom, width, height]
plt.colorbar(im, cax=cbar_ax, label="εₖ")
#---

#+++ Plot εₚ z-averaged
print("Plotting εₚ_zavg")
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[3, i]
    # Data is already time-selected, so just plot directly
    im = ds["εₚ_zavg"].pnplot(ax=ax, x="x", y="y",
                              cmap="inferno", robust=True,
                              add_colorbar=False,
                              rasterized=True,
                              norm=LogNorm(vmin=1e-11, vmax=1e-9))
    ax.set_title("")
    ax.set_xlabel("x [m]")
    ax.set_yticks(yticks)
    if i == 0:
        ax.set_ylabel("y [m]")

# Add colorbar for εₚ row
cbar_ax = fig.add_axes([0.92, 0.05, 0.01, 0.2])  # [left, bottom, width, height]
plt.colorbar(im, cax=cbar_ax, label="εₚ")
#---

#+++ Save
fig.savefig("../figures/dynamics_comparison.png", dpi=300, bbox_inches="tight")
#---