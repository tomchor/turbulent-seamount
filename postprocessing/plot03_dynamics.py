import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import pynanigans as pn
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.aux00_utils import open_simulation

# plt.rcParams["figure.constrained_layout.use"] = True

#+++ Load datasets
print("Reading datasets...")
simdata_path = "../simulations/data/"
postproc_path = "../postprocessing/data/"

resolution = "dz2"
snap_opts = dict(use_advective_periods=True,
                 unique_times=True,
                 squeeze=True,
                 load=False,
                 get_grid=False,
                 open_dataset_kwargs=dict(chunks="auto"))
avgd_opts = dict(unique_times=False,
                 load=False,
                 get_grid=False,
                 open_dataset_kwargs=dict(chunks="auto"))
xyzi_L00 = open_simulation(simdata_path + f"xyzi.seamount_Ro_h0.1_Fr_h1_L0_FWHM500_{resolution}.nc", **snap_opts)["PV"]
xyzd_L00 = open_simulation(postproc_path + f"xyzd.seamount_Ro_h0.1_Fr_h1_L0_FWHM500_{resolution}.nc", **avgd_opts)
aaad_L00 = open_simulation(postproc_path + f"aaad.seamount_Ro_h0.1_Fr_h1_L0_FWHM500_{resolution}.nc", **avgd_opts)
ds_L00 = xr.merge([xyzi_L00, xyzd_L00, aaad_L00])

xyzi_L08 = open_simulation(simdata_path + f"xyzi.seamount_Ro_h0.1_Fr_h1_L0.8_FWHM500_{resolution}.nc", **snap_opts)["PV"]
xyzd_L08 = open_simulation(postproc_path + f"xyzd.seamount_Ro_h0.1_Fr_h1_L0.8_FWHM500_{resolution}.nc", **avgd_opts)
aaad_L08 = open_simulation(postproc_path + f"aaad.seamount_Ro_h0.1_Fr_h1_L0.8_FWHM500_{resolution}.nc", **avgd_opts)
ds_L08 = xr.merge([xyzi_L08, xyzd_L08, aaad_L08])
#---

#+++ Create new variables and restrict volume
def prepare_ds(ds,
               x_slice = slice(-1.5*ds_L00.FWHM, np.inf),
               z_slice = slice(0, ds_L00.Lz - ds_L00.h_sponge),
               t_slice = 20):
    print("  Restricting domain and selecting time...")
    # Restrict domain first and select time immediately to minimize data
    ds = ds.sel(z_aac=z_slice, z_aaf=z_slice, x_caa=x_slice).sel(time=t_slice, method="nearest")

    print("  Computing z-averages efficiently...")
    # Use the mask directly without converting to NaN (much faster)
    mask = ds.distance_condition_5meters
    masked_volume = (mask * ds.Δz_aac).pnsum("z").load()

    ds["PV_zavg"] = (mask * ds["PV"] * ds.Δz_aac).pnsum("z") / masked_volume
    ds["Ro_zavg"] = (mask * ds["Ro"] * ds.Δz_aac).pnsum("z") / masked_volume
    ds["εₖ_zavg"] = (mask * ds["εₖ"] * ds.Δz_aac).pnsum("z") / masked_volume
    ds["εₚ_zavg"] = (mask * ds["εₚ"] * ds.Δz_aac).pnsum("z") / masked_volume
    ds["wb_zavg"] = (mask * ds["⟨w′b′⟩ₜ"] * ds.Δz_aac).pnsum("z") / masked_volume
    
    return ds

print("Preparing L=0 dataset...")
ds_L00 = prepare_ds(ds_L00)
print("Preparing L=0.8 dataset...")
ds_L08 = prepare_ds(ds_L08)
print("Data preparation complete!")
#---

#+++ Create 5x2 subplot grid
print("Creating subplot grid")
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 15), sharex=True, layout=None)
plt.subplots_adjust(wspace=0.05, hspace=0)
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
    # Data is already loaded and time-selected
    pv_data = ds.PV.pnsel(**xyi_sel_opts)
    im = pv_data.plot.imshow(ax=ax, x="x_caa",
                             cmap="RdBu_r",
                             add_colorbar=False,
                             rasterized=True,
                             vmin = -1.5*PV_inf,
                             vmax = +1.5*PV_inf)
    ax.set_title(f"L/FWHM = {L_str}")
    ax.set_xlabel("")
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    if i == 0:
        ax.set_ylabel("y [m]")
    else:
        ax.set_ylabel("")

# Add colorbar for PV row (aligned with rightmost panel)
divider = make_axes_locatable(axes[0, 1])
cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cbar_ax, label="PV", shrink=0.5)
#---

#+++ Plot Ro for both cases
print("Plotting Ro_zavg")

for i, (ds, L_str) in enumerate(datasets):
    ax = axes[1, i]
    # Data is already loaded and processed
    im = ds.Ro_zavg.plot.imshow(ax=ax, x="x_caa",
                                cmap="RdBu_r",
                                add_colorbar=False,
                                rasterized=True,
                                vmin = -0.4,
                                vmax = +0.4)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    if i == 0:
        ax.set_ylabel("y [m]")
    else:
        ax.set_ylabel("")

# Add colorbar for Ro row (aligned with rightmost panel)
divider = make_axes_locatable(axes[1, 1])
cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cbar_ax, label="Ro", shrink=0.5)
#---

#+++ Plot εₖ z-averaged
print("Plotting εₖ_zavg")
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[2, i]
    # Data is already loaded and processed
    im = ds["εₖ_zavg"].plot.imshow(ax=ax, x="x_caa",
                                   cmap="inferno",
                                   add_colorbar=False,
                                   rasterized=True,
                                   norm=LogNorm(vmin=1e-10, vmax=1e-8))
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    if i == 0:
        ax.set_ylabel("y [m]")
    else:
        ax.set_ylabel("")

# Add colorbar for εₖ row (aligned with rightmost panel)
divider = make_axes_locatable(axes[2, 1])
cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cbar_ax, label="εₖ" , shrink=0.5)
#---

#+++ Plot εₚ z-averaged
print("Plotting εₚ_zavg")
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[3, i]
    # Data is already loaded and processed
    im = ds["εₚ_zavg"].plot.imshow(ax=ax, x="x_caa",
                                   cmap="inferno",
                                   add_colorbar=False,
                                   rasterized=True,
                                   norm=LogNorm(vmin=1e-11, vmax=1e-9))
    ax.set_title("")
    ax.set_xlabel("x [m]")
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    if i == 0:
        ax.set_ylabel("y [m]")
    else:
        ax.set_ylabel("")

# Add colorbar for εₚ row (aligned with rightmost panel)
divider = make_axes_locatable(axes[3, 1])
cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cbar_ax, label="εₚ", shrink=0.5)
#---

#+++ Plot wb_zavg
print("Plotting wb_zavg")
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[4, i]
    # Data is already loaded and processed
    im = ds["wb_zavg"].plot.imshow(ax=ax, x="x_caa",
                                   cmap="RdBu_r",
                                   add_colorbar=False,
                                   rasterized=True,
                                   vmin=-4e-11,
                                   vmax=+4e-11)
    ax.set_title("")
    ax.set_xlabel("x [m]")
    ax.set_yticks(yticks)
    ax.set_aspect('equal')
    if i == 0:
        ax.set_ylabel("y [m]")
    else:
        ax.set_ylabel("")

# Add colorbar for wb_zavg row (aligned with rightmost panel)
divider = make_axes_locatable(axes[4, 1])
cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cbar_ax, label="wb", shrink=0.5)
#---

#+++ Save
print("Saving figure...")
fig.savefig(f"../figures/dynamics_comparison_{resolution}.png", dpi=300, bbox_inches="tight", pad_inches=0)
print("Done!")
#---