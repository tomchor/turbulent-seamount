import numpy as np
from src.aux02_plotting import letterize
import xarray as xr
from matplotlib import pyplot as plt
import pynanigans as pn
from matplotlib.colors import LogNorm
from cmocean import cm
from src.aux00_utils import open_simulation

# plt.rcParams["figure.constrained_layout.use"] = True

#+++ Load datasets
print("Reading datasets...")
simdata_path = "../simulations/data/"
postproc_path = "../postprocessing/data/"

resolution = "dz1"
buffer = 5
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
xyzi_L00 = open_simulation(simdata_path  + f"xyzi.balanus_Ro_b0.1_Fr_b1_L0_FWHM500_{resolution}.nc", **snap_opts)[["PV", "Ro", "Δz_aac"]]
xyzd_L00 = open_simulation(postproc_path + f"xyzd.balanus_Ro_b0.1_Fr_b1_L0_FWHM500_{resolution}.nc", **avgd_opts)
aaad_L00 = open_simulation(postproc_path + f"aaad.balanus_Ro_b0.1_Fr_b1_L0_FWHM500_{resolution}.nc", **avgd_opts).sel(buffer=10)
ds_L00 = xr.merge([xyzi_L00, xyzd_L00, aaad_L00])

xyzi_L08 = open_simulation(simdata_path  + f"xyzi.balanus_Ro_b0.1_Fr_b1_L0.8_FWHM500_{resolution}.nc", **snap_opts)[["PV", "Ro", "Δz_aac"]]
xyzd_L08 = open_simulation(postproc_path + f"xyzd.balanus_Ro_b0.1_Fr_b1_L0.8_FWHM500_{resolution}.nc", **avgd_opts)
aaad_L08 = open_simulation(postproc_path + f"aaad.balanus_Ro_b0.1_Fr_b1_L0.8_FWHM500_{resolution}.nc", **avgd_opts).sel(buffer=10)
ds_L08 = xr.merge([xyzi_L08, xyzd_L08, aaad_L08])
#---

#+++ Create new variables and restrict volume
def prepare_ds(ds,
               x_slice = slice(-1.5*ds_L00.FWHM, np.inf),
               z_slice = slice(0, ds_L00.Lz - ds_L00.h_sponge),
               t_slice = 20):
    print("  Restricting domain and selecting time...")
    # Restrict domain first and select time immediately to minimize data
    ds = ds.sel(z_aac=z_slice, x_caa=x_slice).sel(time=t_slice, method="nearest")

    print("  Calculating shear production components...")
    ds["⟨VSPR⟩ᶻ"] = ds["⟨SPR⟩ᶻ"].sel(j=3)
    ds["⟨HSPR⟩ᶻ"] = ds["⟨SPR⟩ᶻ"].sel(j=[1, 2]).sum("j")
    ds["⟨TSPR⟩ᶻ"] = ds["⟨SPR⟩ᶻ"].sum("j")

    return ds

print("Preparing L=0 dataset...")
ds_L00 = prepare_ds(ds_L00)
print("Preparing L=0.8 dataset...")
ds_L08 = prepare_ds(ds_L08)
print("Data preparation complete!")
#---

#+++ Create subplot grid
print("Creating subplot grid")
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12), sharex=True, layout=None)
plt.subplots_adjust(wspace=0.05, hspace=0)

datasets = [(ds_L00, "0"), (ds_L08, "0.8")]
yticks = [-500, 0, 500]
#---

#+++ Plot PV for both cases
print("Plotting PV")
xyi_sel_opts = dict(z=ds_L00.H / 3, method="nearest")

PV_inf = ds_L00.N2_inf * abs(ds_L00.f_0)
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
    if i == 0:
        ax.set_ylabel("y [m]")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.set_aspect('equal')

# Add colorbar for PV row inside the right panel
cax = axes[0, 1].inset_axes([0.8, 0.1, 0.03, 0.8], transform=axes[0, 1].transAxes, clip_on=False)
cbar = plt.colorbar(im, cax=cax, orientation="vertical", label="PV")
#---

#+++ Plot Ro for both cases
var_name = "⟨R̄o⟩ᶻ"
print(f"Plotting {var_name}")
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[1, i]
    # Data is already loaded and processed
    im = ds[var_name].plot.imshow(ax=ax, x="x_caa",
                                  cmap="RdBu_r",
                                  add_colorbar=False,
                                  rasterized=True,
                                  vmin = -0.4,
                                  vmax = +0.4)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_yticks(yticks)
    if i == 0:
        ax.set_ylabel("y [m]")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.set_aspect('equal')

# Add colorbar for Ro row inside the right panel
cax = axes[1, 1].inset_axes([0.8, 0.1, 0.03, 0.8], transform=axes[1, 1].transAxes, clip_on=False)
cbar = plt.colorbar(im, cax=cax, orientation="vertical", label="Ro")
#---

#+++ Plot εₖ z-averaged
var_name = "⟨ε̄ₖ⟩ᶻ"
print(f"Plotting {var_name}")
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[2, i]
    # Data is already loaded and processed
    im = ds[var_name].plot.imshow(ax=ax, x="x_caa",
                                  cmap="inferno",
                                  add_colorbar=False,
                                  rasterized=True,
                                  norm=LogNorm(vmin=1e-11, vmax=1e-8))
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_yticks(yticks)
    if i == 0:
        ax.set_ylabel("y [m]")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.set_aspect('equal')

# Add colorbar for εₖ row inside the right panel, set font and tick color to white
cax = axes[2, 1].inset_axes([0.8, 0.1, 0.03, 0.8], transform=axes[2, 1].transAxes, clip_on=False)
cbar = plt.colorbar(im, cax=cax, orientation="vertical", label="⟨ε̄ₖ⟩ᶻ")
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
cbar.ax.yaxis.label.set_color('white')
#---

#+++ Plot εₚ z-averaged
var_name = "⟨ε̄ₚ⟩ᶻ"
print(f"Plotting {var_name}")
for i, (ds, L_str) in enumerate(datasets):
    ax = axes[3, i]
    # Data is already loaded and processed
    im = ds[var_name].plot.imshow(ax=ax, x="x_caa",
                                  cmap="inferno",
                                  add_colorbar=False,
                                  rasterized=True,
                                  norm=LogNorm(vmin=1e-11, vmax=1e-8))
    ax.set_title("")
    ax.set_xlabel("x [m]")
    ax.set_yticks(yticks)
    if i == 0:
        ax.set_ylabel("y [m]")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.set_aspect('equal')

# Add colorbar for εₚ row inside the right panel
cax = axes[3, 1].inset_axes([0.8, 0.1, 0.03, 0.8], transform=axes[3, 1].transAxes, clip_on=False)
cbar = plt.colorbar(im, cax=cax, orientation="vertical", label="⟨ε̄ₚ⟩ᶻ")
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
cbar.ax.yaxis.label.set_color('white')
#---

#+++ Save
letterize(axes.flatten(), x=0.05, y=0.9, fontsize=9)
print("Saving figure...")
fig.savefig(f"../figures/dynamics_comparison_{resolution}.png", dpi=300, bbox_inches="tight", pad_inches=0)
print("Done!")
#---