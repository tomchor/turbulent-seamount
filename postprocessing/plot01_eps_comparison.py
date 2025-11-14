import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib.gridspec import GridSpec
from src.aux00_utils import open_simulation
from src.aux02_plotting import letterize

# Set figure layout
plt.rcParams["figure.constrained_layout.use"] = True

#+++ Define simulation paths
simdata_path = "../simulations/data/"
simname_base = "balanus"

# Parameters for the comparison
Ro_b = 0.1
Fr_b = 1  # Can be changed to compare different Fr_b values
buffer = 5
resolution = 1

# File paths for L=0 and L=0.8 simulations
simname_L0 = f"{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L0_FWHM500_dz{resolution}"
simname_L08 = f"{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L0.8_FWHM500_dz{resolution}"

fpath_L0 = f"{simdata_path}xyzi.{simname_L0}.nc"
fpath_L08 = f"{simdata_path}xyzi.{simname_L08}.nc"

print(f"Reading NetCDF files:")
print(f"  L=0:   {fpath_L0}")
print(f"  L=0.8: {fpath_L08}")
#---

#+++ Load both datasets
dataset_opts = dict(use_advective_periods=True,
                    unique_times=True,
                    squeeze=True,
                    load=False,
                    get_grid=False,
                    open_dataset_kwargs=dict(chunks="auto"))

xyzi_L0 = open_simulation(fpath_L0, **dataset_opts)
xyzi_L08 = open_simulation(fpath_L08, **dataset_opts)

# Get metadata and parameters
params_L0 = {k: v for k, v in xyzi_L0.attrs.items()}
params_L08 = {k: v for k, v in xyzi_L08.attrs.items()}

# Extract grid coordinates and trim domain
H = params_L0["H"]
FWHM = params_L0["FWHM"]

# Trim domain similar to the Julia script
xyzi_L0 = xyzi_L0.sel(z_aac=slice(5, 1.2*H), x_caa=slice(None, 6*FWHM))
xyzi_L08 = xyzi_L08.sel(z_aac=slice(5, 1.2*H), x_caa=slice(None, 6*FWHM))

x_range = (xyzi_L0.x_caa.min().values, xyzi_L0.x_caa.max().values)
y_range = (xyzi_L0.y_aca.min().values, xyzi_L0.y_aca.max().values)
z_range = (xyzi_L0.z_aac.min().values, xyzi_L0.z_aac.max().values)

# Use the last time step
times = xyzi_L0.time.values
n_final = len(times) - 1
#---

#+++ Extract the two variables at the final time step
if buffer == 5:
    integration_bound = "⁵"
elif buffer == 10:
    integration_bound = "¹⁰"
else:
    raise ValueError(f"Buffer {buffer} wasn't calculated.")

eps_k_L0 = xyzi_L0[f"∫{integration_bound}εₖdy"].isel(time=n_final)
eps_p_L0 = xyzi_L0[f"∫{integration_bound}εₚdy"].isel(time=n_final)

eps_k_L08 = xyzi_L08[f"∫{integration_bound}εₖdy"].isel(time=n_final)
eps_p_L08 = xyzi_L08[f"∫{integration_bound}εₚdy"].isel(time=n_final)

# Extract bottom_height for 3D plots (doesn"t have time dimension)
extent = 1.3 * FWHM
bottom_height_L0 = xyzi_L0["bottom_height"].pnsel(x=slice(-extent, +extent), y=slice(-extent, +extent))
bottom_height_L08 = xyzi_L08["bottom_height"].pnsel(x=slice(-extent, +extent), y=slice(-extent, +extent))
#---

#+++ Create 3x2 figure with GridSpec for height control
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.4, 0.7, 0.7], hspace=0)

# Create 3D axes for bathymetry (top row)
ax_3d_1 = fig.add_subplot(gs[0, 0], projection="3d")
ax_3d_2 = fig.add_subplot(gs[0, 1], projection="3d")

# Create 2D axes for dissipation plots (rows 2 and 3)
ax_PV_L0 = fig.add_subplot(gs[1, 0])
ax_PV_L08 = fig.add_subplot(gs[1, 1])
ax_Ro_L0 = fig.add_subplot(gs[2, 0])
ax_Ro_L08 = fig.add_subplot(gs[2, 1])

axes = np.array([[ax_PV_L0, ax_PV_L08],
                 [ax_Ro_L0, ax_Ro_L08]])
#---

#+++ Plot 3D surface of bottom_height
ls = LightSource(azdeg=270, altdeg=45)
x = bottom_height_L0.x.values
y = bottom_height_L0.y.values
x, y = np.meshgrid(x, y)

# Plot bottom_height for L=0
z = bottom_height_L0.values

rgb = ls.shade(bottom_height_L0.values, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode="soft")
ax_3d_1.plot_surface(x, y, z, rstride=2, cstride=2, facecolors=rgb, linewidth=0, antialiased=False, shade=False, rasterized=True)
ax_3d_1.set_xlabel("x [m]")
ax_3d_1.set_ylabel("y [m]")
ax_3d_1.set_zlabel("z [m]")
ax_3d_1.set_title(f"L/W = {params_L0["L"]}")
ax_3d_1.view_init(elev=25, azim=135)
ax_3d_1.set_box_aspect((1, 1, 0.3))

# Plot bottom_height for L=0.8
z = bottom_height_L08.values

rgb = ls.shade(bottom_height_L08.values, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode="soft")
ax_3d_2.plot_surface(x, y, z, rstride=2, cstride=2, facecolors=rgb, linewidth=0, antialiased=False, shade=False, rasterized=True)
ax_3d_2.set_xlabel("x [m]")
ax_3d_2.set_ylabel("y [m]")
ax_3d_2.set_zlabel("z [m]")
ax_3d_2.set_title(f"L/W = {params_L08["L"]}")
ax_3d_2.view_init(elev=25, azim=135)
ax_3d_2.set_box_aspect((1, 1, 0.3))
#---

#+++ Load additional datasets for PV and Ro plots
import pynanigans as pn
from cmocean import cm

postproc_path = "../postprocessing/data/"

# Load datasets for both L values
avgd_opts = dict(unique_times=False,
                 load=False,
                 get_grid=False,
                 open_dataset_kwargs=dict(chunks="auto"))

xyzd_L0 = open_simulation(postproc_path + f"xyzd.{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L0_FWHM500_dz{resolution}.nc", **avgd_opts)
aaad_L0 = open_simulation(postproc_path + f"aaad.{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L0_FWHM500_dz{resolution}.nc", **avgd_opts).sel(buffer=10)
ds_L0 = xr.merge([xyzi_L0, xyzd_L0, aaad_L0])

xyzd_L08 = open_simulation(postproc_path + f"xyzd.{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L0.8_FWHM500_dz{resolution}.nc", **avgd_opts)
aaad_L08 = open_simulation(postproc_path + f"aaad.{simname_base}_Ro_b{Ro_b}_Fr_b{Fr_b}_L0.8_FWHM500_dz{resolution}.nc", **avgd_opts).sel(buffer=10)
ds_L08 = xr.merge([xyzi_L08, xyzd_L08, aaad_L08])

# Restrict domain and select time
t_slice = 20
x_slice = slice(-1.5*FWHM, np.inf)
z_slice = slice(0, H * 2 - ds_L0.h_sponge)  # Lz = 2*H from Lz_ratio

ds_L0 = ds_L0.sel(z_aac=z_slice, x_caa=x_slice).sel(time=t_slice, method="nearest")
ds_L08 = ds_L08.sel(z_aac=z_slice, x_caa=x_slice).sel(time=t_slice, method="nearest")
#---

#+++ Plot PV and Ro in rows 1 and 2
datasets_2d = [(ds_L0, "0"), (ds_L08, "0.8")]
yticks = [-500, 0, 500]

# Row 0: PV slice
row_idx = 0
print("Plotting PV")
for i, (ds, L_str) in enumerate(datasets_2d):
    ax = axes[row_idx, i]

    # Get PV data at z = H/3
    data = ds["PV"].pnsel(z=H / 3, method="nearest")
    bathymetry = ds.peripheral_nodes_ccc.pnsel(z=H / 3, method="nearest")

    vmin = -1.5 * ds.N2_inf * abs(ds.f_0)
    vmax = 1.5 * ds.N2_inf * abs(ds.f_0)

    im = data.plot.imshow(ax=ax, x="x_caa", cmap="RdBu_r",
                          vmin=vmin, vmax=vmax,
                          add_colorbar=False, rasterized=True)
    bathymetry.plot.imshow(ax=ax, cmap="Greys", vmin=0, vmax=1, origin="lower", alpha=0.25, zorder=2, add_colorbar=False)

    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.set_ylabel("y [m]" if i == 0 else "")
    ax.set_yticks(yticks)
    if i > 0:
        ax.set_yticklabels([])
    ax.set_aspect("equal")
    ax.set_title("")

# Add colorbar for PV
from mpl_toolkits.axes_grid1 import make_axes_locatable
cax = axes[row_idx, 1].inset_axes([0.75, 0.1, 0.03, 0.8],
                                   transform=axes[row_idx, 1].transAxes, clip_on=False)
cbar = plt.colorbar(im, cax=cax, orientation="vertical", label="Potential vorticity")

# Row 1: ⟨Ro⟩ᶻ
row_idx = 1
print("Plotting ⟨Ro⟩ᶻ")
for i, (ds, L_str) in enumerate(datasets_2d):
    ax = axes[row_idx, i]

    data = ds["⟨R̄o⟩ᶻ"]

    im = data.plot.imshow(ax=ax, x="x_caa", cmap="RdBu_r",
                          vmin=-0.4, vmax=0.4,
                          add_colorbar=False, rasterized=True)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]" if i == 0 else "")
    ax.set_yticks(yticks)
    if i > 0:
        ax.set_yticklabels([])
    ax.set_aspect("equal")
    ax.set_title("")

# Add colorbar for Ro
cax = axes[row_idx, 1].inset_axes([0.75, 0.1, 0.03, 0.8],
                                   transform=axes[row_idx, 1].transAxes, clip_on=False)
cbar = plt.colorbar(im, cax=cax, orientation="vertical", label="⟨Ro⟩ᶻ")
#---

#+++ Add overall title
title = f"Ro$_b$ = {params_L0["Ro_b"]}, Fr$_b$ = {params_L0["Fr_b"]}; "
fig.suptitle(title, fontsize=14, y=0.995)
letterize(fig.axes[:6], x=0.05, y=0.9, fontsize=12, bbox=dict(boxstyle="square", facecolor="white", alpha=0.8))
#---

#+++ Save the plot
output_path = f"../figures/{simname_base}_eps_comparison_L0_vs_L08_dz{resolution}_buffer{buffer}.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")
#---
