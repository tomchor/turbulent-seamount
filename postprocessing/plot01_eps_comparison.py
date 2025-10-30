import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib.gridspec import GridSpec
from src.aux00_utils import open_simulation

# Set figure layout
plt.rcParams["figure.constrained_layout.use"] = True

#+++ Define simulation paths
simdata_path = "../simulations/data/"

# Parameters for the comparison
Ro_b = 0.1
Fr_b = 1  # Can be changed to compare different Fr_b values
resolution = "dz1"

# File paths for L=0 and L=0.8 simulations
simname_L0 = f"balanus_Ro_b{Ro_b}_Fr_b{Fr_b}_L0_FWHM500_{resolution}"
simname_L08 = f"balanus_Ro_b{Ro_b}_Fr_b{Fr_b}_L0.8_FWHM500_{resolution}"

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
var_names = ["∫⁵εₖdy", "∫⁵εₚdy"]

eps_k_L0 = xyzi_L0["∫⁵εₖdy"].isel(time=n_final)
eps_p_L0 = xyzi_L0["∫⁵εₚdy"].isel(time=n_final)

eps_k_L08 = xyzi_L08["∫⁵εₖdy"].isel(time=n_final)
eps_p_L08 = xyzi_L08["∫⁵εₚdy"].isel(time=n_final)

# Extract bottom_height for 3D plots (doesn"t have time dimension)
extent = 1.3 * FWHM
bottom_height_L0 = xyzi_L0["bottom_height"].pnsel(x=slice(-extent, +extent), y=slice(-extent, +extent))
bottom_height_L08 = xyzi_L08["bottom_height"].pnsel(x=slice(-extent, +extent), y=slice(-extent, +extent))
#---

#+++ Create 3x2 figure with GridSpec for height control
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.4, 0.7, 0.7])

# Create 3D axes for bathymetry (top row)
ax_3d_1 = fig.add_subplot(gs[0, 0], projection="3d")
ax_3d_2 = fig.add_subplot(gs[0, 1], projection="3d")

# Create 2D axes for dissipation plots (rows 2 and 3)
ax_eps_k_L0 = fig.add_subplot(gs[1, 0])
ax_eps_k_L08 = fig.add_subplot(gs[1, 1])
ax_eps_p_L0 = fig.add_subplot(gs[2, 0])
ax_eps_p_L08 = fig.add_subplot(gs[2, 1])

axes = np.array([[ax_eps_k_L0, ax_eps_k_L08],
                 [ax_eps_p_L0, ax_eps_p_L08]])

# Define common color range for each variable
eps_k_range = (1e-7, 5e-5)
eps_p_range = (5e-9, 1e-6)
#---

#+++ Plot 3D surface of bottom_height
ls = LightSource(azdeg=270, altdeg=45)
x = bottom_height_L0.x.values
y = bottom_height_L0.y.values
x, y = np.meshgrid(x, y)

# Plot bottom_height for L=0
z = bottom_height_L0.values

rgb = ls.shade(bottom_height_L0.values, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode="soft")
ax_3d_1.plot_surface(x, y, z, rstride=2, cstride=2, facecolors=rgb, linewidth=0, antialiased=False, shade=False)
ax_3d_1.set_xlabel("x [m]")
ax_3d_1.set_ylabel("y [m]")
ax_3d_1.set_zlabel("z [m]")
ax_3d_1.set_title(f"L/W = {params_L0["L"]}")
ax_3d_1.view_init(elev=25, azim=135)
ax_3d_1.set_box_aspect((1, 1, 0.3))

# Plot bottom_height for L=0.8
z = bottom_height_L08.values

rgb = ls.shade(bottom_height_L08.values, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode="soft")
ax_3d_2.plot_surface(x, y, z, rstride=2, cstride=2, facecolors=rgb, linewidth=0, antialiased=False, shade=False)
ax_3d_2.set_xlabel("x [m]")
ax_3d_2.set_ylabel("y [m]")
ax_3d_2.set_zlabel("z [m]")
ax_3d_2.set_title(f"L/W = {params_L08["L"]}")
ax_3d_2.view_init(elev=25, azim=135)
ax_3d_2.set_box_aspect((1, 1, 0.3))
#---
#--- Simplified plotting for ∫⁵εₖdy and ∫⁵εₚdy

plot_configs = [
    # (row_idx, col_idx, data, norm_range, label, cbar_label)
    (0, 0, eps_k_L0, eps_k_range, "∫⁵εₖdy", "∫⁵εₖdy [m³/s³]"),
    (0, 1, eps_k_L08, eps_k_range, "∫⁵εₖdy", None),
    (1, 0, eps_p_L0, eps_p_range, "∫⁵εₚdy", "∫⁵εₚdy [m³/s³]"),
    (1, 1, eps_p_L08, eps_p_range, "∫⁵εₚdy", None)
]

ims = []

for (row, col, data, norm_range, label, cbar_label) in plot_configs:
    ax = axes[row, col]
    im = data.plot(
        ax=ax, x="x_caa", y="z_aac",
        norm=LogNorm(vmin=norm_range[0], vmax=norm_range[1]),
        cmap="inferno", add_colorbar=False
    )
    ims.append(im)
    ax.set_xlabel("x [m]" if row == 1 else "")
    ax.set_ylabel("z [m]" if col == 0 else "")
    ax.set_title("")
    ax.text(
        0.05, 0.95, label, transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top", fontsize=12, fontweight="bold"
    )
    # Add colorbar if needed
    if cbar_label:
        cbar = plt.colorbar(im, ax=axes[row, :], orientation="vertical", pad=0.01)
        cbar.set_label(cbar_label, fontsize=10)
#---

#+++ Add overall title
title = f"Ro$_b$ = {params_L0["Ro_b"]}, Fr$_b$ = {params_L0["Fr_b"]}; "
fig.suptitle(title, fontsize=14, y=0.995)
#---

#+++ Save the plot
output_path = f"../figures/eps_comparison_L0_vs_L08_{resolution}.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")
#---
