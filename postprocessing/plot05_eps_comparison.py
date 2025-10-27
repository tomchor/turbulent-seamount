import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
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
bottom_height_L0 = xyzi_L0["bottom_height"].pnsel(x=slice(-FWHM, +FWHM))
bottom_height_L08 = xyzi_L08["bottom_height"].pnsel(x=slice(-FWHM, +FWHM))
#---

#+++ Create 3x2 figure
fig = plt.figure(figsize=(14, 10))

# Create 3D axes for bathymetry (top row)
ax_3d_1 = fig.add_subplot(3, 2, 1, projection="3d")
ax_3d_2 = fig.add_subplot(3, 2, 2, projection="3d")

# Create 2D axes for dissipation plots (rows 2 and 3)
axes = np.array([[fig.add_subplot(3, 2, 3), fig.add_subplot(3, 2, 4)],
                 [fig.add_subplot(3, 2, 5), fig.add_subplot(3, 2, 6)]])

# Define common color range for each variable
eps_k_range = (1e-7, 1e-4)
eps_p_range = (1e-8, 1e-6)
#---

#+++ Plot 3D surface of bottom_height
ls = LightSource(azdeg=270, altdeg=45)
x = bottom_height_L0.x.values
y = bottom_height_L0.y_aca.values
x, y = np.meshgrid(x, y)

# Plot bottom_height for L=0
z = bottom_height_L0.values

rgb = ls.shade(bottom_height_L0.values, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode="soft")
ax_3d_1.plot_surface(x, y, z, rstride=2, cstride=2, facecolors=rgb, linewidth=0, antialiased=False, shade=False)
ax_3d_1.set_xlabel("x [m]")
ax_3d_1.set_ylabel("y [m]")
ax_3d_1.set_zlabel("z [m]")
ax_3d_1.set_title(f"L/FWHM = {params_L0["L"]}")
ax_3d_1.view_init(elev=25, azim=135)
ax_3d_1.set_box_aspect((1, 1, 0.3))

# Plot bottom_height for L=0.8
z = bottom_height_L08.values

rgb = ls.shade(bottom_height_L08.values, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode="soft")
ax_3d_2.plot_surface(x, y, z, rstride=2, cstride=2, facecolors=rgb, linewidth=0, antialiased=False, shade=False)
ax_3d_2.set_xlabel("x [m]")
ax_3d_2.set_ylabel("y [m]")
ax_3d_2.set_zlabel("z [m]")
ax_3d_2.set_title(f"L/FWHM = {params_L08["L"]}")
ax_3d_2.view_init(elev=25, azim=135)
ax_3d_2.set_box_aspect((1, 1, 0.3))
#---

#+++ Plot ∫⁵εₖdy for L=0 (second row, left)
ax = axes[0, 0]
im1 = eps_k_L0.plot(ax=ax, x="x_caa", y="z_aac",
                    norm=LogNorm(vmin=eps_k_range[0], vmax=eps_k_range[1]),
                    cmap="inferno", add_colorbar=False)
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_title(f"L/FWHM = {params_L0["L"]}")
ax.text(0.05, 0.95, "∫⁵εₖdy", transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top", fontsize=12, fontweight="bold")

# Plot ∫⁵εₖdy for L=0.8 (second row, right)
ax = axes[0, 1]
im2 = eps_k_L08.plot(ax=ax, x="x_caa", y="z_aac",
                     norm=LogNorm(vmin=eps_k_range[0], vmax=eps_k_range[1]),
                     cmap="inferno", add_colorbar=False)
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_title(f"L/FWHM = {params_L08["L"]}")
ax.text(0.05, 0.95, "∫⁵εₖdy", transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top", fontsize=12, fontweight="bold")

# Add colorbar for first row (∫⁵εₖdy)
cbar1 = plt.colorbar(im1, ax=axes[0, :], orientation="vertical", pad=0.01)
cbar1.set_label("∫⁵εₖdy [m³/s³]", fontsize=10)

# Plot ∫⁵εₚdy for L=0 (third row, left)
ax = axes[1, 0]
im3 = eps_p_L0.plot(ax=ax, x="x_caa", y="z_aac",
                    norm=LogNorm(vmin=eps_p_range[0], vmax=eps_p_range[1]),
                    cmap="inferno", add_colorbar=False)
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.text(0.05, 0.95, "∫⁵εₚdy", transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top", fontsize=12, fontweight="bold")

# Plot ∫⁵εₚdy for L=0.8 (third row, right)
ax = axes[1, 1]
im4 = eps_p_L08.plot(ax=ax, x="x_caa", y="z_aac",
                     norm=LogNorm(vmin=eps_p_range[0], vmax=eps_p_range[1]),
                     cmap="inferno", add_colorbar=False)
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.text(0.05, 0.95, "∫⁵εₚdy", transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top", fontsize=12, fontweight="bold")

# Add colorbar for second row (∫⁵εₚdy)
cbar2 = plt.colorbar(im3, ax=axes[1, :], orientation="vertical", pad=0.01)
cbar2.set_label("∫⁵εₚdy [m³/s³]", fontsize=10)
#---

#+++ Add overall title
title = f"Roₕ = {params_L0["Ro_b"]}, Frₕ = {params_L0["Fr_b"]}; " \
        f"Time = {times[n_final]:.1f}"
fig.suptitle(title, fontsize=14, y=0.995)
#---

#+++ Save the plot
output_path = "../figures/eps_comparison_L0_vs_L08.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")
#---
