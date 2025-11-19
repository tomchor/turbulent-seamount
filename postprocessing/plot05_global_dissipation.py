import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from src.aux02_plotting import letterize

plt.rcParams["figure.constrained_layout.use"] = True

seamounts = xr.open_dataset("../bathymetry/seamount_data.nc")
seamounts["Slope_Bu"] = seamounts.rossby_number / seamounts.froude_height
ρ = 1025
top_lat = 75

#+++ Dissipation calculations
# Dissipation according to linear formula (smooth seamounts)
seamounts["dissip_linear"] = 2e-2 * seamounts.Slope_Bu
seamounts["dissip_minimum"] = 2e-2
seamounts["dissip_piecewise"] = np.maximum(seamounts.dissip_linear, seamounts.dissip_minimum)

seamounts["mixing_linear"] = 2e-2 * seamounts.Slope_Bu
seamounts["mixing_quadratic"] = 2e-2 * seamounts.Slope_Bu**2

seamounts["dissip_scale"] = seamounts.velocity**3 / seamounts.height
seamounts["mixing_scale"] = seamounts.dissip_scale
seamounts["total_volume"] = seamounts.basal_radius_L**2 * seamounts.height

seamounts["total_dissip_smooth"] = seamounts.dissip_linear * seamounts.dissip_scale * seamounts.total_volume
seamounts["total_dissip_rough"] = seamounts.dissip_piecewise * seamounts.dissip_scale * seamounts.total_volume

seamounts["total_mixing_linear"] = seamounts.mixing_linear * seamounts.mixing_scale * seamounts.total_volume
seamounts["total_mixing_quadratic"] = seamounts.mixing_quadratic * seamounts.mixing_scale * seamounts.total_volume
#---

#+++ Convert to watts
seamounts["total_dissip_smooth"] = seamounts.total_dissip_smooth * ρ
seamounts.total_mixing_linear.attrs = dict(units="W", longer_name="KE dissipation")

seamounts["total_dissip_rough"] = seamounts.total_dissip_rough * ρ
seamounts.total_mixing_linear.attrs = dict(units="W", longer_name="Buoyancy mixing")

seamounts["total_mixing_linear"] = seamounts.total_mixing_linear * ρ
seamounts.total_mixing_linear.attrs = dict(units="W", longer_name="Buoyancy mixing")

seamounts["total_mixing_quadratic"] = seamounts.total_mixing_quadratic * ρ
seamounts.total_mixing_quadratic.attrs = dict(units="W", longer_name="Buoyancy mixing")
#---

#+++ Plot dissipation maps
#+++ Create figure with projections
fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0)

# Create left subplots with projection for heatmaps (2 rows)
ax_map_dissip_smooth = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_map_dissip_rough = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())

# Create right subplot for line plot (spans 2 rows)
ax_plot_dissip = fig.add_subplot(gs[0:2, 1])
#---

#+++ Plot heatmaps on the left column (2 rows)
fixed_options = dict(x="longitude", y="latitude", edgecolors="face", s=1, transform=ccrs.PlateCarree(), rasterized=True)

# Row 1: KE dissipation for smooth seamounts
gl1 = ax_map_dissip_smooth.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl1.top_labels = False
gl1.right_labels = False
gl1.bottom_labels = False

vmin_dissip = 1e4
vmax_dissip = 1e8
cmap_dissip = "YlOrRd"
im_dissip_smooth = seamounts.plot.scatter(ax=ax_map_dissip_smooth, hue="total_dissip_smooth", cmap=cmap_dissip, **fixed_options, norm=LogNorm(), vmin=vmin_dissip, vmax=vmax_dissip, add_colorbar=False)
cbar_dissip_smooth = plt.colorbar(im_dissip_smooth, ax=ax_map_dissip_smooth, orientation="vertical", pad=0.02, location="left", shrink=0.5)
cbar_dissip_smooth.set_label(r"Dissipation [W]")
ax_map_dissip_smooth.add_feature(cfeature.LAND, color="black", zorder=0)
ax_map_dissip_smooth.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax_map_dissip_smooth.set_ylim(-top_lat, top_lat)
ax_map_dissip_smooth.set_title("KE dissipation (smooth)")

# Row 2: KE dissipation for rough seamounts
gl2 = ax_map_dissip_rough.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl2.top_labels = False
gl2.right_labels = False

im_dissip_rough = seamounts.plot.scatter(ax=ax_map_dissip_rough, hue="total_dissip_rough", cmap=cmap_dissip, **fixed_options, norm=LogNorm(), vmin=vmin_dissip, vmax=vmax_dissip, add_colorbar=False)
cbar_dissip_rough = plt.colorbar(im_dissip_rough, ax=ax_map_dissip_rough, orientation="vertical", pad=0.02, location="left", shrink=0.5)
cbar_dissip_rough.set_label(r"Dissipation [W]")
ax_map_dissip_rough.add_feature(cfeature.LAND, color="black", zorder=0)
ax_map_dissip_rough.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax_map_dissip_rough.set_ylim(-top_lat, top_lat)
ax_map_dissip_rough.set_title("KE dissipation (rough)")
#---

#+++ Plot integrated line plot on the right column (spanning 2 rows)
# Bin seamounts data by latitude and longitude with 1 degree resolution
lat_bins = np.arange(-top_lat, top_lat + 1, 1)

# Create binned statistics for various quantities
print("Binning data")
binned_seamounts = seamounts.groupby_bins("latitude", lat_bins).sum()

# Right plot: KE dissipation line plot
binned_seamounts.total_dissip_smooth.plot(ax=ax_plot_dissip, y="latitude_bins", color="blue", linestyle="--", label="Smooth")
binned_seamounts.total_dissip_rough.plot(ax=ax_plot_dissip, y="latitude_bins", color="red", linestyle="--", label="Rough")
ax_plot_dissip.set_ylabel("Latitude (degrees)")
ax_plot_dissip.set_xlabel("KE dissipation [W]")
ax_plot_dissip.set_ylim(-top_lat, top_lat)
ax_plot_dissip.set_title("KE dissipation\nper degree of latitude")
ax_plot_dissip.grid(True)
ax_plot_dissip.legend(loc="upper right", borderaxespad=0, framealpha=0.9, edgecolor="black", fancybox=False)
#---

#+++ Save figure
letterize(fig.axes[:3], x=0.05, y=0.92, fontsize=12, bbox=dict(boxstyle="square", facecolor="white", alpha=0.9))
fig.savefig("../figures/global_dissipation.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
#---
#---

#+++ Plot mixing maps
#+++ Create figure with projections
fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0)

# Create left subplots with projection for heatmaps (2 rows)
ax_map_mixing_smooth = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_map_mixing_rough = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())

# Create right subplot for line plot (spans 2 rows)
ax_plot_mixing = fig.add_subplot(gs[0:2, 1])
#---

#+++ Plot heatmaps on the left column (2 rows)
fixed_options = dict(x="longitude", y="latitude", edgecolors="face", s=1, transform=ccrs.PlateCarree(), rasterized=True)

# Row 1: Buoyancy mixing for smooth seamounts
gl1 = ax_map_mixing_smooth.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl1.top_labels = False
gl1.right_labels = False
gl1.bottom_labels = False

vmin_mixing = 1e3
vmax_mixing = 1e7
cmap_mixing = "GnBu"
im_mixing_smooth = seamounts.plot.scatter(ax=ax_map_mixing_smooth, hue="total_mixing_linear", cmap=cmap_mixing, **fixed_options, norm=LogNorm(), vmin=vmin_mixing, vmax=vmax_mixing, add_colorbar=False)
cbar_mixing_smooth = plt.colorbar(im_mixing_smooth, ax=ax_map_mixing_smooth, orientation="vertical", pad=0.02, location="left", shrink=0.5)
cbar_mixing_smooth.set_label(r"Buoyancy mixing [W]")
ax_map_mixing_smooth.add_feature(cfeature.LAND, color="black", zorder=0)
ax_map_mixing_smooth.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax_map_mixing_smooth.set_ylim(-top_lat, top_lat)
ax_map_mixing_smooth.set_title("Buoyancy mixing (linear)")

# Row 2: Buoyancy mixing for rough seamounts
gl2 = ax_map_mixing_rough.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
gl2.top_labels = False
gl2.right_labels = False

im_mixing_rough = seamounts.plot.scatter(ax=ax_map_mixing_rough, hue="total_mixing_quadratic", cmap=cmap_mixing, **fixed_options, norm=LogNorm(), vmin=vmin_mixing, vmax=vmax_mixing, add_colorbar=False)
cbar_mixing_rough = plt.colorbar(im_mixing_rough, ax=ax_map_mixing_rough, orientation="vertical", pad=0.02, location="left", shrink=0.5)
cbar_mixing_rough.set_label(r"Buoyancy mixing [W]")
ax_map_mixing_rough.add_feature(cfeature.LAND, color="black", zorder=0)
ax_map_mixing_rough.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=1)
ax_map_mixing_rough.set_ylim(-top_lat, top_lat)
ax_map_mixing_rough.set_title("Buoyancy mixing (quadratic)")
#---

#+++ Plot integrated line plot on the right column (spanning 2 rows)
# Bin seamounts data by latitude and longitude with 1 degree resolution
lat_bins = np.arange(-top_lat, top_lat + 1, 1)

# Create binned statistics for various quantities
print("Binning data")
binned_seamounts = seamounts.groupby_bins("latitude", lat_bins).sum()

# Right plot: KE dissipation line plot
binned_seamounts.total_mixing_linear.plot(ax=ax_plot_mixing, y="latitude_bins", color="blue", linestyle="--", label="Smooth")
binned_seamounts.total_mixing_quadratic.plot(ax=ax_plot_mixing, y="latitude_bins", color="red", linestyle="--", label="Rough")
ax_plot_mixing.set_ylabel("Latitude (degrees)")
ax_plot_mixing.set_xlabel("Buoyancy mixing [W]")
ax_plot_mixing.set_ylim(-top_lat, top_lat)
ax_plot_mixing.set_title("Buoyancy mixing\nper degree of latitude")
ax_plot_mixing.grid(True)
ax_plot_mixing.legend(loc="upper right", borderaxespad=0, framealpha=0.9, edgecolor="black", fancybox=False)
#---

#+++ Save figure
letterize(fig.axes[:3], x=0.05, y=0.92, fontsize=12, bbox=dict(boxstyle="square", facecolor="white", alpha=0.9))
fig.savefig("../figures/global_mixing.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
#---
#---

#+++ Calculate difference
global_dissip_ratio = seamounts.total_dissip_smooth.sum("index") / seamounts.total_dissip_rough.sum("index")
global_mixing_ratio = seamounts.total_mixing_linear.sum("index") / seamounts.total_mixing_quadratic.sum("index")

south = seamounts.where(seamounts.latitude<-40)
south_dissip_ratio = south.total_dissip_smooth.sum("index") / south.total_dissip_rough.sum("index")
south_mixing_ratio = south.total_mixing_linear.sum("index") / south.total_mixing_quadratic.sum("index")

print(f"Global dissipation ratio: {global_dissip_ratio:.2f}")
print(f"South dissipation ratio: {south_dissip_ratio:.2f}")

print(f"Global mixing ratio: {global_mixing_ratio:.2f}")
print(f"South mixing ratio: {south_mixing_ratio:.2f}")
#---